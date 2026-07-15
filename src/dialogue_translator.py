"""
Dialogue Translator — Sprint 35.

Objectif :
  Depuis le Sprint 35, une seule niche/histoire est produite chaque jour et
  alimente 2 vidéos (anglais + français) qui partagent EXACTEMENT le même
  script (scènes, description visuelle, style, caméra, son) — seules les
  répliques parlées (dialogue/narration) changent de langue. Ce moteur
  traduit UNIQUEMENT les répliques d'un Script déjà généré (en anglais) vers
  une langue cible, sans jamais reformuler scene.description/transition
  (qui restent toujours en anglais, cf. llm_script_generator.py).

Contrat :
  DialogueTranslator.translate(script: Script, target_language: str = "fr") -> Script

  Le Script retourné a EXACTEMENT les mêmes scenes/description/transition/
  personnage que l'original — seuls `Dialogue.replique` (traduit) et
  `ScriptScene.duration_seconds` (recalculé via estimate_scene_duration(),
  le français étant en général ~15-20% plus long à l'oral que l'anglais)
  changent, plus `Script.language`/`estimated_duration`.

  Si le LLM échoue malgré les tentatives : fallback déterministe qui renvoie
  les répliques D'ORIGINE (non traduites), jamais d'exception — cohérent
  avec la philosophie fail-soft des autres moteurs (LLMImageGenerator,
  LLMAnimationGenerator). Le fallback est tracé dans `Script.metadata`
  (jamais sérialisé dans final_script.json — Sprint 31.1) et loggé en WARNING.
"""

import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm import LLMMessage, build_llm
from src.script_engine import Dialogue, Script, ScriptScene, cap_dialogues_to_duration, estimate_scene_duration

logger = logging.getLogger(__name__)


_DEEPSEEK_TRANSLATION_MODEL = os.environ.get("DEEPSEEK_TRANSLATION_MODEL", "deepseek-chat")

_LANGUAGE_NAMES = {
    "fr": "French", "en": "English", "es": "Spanish", "pt": "Portuguese",
    "de": "German", "it": "Italian", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean",
}


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "dialogue_translation_system_prompt.txt").read_text(encoding="utf-8")


# ── Extraction/nettoyage JSON robustes (mêmes garanties que les autres moteurs) ──

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.replace("```json", "").replace("```", "").strip()


def _isolate_json_object(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return text.strip()
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:].strip()


def _clean_json_text(text: str) -> str:
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text.strip()


def _extract_json(text: str) -> str:
    text = text.strip()
    text = _strip_think_tags(text)
    text = _strip_code_fence(text)
    text = _isolate_json_object(text)
    text = _clean_json_text(text)
    return text.strip()


_JSON_REPAIR_INSTRUCTION = (
    "Le JSON precedent est invalide ou ne respecte pas exactement le nombre de scenes/repliques attendu.\n"
    "Corrige UNIQUEMENT le JSON.\n"
    "Ne produis aucun texte supplementaire.\n"
    "Respecte exactement le meme nombre de scenes et de repliques par scene que l'entree, dans le meme ordre."
)


class _TranslationJsonError(RuntimeError):
    """Erreur typée pour classifier précisément la cause d'un échec de traduction."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"[{reason}] {detail}" if detail else reason)
        self.reason = reason


class DialogueTranslator:
    """
    Traduit les répliques d'un Script vers une langue cible, en conservant
    strictement scenes/description/transition/personnage à l'identique.
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1536,
        max_retries: int = 2,
    ) -> None:
        self._provider_name = provider_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._provider = None  # lazy init
        self._stats: Dict[str, Any] = {
            "llm_calls": 0, "llm_success": 0, "llm_failures": 0, "fallbacks": 0,
            "total_time_ms": 0, "total_cost_usd": 0.0,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def _resolve_model(self) -> Optional[str]:
        if self._model is not None:
            return self._model
        provider = self._provider_name or (
            "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else None
        )
        if provider == "deepseek":
            return _DEEPSEEK_TRANSLATION_MODEL
        return None

    # ── Point d'entrée public ────────────────────────────────────────────────

    def translate(self, script: Script, target_language: str = "fr") -> Script:
        last_reason = "unknown"
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._try_translate_llm(script, target_language)
            except Exception as exc:
                last_reason = getattr(exc, "reason", "unknown") or "unknown"
                logger.warning(
                    "DialogueTranslator — tentative %d/%d échouée (raison=%s) : %s",
                    attempt, self._max_retries, last_reason, exc,
                )
                self._stats["llm_failures"] += 1

        logger.warning(
            "DialogueTranslator — fallback vers les répliques NON traduites "
            "(langue cible=%s, raison=%s) — la version %s de cette vidéo "
            "restera dans la langue d'origine.",
            target_language, last_reason, target_language,
        )
        self._stats["fallbacks"] += 1
        return self._fallback_script(script, target_language, reason=last_reason)

    # ── Logique LLM ──────────────────────────────────────────────────────────

    def _try_translate_llm(self, script: Script, target_language: str) -> Script:
        if self._provider is None:
            self._provider = build_llm(provider=self._provider_name, model=self._resolve_model())
            logger.info("DialogueTranslator utilise %s / %s", self._provider.name, self._provider.model)

        user_prompt = self._build_user_prompt(script, target_language)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        response, elapsed_ms = self._call_llm(messages)
        self._raise_if_api_error(response)

        try:
            data = self._parse_and_validate(response, script)
        except _TranslationJsonError as first_err:
            logger.warning(
                "DialogueTranslator — JSON invalide (raison=%s) — tentative de correction.",
                first_err.reason,
            )
            repair_messages = messages + [
                LLMMessage(role="assistant", content=response.content[:4000]),
                LLMMessage(role="user", content=_JSON_REPAIR_INSTRUCTION),
            ]
            repair_response, repair_elapsed_ms = self._call_llm(repair_messages)
            self._raise_if_api_error(repair_response)
            data = self._parse_and_validate(repair_response, script)

        self._stats["llm_success"] += 1
        return self._build_translated_script(script, data, target_language)

    def _call_llm(self, messages: List[LLMMessage]):
        start = time.time()
        self._stats["llm_calls"] += 1
        response = self._provider.generate(
            messages, temperature=self._temperature, max_tokens=self._max_tokens, json_mode=True,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        self._stats["total_time_ms"] += elapsed_ms
        self._stats["total_cost_usd"] += response.cost_usd
        return response, elapsed_ms

    @staticmethod
    def _raise_if_api_error(response: Any) -> None:
        if response.finish_reason != "error":
            return
        reason = "timeout" if "timeout" in response.content.lower() else "api_error"
        raise _TranslationJsonError(reason, response.content[:200])

    @staticmethod
    def _parse_and_validate(response: Any, script: Script) -> Dict[str, Any]:
        content = (response.content or "").strip()
        if not content:
            raise _TranslationJsonError("empty_response", "réponse vide")

        json_str = _extract_json(content)
        incomplete = getattr(response, "finish_reason", None) == "length"
        if not json_str:
            raise _TranslationJsonError(
                "json_incomplete" if incomplete else "json_invalid",
                "aucun objet JSON isolable dans la réponse",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise _TranslationJsonError(
                "json_incomplete" if incomplete else "json_invalid", str(exc)
            ) from exc

        DialogueTranslator._validate_json_structure(data, script)
        return data

    @staticmethod
    def _validate_json_structure(data: Dict[str, Any], script: Script) -> None:
        scenes = data.get("scenes")
        if not isinstance(scenes, list):
            raise _TranslationJsonError("validation_failed", "Champ 'scenes' manquant ou invalide")
        if len(scenes) != len(script.scenes):
            raise _TranslationJsonError(
                "validation_failed",
                f"Nombre de scènes traduites ({len(scenes)}) != nombre de scènes du script ({len(script.scenes)})",
            )
        for orig_scene, translated in zip(script.scenes, scenes):
            dialogues = translated.get("dialogues")
            if not isinstance(dialogues, list) or len(dialogues) != len(orig_scene.dialogues):
                raise _TranslationJsonError(
                    "validation_failed",
                    f"Scène {orig_scene.scene.number} : nombre de répliques traduites incorrect",
                )
            for d in dialogues:
                if not isinstance(d, dict) or not str(d.get("replique", "")).strip():
                    raise _TranslationJsonError(
                        "validation_failed",
                        f"Scène {orig_scene.scene.number} : réplique traduite vide",
                    )

    # ── Construction du prompt ─────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(script: Script, target_language: str) -> str:
        language_name = _language_name(target_language)
        lines: List[str] = [
            f"Translate ONLY the spoken dialogues below into {language_name.upper()}.",
            f"Title (context only, do not translate): {script.title}",
            "",
            "=== DIALOGUES TO TRANSLATE (by scene, in order) ===",
        ]
        for scene in script.scenes:
            lines.append(f"Scene {scene.scene.number}:")
            for d in scene.dialogues:
                lines.append(f'  {d.personnage}: "{d.replique}"')
        lines += ["", f"Return the {language_name} translation JSON now."]
        return "\n".join(lines)

    # ── Construction du Script traduit ───────────────────────────────────────

    @staticmethod
    def _build_translated_script(script: Script, data: Dict[str, Any], target_language: str) -> Script:
        translated_scenes: List[ScriptScene] = []
        for orig_scene, translated in zip(script.scenes, data["scenes"]):
            new_dialogues = [
                Dialogue(personnage=orig_d.personnage, replique=str(t["replique"]).strip())
                for orig_d, t in zip(orig_scene.dialogues, translated["dialogues"])
            ]
            # Sprint 37 — le français est souvent plus long à l'oral que
            # l'anglais : une traduction fidèle peut dépasser le budget de
            # 6s/scène même quand l'original le respectait. Même garantie
            # APPLIQUÉE qu'à la génération (voir cap_dialogues_to_duration).
            new_dialogues = cap_dialogues_to_duration(new_dialogues)
            duration = estimate_scene_duration(new_dialogues)
            translated_scenes.append(
                dataclasses.replace(orig_scene, dialogues=new_dialogues, duration_seconds=duration)
            )
        return dataclasses.replace(
            script,
            scenes=translated_scenes,
            language=target_language,
            estimated_duration=sum(s.duration_seconds for s in translated_scenes),
        )

    @staticmethod
    def _fallback_script(script: Script, target_language: str, reason: str) -> Script:
        """
        Fallback déterministe : conserve les répliques D'ORIGINE (non
        traduites) plutôt que de lever une exception — jamais de vidéo non
        produite à cause d'un échec de traduction. Le fallback est tracé
        dans `metadata` (jamais sérialisé dans final_script.json).
        """
        fallback_metadata = dict(script.metadata)
        fallback_metadata["translation_fallback_reason"] = reason
        fallback_metadata["translation_fallback_target_language"] = target_language
        return dataclasses.replace(script, metadata=fallback_metadata)
