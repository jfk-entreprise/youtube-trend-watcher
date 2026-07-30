"""
Story Narrator — Sprint 38.1.

Objectif :
  story.txt (voir ProductionPackageBuilder) doit se lire comme une histoire
  racontée, jamais comme un simple enchaînement scène par scène des
  répliques déjà écrites. Ce moteur transforme le script FR déjà généré et
  traduit (LLMScriptGenerator + DialogueTranslator) en un récit fluide en
  français — aucun nouveau fait, aucune nouvelle réplique : uniquement une
  reformulation narrative de ce qui est déjà écrit.

Contrat :
  StoryNarrator.narrate(script: Script) -> str

  Si le LLM échoue malgré les tentatives : fallback déterministe qui
  enchaîne simplement les répliques déjà écrites (comportement historique,
  Sprint 37.4), jamais d'exception — cohérent avec la philosophie fail-soft
  des autres moteurs (LLMImageGenerator, LLMAnimationGenerator,
  DialogueTranslator).

Moteur totalement indépendant : ne dépend que du contrat Script en entrée,
n'importe aucun autre moteur créatif.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm import LLMMessage, build_llm
from src.script_engine import Script

logger = logging.getLogger(__name__)


_DEEPSEEK_STORY_MODEL = os.environ.get("DEEPSEEK_STORY_MODEL", "deepseek-chat")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT = (_PROMPTS_DIR / "story_narrator_system_prompt.txt").read_text(encoding="utf-8")


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
                return text[start : i + 1]
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
    "Le JSON precedent est invalide.\n"
    "Corrige UNIQUEMENT le JSON.\n"
    "Ne produis aucun texte supplementaire.\n"
    "Respecte exactement le schema demande : {\"story\": \"...\"}."
)


class _StoryJsonError(RuntimeError):
    """Erreur typée pour classifier précisément la cause d'un échec de narration."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"[{reason}] {detail}" if detail else reason)
        self.reason = reason


class StoryNarrator:
    """
    Transforme un Script (répliques déjà écrites/traduites) en un récit
    fluide en français, prêt à être lu comme story.txt.
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.8,
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
            return _DEEPSEEK_STORY_MODEL
        return None

    # ── Point d'entrée public ────────────────────────────────────────────────

    def narrate(self, script: Script) -> str:
        """
        Retourne le récit fluide en français pour `script`. Ne lève jamais
        d'exception : retombe sur un enchaînement simple des répliques déjà
        écrites si le LLM échoue malgré les tentatives.
        """
        last_reason = "unknown"
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._try_narrate_llm(script)
            except Exception as exc:
                last_reason = getattr(exc, "reason", "unknown") or "unknown"
                logger.warning(
                    "StoryNarrator — tentative %d/%d échouée (raison=%s) : %s",
                    attempt, self._max_retries, last_reason, exc,
                )
                self._stats["llm_failures"] += 1

        logger.warning(
            "StoryNarrator — fallback vers l'enchaînement simple des répliques "
            "(raison=%s) — story.txt restera lisible mais moins narratif.",
            last_reason,
        )
        self._stats["fallbacks"] += 1
        return self._fallback_story(script)

    # ── Logique LLM ──────────────────────────────────────────────────────────

    def _try_narrate_llm(self, script: Script) -> str:
        if self._provider is None:
            self._provider = build_llm(provider=self._provider_name, model=self._resolve_model())
            logger.info("StoryNarrator utilise %s / %s", self._provider.name, self._provider.model)

        user_prompt = self._build_user_prompt(script)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        response, elapsed_ms = self._call_llm(messages)
        self._raise_if_api_error(response)

        try:
            data = self._parse_and_validate(response)
        except _StoryJsonError as first_err:
            logger.warning(
                "StoryNarrator — JSON invalide (raison=%s) — tentative de correction.",
                first_err.reason,
            )
            repair_messages = messages + [
                LLMMessage(role="assistant", content=response.content[:4000]),
                LLMMessage(role="user", content=_JSON_REPAIR_INSTRUCTION),
            ]
            repair_response, repair_elapsed_ms = self._call_llm(repair_messages)
            self._raise_if_api_error(repair_response)
            data = self._parse_and_validate(repair_response)

        self._stats["llm_success"] += 1
        return str(data["story"]).strip()

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
        raise _StoryJsonError(reason, response.content[:200])

    @staticmethod
    def _parse_and_validate(response: Any) -> Dict[str, Any]:
        content = (response.content or "").strip()
        if not content:
            raise _StoryJsonError("empty_response", "réponse vide")

        json_str = _extract_json(content)
        incomplete = getattr(response, "finish_reason", None) == "length"
        if not json_str:
            raise _StoryJsonError(
                "json_incomplete" if incomplete else "json_invalid",
                "aucun objet JSON isolable dans la réponse",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise _StoryJsonError(
                "json_incomplete" if incomplete else "json_invalid", str(exc)
            ) from exc

        if not isinstance(data.get("story"), str) or not data["story"].strip():
            raise _StoryJsonError("validation_failed", "Champ 'story' manquant ou vide")

        return data

    # ── Construction du prompt ─────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(script: Script) -> str:
        lines: List[str] = [
            "Weave the spoken lines below into ONE flowing French story, in order.",
            f"Title (context only, do not repeat it verbatim as a heading): {script.title}",
            "",
            "=== SPOKEN LINES (by scene, in order) ===",
        ]
        for scene in script.scenes:
            lines.append(f"Scene {scene.scene.number}:")
            for d in scene.dialogues:
                lines.append(f'  {d.personnage}: "{d.replique}"')
        lines += ["", "Return the story JSON now."]
        return "\n".join(lines)

    # ── Fallback déterministe ────────────────────────────────────────────────

    @staticmethod
    def _fallback_story(script: Script) -> str:
        """
        Enchaîne simplement les répliques déjà écrites (comportement
        historique, Sprint 37.4) — jamais d'exception, jamais de story.txt
        vide même si le LLM échoue.
        """
        paragraphs = [scene.narration_text for scene in script.scenes if scene.narration_text]
        return "\n\n".join(paragraphs)
