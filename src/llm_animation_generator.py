"""
LLM Animation Generator — Contrat AnimationPrompt universel via LLM, optimisé
pour les générateurs vidéo texte-vers-vidéo de dernière génération (Google
Veo, Kling, Hailuo, Runway, Luma, Pika) (Sprint 25).

Objectif :
  Ne PAS générer de vidéo. Produire, pour chaque scène, un prompt d'animation
  extrêmement riche qui explique comment une image fixe (déjà produite par
  LLMImageGenerator) prend vie pendant quelques secondes : mouvement de
  caméra, mouvement du sujet, réaction de l'environnement, évolution de la
  lumière, effets, ambiance sonore, transition.

Contrat de sortie — AnimationPrompt (Sprint 25) :
    {
      "camera_motion": "...", "subject_motion": "...",
      "environment_motion": "...", "lighting_changes": "...",
      "effects": "...", "sound_design": "...", "transition": "...",
      "duration": 0, "prompt": "...",
      "metadata": {
        "goal": "...", "emotion": "...", "pace": "...",
        "provider": "...", "model": "...", "time_ms": 0, "cost_usd": 0.0,
      },
    }

Architecture (identique à LLMImageGenerator — Sprint 24.2/24.3/24.5) :
  1. Le LLM (via build_llm()) agit comme un réalisateur de cinéma —
     raisonnement d'abord (goal, emotion, pace), puis les champs techniques
     du mouvement, puis "prompt" en dernier (fusion intelligente de tous
     les champs précédents).
  2. Cohérence : chaque appel reçoit le SCRIPT COMPLET (toutes les scènes,
     dans l'ordre) + une « bible de personnages » interne
     (self._characters_bible, alimentée par les métadonnées de l'ImagePrompt
     de chaque scène) pour que l'animation ne casse jamais l'apparence d'un
     personnage déjà établi.
  3. Si le LLM échoue (JSON invalide, erreur API, timeout) : retry
     automatique (max_retries tentatives) avec une correction JSON
     intelligente intercalée, puis fallback déterministe (aucune exception
     remontée, JSON toujours valide).

Contrat :
  - Moteur totalement indépendant des autres (LLMScriptGenerator,
    RewriteEngine, LLMImageGenerator, LLMScriptEvaluator) — ne les appelle
    jamais, ne dépend que de leurs CONTRATS DE DONNÉES en entrée
    (ScriptScene, VisualScene, ImagePrompt).
  - Entrée : generate_from_scenes(script_scene, visual_scene, image_prompt, script=None)
    → AnimationPrompt (contrat universel Sprint 25).
  - Prompt système externalisé dans prompts/animation_system_prompt.txt.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm import LLMMessage, build_llm, supports_reasoning
from src.llm_image_generator import ImagePrompt
from src.script_engine import Dialogue, Script, ScriptScene
from src.visual_engine import VisualScene

logger = logging.getLogger(__name__)


# ── Contrat AnimationPrompt (Sprint 25) ──────────────────────────────────────

@dataclass(frozen=True)
class AnimationPrompt:
    """
    Contrat universel de prompt d'animation — directement exploitable par des
    générateurs vidéo texte-vers-vidéo (Google Veo, Kling, Hailuo, Runway,
    Luma, Pika).

    Champs :
      camera_motion     : mouvement de caméra UNIQUEMENT (dolly, truck, crane,
                          pan, tilt, slow push-in, orbit, handheld...).
      subject_motion    : mouvement du sujet principal UNIQUEMENT.
      environment_motion: réaction de l'environnement UNIQUEMENT (particules,
                          vent, fumée, reflets, foule...).
      lighting_changes  : évolution de la lumière UNIQUEMENT.
      effects           : effets visuels additionnels UNIQUEMENT.
      sound_design      : ambiance sonore UNIQUEMENT.
      dialogues         : répliques de la scène (Sprint 31.1) — copiées
                          VERBATIM depuis ScriptScene.dialogues, jamais
                          reformulées, pour que l'AnimationPrompt soit
                          entièrement autonome (aucune dépendance à
                          final_script.json pour connaître ce qui se dit).
      transition        : fin du plan et enchaînement vers la scène suivante.
      duration          : durée du plan en secondes.
      prompt            : fusion intelligente de tous les champs précédents —
                          le brief final, directement exploitable.
      metadata          : goal, emotion (technique : provider/model/time_ms/
                          cost_usd ne sont jamais écrits dans le fichier de
                          production — voir ProductionPackageBuilder).
    """
    camera_motion: str
    subject_motion: str
    environment_motion: str
    lighting_changes: str
    effects: str
    sound_design: str
    dialogues: List[Dialogue]
    transition: str
    duration: int
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Modèle DeepSeek par défaut (Sprint 25) ──────────────────────────────────
# deepseek-chat est plus fiable en json_mode strict que deepseek-reasoner
# (réservé aux tâches de raisonnement long : scripts, évaluation, réécriture).
# Configurable via DEEPSEEK_ANIMATION_MODEL.
_DEEPSEEK_ANIMATION_MODEL = os.environ.get("DEEPSEEK_ANIMATION_MODEL", "deepseek-chat")


# ── Champs requis dans la réponse LLM ────────────────────────────────────────
# L'ORDRE de ces champs EST le raisonnement demandé au LLM : objectif,
# émotion et rythme d'abord, puis les champs techniques de mouvement, puis
# "prompt" EN DERNIER (fusion de tout ce qui précède).

_REQUIRED_STRING_FIELDS = (
    "goal", "emotion", "pace",
    "camera_motion", "subject_motion", "environment_motion",
    "lighting_changes", "effects", "sound_design", "transition",
    "prompt",
)
_REQUIRED_INT_FIELDS = ("duration",)
_REQUIRED_FIELDS = _REQUIRED_STRING_FIELDS + _REQUIRED_INT_FIELDS

_GENERIC_CHARACTER_LABELS = {"name", "nom", "personnage", "character", "characters"}

_DEFAULT_DURATION_SECONDS = 5
_MIN_DURATION_SECONDS = 2
_MAX_DURATION_SECONDS = 10


# ── Prompt système (Sprint 25) ───────────────────────────────────────────────
# Aucune longue chaîne de caractères ne reste dans le code Python — tout le
# raisonnement demandé au LLM est externalisé dans prompts/animation_system_prompt.txt.

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT_BASE = (_PROMPTS_DIR / "animation_system_prompt.txt").read_text(encoding="utf-8")


def _build_system_prompt(reasoning_enabled: bool) -> str:
    """
    Point d'extension pour un futur raisonnement explicite additionnel quand
    le modèle actif n'a pas de mode Reasoning natif (voir
    src.llm.supports_reasoning()) — le prompt de base couvre déjà le
    raisonnement étape par étape pour tous les modèles.
    """
    return _SYSTEM_PROMPT_BASE


# ── Extraction/nettoyage JSON robustes (mêmes garanties que LLMImageGenerator) ──
# Le LLM peut entourer le JSON de texte parasite, de balises <think>...</think>,
# de blocs Markdown, ou de petites erreurs de format. Ce module reste
# volontairement indépendant de llm_image_generator.py (aucun import croisé)
# et duplique ces quelques helpers de nettoyage, minimes et sans état.

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.IGNORECASE | re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"^[ \t]*//.*$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_think_tags(text: str) -> str:
    """Retire les blocs <think>...</think> (raisonnement qui fuite dans le contenu)."""
    return _THINK_TAG_RE.sub("", text).strip()


def _strip_code_fence(text: str) -> str:
    """Extrait le contenu d'un premier bloc ```json ... ``` s'il y en a un."""
    match = _CODE_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.replace("```json", "").replace("```", "").strip()


def _isolate_json_object(text: str) -> str:
    """Isole le premier objet JSON complet en comptant les accolades (en
    ignorant celles qui apparaissent à l'intérieur de chaînes)."""
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
    """Nettoyage final avant json.loads() : caractères de contrôle, faux
    commentaires, virgules traînantes."""
    text = _CONTROL_CHAR_RE.sub("", text)
    text = _LINE_COMMENT_RE.sub("", text)
    text = _BLOCK_COMMENT_RE.sub("", text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)
    return text.strip()


_JSON_REPAIR_INSTRUCTION = (
    "Le JSON precedent est invalide.\n"
    "Corrige UNIQUEMENT le JSON.\n"
    "Ne produis aucun texte supplementaire.\n"
    "Respecte exactement le schema demande (les 11 champs, dans le meme ordre, JSON valide et complet)."
)


class _AnimationJsonError(RuntimeError):
    """Erreur typée pour classifier précisément la cause d'un échec de génération."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"[{reason}] {detail}" if detail else reason)
        self.reason = reason


class LLMAnimationGenerator:
    """
    Générateur d'animation piloté par LLM — transforme une image statique
    (ImagePrompt) en un AnimationPrompt (contrat universel Sprint 25),
    cohérent sur l'ensemble d'un script, optimisé pour Google Veo / Kling /
    Hailuo / Runway / Luma / Pika.

    Moteur totalement indépendant : aucun appel à LLMScriptGenerator,
    RewriteEngine, LLMImageGenerator ou LLMScriptEvaluator — ne dépend que
    des contrats de données ScriptScene / VisualScene / ImagePrompt en entrée.

    Cohérence visuelle : une même instance accumule une « bible de
    personnages » (self._characters_bible) au fil des appels successifs sur
    les scènes d'UN MÊME script, dans l'ordre — instancier un nouveau
    générateur par script (ou appeler reset_continuity()) pour repartir
    d'une bible vide.
    """

    def __init__(
        self,
        script: Optional[Script] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        max_retries: int = 2,
    ) -> None:
        self._script = script
        self._provider_name = provider_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        self._provider = None  # lazy init
        self._characters_bible: Dict[str, str] = {}
        self._stats: Dict[str, Any] = {
            "llm_calls": 0,
            "llm_success": 0,
            "llm_failures": 0,
            "fallbacks": 0,
            "json_repair_attempts": 0,
            "json_repairs_success": 0,
            "fallback_reasons": {},
            "total_time_ms": 0,
            "total_cost_usd": 0.0,
        }

    @property
    def name(self) -> str:
        model_part = f"/{self._model}" if self._model else ""
        return f"llm_animation_{self._provider_name or 'auto'}{model_part}"

    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques cumulées du générateur (lecture seule)."""
        stats = dict(self._stats)
        stats["fallback_reasons"] = dict(self._stats["fallback_reasons"])
        return stats

    @property
    def characters_bible(self) -> Dict[str, str]:
        """Bible de personnages verrouillée (lecture seule)."""
        return dict(self._characters_bible)

    def reset_continuity(self) -> None:
        """Vide la bible de personnages — à appeler entre deux scripts distincts."""
        self._characters_bible = {}

    def _resolve_model(self) -> Optional[str]:
        """
        Résout le modèle à utiliser pour build_llm().

        Un `model=` explicite au constructeur est toujours prioritaire.
        Sinon, si le provider résolu (explicite ou auto-détecté) est
        DeepSeek, utilise _DEEPSEEK_ANIMATION_MODEL.
        """
        if self._model is not None:
            return self._model

        provider = self._provider_name or (
            "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else None
        )
        if provider == "deepseek":
            return _DEEPSEEK_ANIMATION_MODEL
        return None

    # ── Entrée riche — contrat AnimationPrompt (Sprint 25) ───────────────────

    def generate_from_scenes(
        self,
        script_scene: ScriptScene,
        visual_scene: VisualScene,
        image_prompt: ImagePrompt,
        script: Optional[Script] = None,
    ) -> AnimationPrompt:
        """
        Génère un AnimationPrompt (contrat universel) pour une scène, à
        partir de l'image déjà produite pour cette scène (`image_prompt`),
        avec cohérence sur l'ensemble du `script` fourni.

        Args:
            script_scene: Scène du script (narration, contexte).
            visual_scene: Scène visuelle (composition, éclairage, caméra de base).
            image_prompt: ImagePrompt déjà généré pour cette scène — l'animation
                doit rester fidèle au sujet/décor/style qu'il décrit.
            script: Script complet, pour la continuité narrative (personnages,
                ordre des scènes). Si omis, utilise self._script si disponible ;
                sinon la scène est traitée isolément (pas de continuité).

        Returns:
            AnimationPrompt — retombe sur une version déterministe dégradée
            si toutes les tentatives LLM échouent (aucune exception remontée).
        """
        effective_script = script if script is not None else self._script

        last_reason = "unknown"
        for attempt in range(1, self._max_retries + 1):
            try:
                animation_prompt = self._try_generate_llm(
                    effective_script, script_scene, visual_scene, image_prompt,
                )
                self._update_characters_bible(image_prompt)
                return animation_prompt
            except Exception as exc:
                last_reason = getattr(exc, "reason", "unknown") or "unknown"
                logger.warning(
                    "LLM Animation Generator — tentative %d/%d échouée (scène #%d, raison=%s) : %s",
                    attempt, self._max_retries, visual_scene.scene_order, last_reason, exc,
                )
                self._stats["llm_failures"] += 1

        logger.info(
            "LLM Animation Generator — fallback pour la scène #%d (raison=%s)",
            visual_scene.scene_order, last_reason,
        )
        self._stats["fallbacks"] += 1
        self._stats["fallback_reasons"][last_reason] = self._stats["fallback_reasons"].get(last_reason, 0) + 1
        self._update_characters_bible(image_prompt)
        return self._build_fallback_animation_prompt(script_scene, visual_scene, image_prompt, reason=last_reason)

    # ── Logique LLM ──────────────────────────────────────────────────────────

    def _try_generate_llm(
        self,
        script: Optional[Script],
        script_scene: ScriptScene,
        visual_scene: VisualScene,
        image_prompt: ImagePrompt,
    ) -> AnimationPrompt:
        if self._provider is None:
            self._provider = build_llm(provider=self._provider_name, model=self._resolve_model())
            logger.info("LLMAnimationGenerator utilise %s / %s", self._provider.name, self._provider.model)

        reasoning_enabled = supports_reasoning(self._provider.model)
        system_prompt = _build_system_prompt(reasoning_enabled)

        user_prompt = self._build_user_prompt(script, script_scene, visual_scene, image_prompt)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        response, elapsed_ms = self._call_llm(messages)
        self._raise_if_api_error(response)

        try:
            data = self._parse_and_validate(response)
        except _AnimationJsonError as first_err:
            logger.warning(
                "LLM Animation Generator — JSON invalide (raison=%s) pour la scène #%d — "
                "tentative de correction intelligente.",
                first_err.reason, visual_scene.scene_order,
            )
            self._stats["json_repair_attempts"] += 1
            repair_messages = messages + [
                LLMMessage(role="assistant", content=response.content[:4000]),
                LLMMessage(role="user", content=_JSON_REPAIR_INSTRUCTION),
            ]
            repair_response, repair_elapsed_ms = self._call_llm(repair_messages)
            self._raise_if_api_error(repair_response)
            try:
                data = self._parse_and_validate(repair_response)
            except _AnimationJsonError as second_err:
                logger.warning(
                    "LLM Animation Generator — correction JSON échouée (raison=%s) pour la scène #%d.",
                    second_err.reason, visual_scene.scene_order,
                )
                raise
            response, elapsed_ms = repair_response, repair_elapsed_ms
            self._stats["json_repairs_success"] += 1
            logger.info(
                "LLM Animation Generator — JSON corrigé avec succès pour la scène #%d.",
                visual_scene.scene_order,
            )

        animation_prompt = self._build_animation_prompt(data, response, elapsed_ms, script_scene.dialogues)

        self._stats["llm_success"] += 1
        logger.info(
            "LLM Animation OK — scène #%d, %d tokens, $%.6f, %d ms, provider=%s, model=%s, reasoning=%s",
            visual_scene.scene_order, response.total_tokens, response.cost_usd,
            elapsed_ms, response.provider_name, response.model, reasoning_enabled,
        )
        return animation_prompt

    def _call_llm(self, messages: List[LLMMessage]):
        """Appelle le LLM et accumule les statistiques (temps, coût, nombre d'appels)."""
        start = time.time()
        self._stats["llm_calls"] += 1
        response = self._provider.generate(
            messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_mode=True,
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
        raise _AnimationJsonError(reason, response.content[:200])

    @classmethod
    def _parse_and_validate(cls, response: Any) -> Dict[str, Any]:
        """Extrait, nettoie, parse et valide le JSON d'une réponse LLM —
        classifie précisément la cause d'un éventuel échec."""
        content = (response.content or "").strip()
        if not content:
            raise _AnimationJsonError("empty_response", "réponse vide")

        json_str = cls._extract_json(content)
        incomplete = getattr(response, "finish_reason", None) == "length"

        if not json_str:
            raise _AnimationJsonError(
                "json_incomplete" if incomplete else "json_invalid",
                "aucun objet JSON isolable dans la réponse",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise _AnimationJsonError(
                "json_incomplete" if incomplete else "json_invalid", str(exc)
            ) from exc

        try:
            cls._validate_json_structure(data)
        except ValueError as exc:
            raise _AnimationJsonError("validation_failed", str(exc)) from exc

        return data

    # ── Continuité narrative ──────────────────────────────────────────────────

    def _build_continuity_block(self, script: Optional[Script], current_order: int) -> str:
        """
        Construit le bloc « CONTINUITE NARRATIVE » injecté dans le prompt
        utilisateur : le script complet (dans l'ordre) pour situer la scène
        courante, et la bible de personnages déjà verrouillés (alimentée par
        les ImagePrompt successifs).
        """
        if script is None:
            return ""

        lines: List[str] = ["", "=== NARRATIVE CONTINUITY (full script, in order) ==="]
        for scene in sorted(script.scenes, key=lambda s: s.order):
            marker = " <<< SCENE ACTUELLE" if scene.order == current_order else ""
            lines.append(f"  [{scene.order}] {scene.narration_text}{marker}")

        if self._characters_bible:
            lines.append("")
            lines.append("=== CHARACTERS ALREADY ESTABLISHED (respect EXACTLY if they reappear) ===")
            for name, desc in self._characters_bible.items():
                lines.append(f"  - {name} : {desc}")
        else:
            lines.append("")
            lines.append("(No character established yet.)")

        return "\n".join(lines)

    def _update_characters_bible(self, image_prompt: ImagePrompt) -> None:
        """
        Verrouille la description de chaque NOUVEAU personnage rencontré,
        à partir des personnages listés dans image_prompt.metadata["characters"]
        (déjà établis par LLMImageGenerator pour cette même scène). Les
        personnages déjà connus conservent leur description d'origine (jamais
        écrasée), pour garantir une animation cohérente dans le temps.
        """
        characters = image_prompt.metadata.get("characters") or []
        for entry in characters:
            if not isinstance(entry, str) or not entry.strip():
                continue
            entry = entry.strip()
            name, sep, desc = entry.partition(":")
            name = name.strip()
            if sep and name.lower() not in _GENERIC_CHARACTER_LABELS:
                desc = desc.strip() or entry
            else:
                name, desc = entry, entry
            if name and name not in self._characters_bible:
                self._characters_bible[name] = desc

    # ── Construction du prompt ────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        script: Optional[Script],
        script_scene: ScriptScene,
        visual_scene: VisualScene,
        image_prompt: ImagePrompt,
    ) -> str:
        desc = script_scene.scene.description
        lines: List[str] = [
            "Write the AnimationPrompt contract for the following scene — animate the image already generated, do not re-describe it.",
            "First reason about the goal, the emotion, and the pace before deciding camera/subject/environment/lighting motion, then synthesize 'prompt' last.",
            "",
            "=== SCRIPT SCENE (current) — structured storyboard fields used to build camera motion ===",
            f"  Camera           : {desc.camera}",
            f"  Composition      : {desc.composition}",
            f"  Lighting         : {desc.lighting}",
            f"  Mood             : {desc.mood}",
            f"  Viewer emotion   : {desc.viewer_emotion}",
            f"  Director's notes : {desc.director_notes}",
            f"  Duration         : {script_scene.duration_seconds}s",
            "",
            "=== VISUAL PLAN (base) ===",
            f"  Shot type      : {visual_scene.shot_type}",
            f"  Camera motion  : {visual_scene.camera_motion}",
            f"  Transition     : {visual_scene.transition}",
            f"  Animation notes: {visual_scene.animation_notes}",
            "",
            "=== IMAGE ALREADY GENERATED FOR THIS SCENE (animate it, do not re-describe it) ===",
            f"  Subject         : {image_prompt.subject}",
            f"  Setting/ambiance: {image_prompt.scene_description}",
            f"  Style           : {image_prompt.style}",
            f"  Characters      : {', '.join(str(c) for c in image_prompt.metadata.get('characters', []) or []) or 'none'}",
        ]
        lines.append(self._build_continuity_block(script, script_scene.order))
        lines += ["", "Generate the AnimationPrompt contract JSON now."]
        return "\n".join(lines)

    # ── Extraction et validation JSON ────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extrait un objet JSON exploitable d'une réponse LLM potentiellement
        « sale » : retire les balises <think>...</think>, les blocs Markdown
        ```json ... ```, isole le premier objet JSON complet par comptage
        d'accolades, puis nettoie caractères de contrôle / faux commentaires /
        virgules traînantes."""
        text = text.strip()
        text = _strip_think_tags(text)
        text = _strip_code_fence(text)
        text = _isolate_json_object(text)
        text = _clean_json_text(text)
        return text.strip()

    @staticmethod
    def _validate_json_structure(data: Dict[str, Any]) -> None:
        """Valide la structure du JSON de réponse."""
        for field_name in _REQUIRED_STRING_FIELDS:
            if field_name not in data:
                raise ValueError(f"Champ obligatoire manquant dans la réponse : '{field_name}'")
            if not isinstance(data[field_name], str) or not data[field_name].strip():
                raise ValueError(f"Le champ '{field_name}' doit être une chaîne non vide")
        for field_name in _REQUIRED_INT_FIELDS:
            if field_name not in data:
                raise ValueError(f"Champ obligatoire manquant dans la réponse : '{field_name}'")
            if not isinstance(data[field_name], (int, float)) or isinstance(data[field_name], bool):
                raise ValueError(f"Le champ '{field_name}' doit être un nombre")

    # ── Construction AnimationPrompt ──────────────────────────────────────────

    @staticmethod
    def _clamp_duration(duration: Any) -> int:
        try:
            value = int(round(float(duration)))
        except (TypeError, ValueError):
            value = _DEFAULT_DURATION_SECONDS
        return max(_MIN_DURATION_SECONDS, min(_MAX_DURATION_SECONDS, value))

    @classmethod
    def _build_animation_prompt(
        cls, data: Dict[str, Any], response: Any, elapsed_ms: int, dialogues: List[Dialogue],
    ) -> AnimationPrompt:
        """
        Construit l'AnimationPrompt (contrat Sprint 25/31.1) à partir du JSON
        validé. `dialogues` est copié VERBATIM depuis ScriptScene.dialogues —
        jamais généré ni reformulé par le LLM (Sprint 31.1 : l'AnimationPrompt
        doit être autonome, sans réécriture des répliques).
        """
        return AnimationPrompt(
            camera_motion=data["camera_motion"].strip(),
            subject_motion=data["subject_motion"].strip(),
            environment_motion=data["environment_motion"].strip(),
            lighting_changes=data["lighting_changes"].strip(),
            effects=data["effects"].strip(),
            sound_design=data["sound_design"].strip(),
            dialogues=list(dialogues),
            transition=data["transition"].strip(),
            duration=cls._clamp_duration(data["duration"]),
            prompt=data["prompt"].strip(),
            metadata={
                "goal": data["goal"].strip(),
                "emotion": data["emotion"].strip(),
                "pace": data["pace"].strip(),
                "provider": response.provider_name,
                "model": response.model,
                "time_ms": elapsed_ms,
                "cost_usd": round(response.cost_usd, 6),
            },
        )

    @classmethod
    def _build_fallback_animation_prompt(
        cls,
        script_scene: ScriptScene,
        visual_scene: VisualScene,
        image_prompt: ImagePrompt,
        reason: str = "unknown",
    ) -> AnimationPrompt:
        """
        Construit un AnimationPrompt déterministe à partir des données déjà
        disponibles (VisualScene + ImagePrompt), sans appel LLM — utilisé
        quand toutes les tentatives LLM échouent. Garantit un contrat de
        sortie toujours valide et un comportement déterministe, sans jamais
        remonter d'exception à l'appelant.

        `reason` documente pourquoi le fallback a été déclenché (ex:
        "json_invalid", "json_incomplete", "timeout", "api_error",
        "validation_failed", "empty_response") — utile pour le diagnostic.
        """
        camera_motion = visual_scene.camera_motion.strip() or "slow push-in"
        subject = image_prompt.subject.strip() or "the main subject"
        duration = cls._clamp_duration(visual_scene.duration_seconds or script_scene.duration_seconds)

        prompt = (
            f"Slow, cinematic {camera_motion} on {subject}, subtle ambient motion in the "
            f"environment, stable lighting, natural continuation of the scene, {visual_scene.transition or 'smooth cut'}."
        ).strip()

        return AnimationPrompt(
            camera_motion=camera_motion,
            subject_motion="subtle, natural, near-static motion",
            environment_motion="minimal ambient motion (subtle air movement, soft light shifts)",
            lighting_changes="stable lighting, no abrupt changes",
            effects="none",
            sound_design="ambient sound consistent with the scene mood",
            dialogues=list(script_scene.dialogues),
            transition=visual_scene.transition.strip() or "smooth cut",
            duration=duration,
            prompt=prompt,
            metadata={
                "goal": "",
                "emotion": "",
                "pace": "steady",
                "provider": "fallback_heuristic",
                "model": "",
                "time_ms": 0,
                "cost_usd": 0.0,
                "fallback_reason": reason,
            },
        )
