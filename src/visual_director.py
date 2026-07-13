"""
Visual Director — Shot Planning Engine (Sprint 26).

Objectif :
  Centraliser TOUTES les décisions de réalisation (cadrage, focale, angle,
  composition, profondeur de champ, éclairage, palette, focal point, moment
  miniature) dans un moteur unique, AVANT toute génération d'image ou
  d'animation. Ce moteur devient la « bible visuelle » de chaque scène : il
  ne décrit jamais l'image, il ne prend que des décisions artistiques.

Pipeline cible :
    ScriptScene
        │
        ▼
    VisualDirector
        │
        ▼
    ShotPlan
        │
        ├────────► LLMImageGenerator (reçoit le cadrage/la focale/la composition)
        │
        └────────► LLMAnimationGenerator (utilise les mêmes informations)

Contrat de sortie — ShotPlan (Sprint 26) :
    {
      "shot_type": "...", "camera_angle": "...", "lens": "...",
      "composition": "...", "depth_of_field": "...", "lighting_style": "...",
      "color_palette": "...", "focal_point": "...", "visual_priority": [...],
      "thumbnail_moment": "...", "cinematic_goal": "...",
      "metadata": {
        "provider": "...", "model": "...", "time_ms": 0, "cost_usd": 0.0,
      },
    }

Architecture (identique aux autres moteurs LLM — retry, réparation JSON
intelligente, fallback déterministe sans exception, statistiques internes) :
  1. Le LLM (via build_llm()) agit comme un Director of Photography — il
     raisonne en silence (émotion, moment à figer, hiérarchie visuelle) avant
     d'écrire le JSON, dans l'ordre exact du contrat ShotPlan.
  2. Cohérence : chaque appel reçoit le SCRIPT COMPLET (toutes les scènes,
     dans l'ordre) + un historique des ShotPlan déjà établis pour les scènes
     précédentes (self._shot_plans_history), pour que les choix de cadrage
     évoluent naturellement au sein du même film.
  3. Si le LLM échoue (JSON invalide, erreur API, timeout) : retry
     automatique (max_retries tentatives) avec une correction JSON
     intelligente intercalée, puis fallback déterministe (aucune exception
     remontée, JSON toujours valide).

Contrat :
  - Moteur totalement indépendant des autres moteurs (LLMScriptGenerator,
    RewriteEngine, LLMImageGenerator, LLMAnimationGenerator, LLMScriptEvaluator)
    — ne les appelle jamais, ne dépend que des contrats de données publics
    ScriptScene / BrandProfile en entrée et de ShotPlan en sortie.
  - Entrée : generate_shot_plan(script_scene, brand_profile, script=None)
    → ShotPlan (contrat universel Sprint 26).
  - Prompt système externalisé dans prompts/visual_director_system_prompt.txt.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.brand_engine import BrandProfile
from src.llm import LLMMessage, build_llm, supports_reasoning
from src.script_engine import Script, ScriptScene

logger = logging.getLogger(__name__)


# ── Contrat ShotPlan (Sprint 26) ─────────────────────────────────────────────

@dataclass(frozen=True)
class ShotPlan:
    """
    Contrat universel de plan de tournage — la « bible visuelle » d'une
    scène. Décisions artistiques UNIQUEMENT, jamais de description d'image.

    Champs :
      shot_type        : type de plan (Close-Up, Medium Shot, Wide Shot...).
      camera_angle      : angle de caméra (Eye Level, Low Angle...).
      lens              : focale réaliste (24mm, 35mm, 50mm, 85mm...).
      composition       : règle de composition (Rule of Thirds, Symmetry...).
      depth_of_field    : profondeur de champ (shallow/deep, description).
      lighting_style    : direction photo (Rembrandt, Golden Hour, Neon...).
      color_palette     : palette dominante ("Warm Orange + Cyan"...).
      focal_point       : UNE seule zone/sujet qui doit attirer le regard.
      visual_priority   : hiérarchie des éléments, du plus au moins important.
      thumbnail_moment  : LE moment de la scène digne d'une miniature YouTube.
      cinematic_goal    : objectif cinématographique de ce plan.
      metadata          : provider, model, time_ms, cost_usd.
    """
    shot_type: str
    camera_angle: str
    lens: str
    composition: str
    depth_of_field: str
    lighting_style: str
    color_palette: str
    focal_point: str
    visual_priority: List[str]
    thumbnail_moment: str
    cinematic_goal: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Modèle DeepSeek par défaut (Sprint 26) ──────────────────────────────────
# deepseek-chat est plus fiable en json_mode strict que deepseek-reasoner.
# Configurable via DEEPSEEK_VISUAL_DIRECTOR_MODEL.
_DEEPSEEK_VISUAL_DIRECTOR_MODEL = os.environ.get("DEEPSEEK_VISUAL_DIRECTOR_MODEL", "deepseek-chat")


# ── Champs requis dans la réponse LLM ────────────────────────────────────────
# L'ORDRE de ces champs EST l'ordre exact du contrat ShotPlan.

_REQUIRED_STRING_FIELDS = (
    "shot_type", "camera_angle", "lens", "composition", "depth_of_field",
    "lighting_style", "color_palette", "focal_point",
    "thumbnail_moment", "cinematic_goal",
)
_REQUIRED_LIST_FIELDS = ("visual_priority",)
_REQUIRED_FIELDS = _REQUIRED_STRING_FIELDS + _REQUIRED_LIST_FIELDS

_MAX_VISUAL_PRIORITY_ITEMS = 5

_DEFAULT_SHOT_TYPE = "Medium Shot"
_DEFAULT_CAMERA_ANGLE = "Eye Level"
_DEFAULT_LENS = "35mm"
_DEFAULT_COMPOSITION = "Rule of Thirds"
_DEFAULT_DEPTH_OF_FIELD = "Moderate depth of field, subject in sharp focus, softly blurred background"
_DEFAULT_LIGHTING_STYLE = "Soft Diffused Lighting"
_DEFAULT_COLOR_PALETTE = "Neutral Tones + Cool Blue"


# ── Prompt système (Sprint 26) ───────────────────────────────────────────────
# Aucune longue chaîne de caractères ne reste dans le code Python — tout le
# raisonnement demandé au LLM est externalisé dans
# prompts/visual_director_system_prompt.txt.

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT_BASE = (_PROMPTS_DIR / "visual_director_system_prompt.txt").read_text(encoding="utf-8")


def _build_system_prompt(reasoning_enabled: bool) -> str:
    """
    Point d'extension pour un futur raisonnement explicite additionnel quand
    le modèle actif n'a pas de mode Reasoning natif (voir
    src.llm.supports_reasoning()) — le prompt de base couvre déjà le
    raisonnement étape par étape pour tous les modèles.
    """
    return _SYSTEM_PROMPT_BASE


# ── Extraction/nettoyage JSON robustes (mêmes garanties que les autres moteurs LLM) ──
# Ce module reste volontairement indépendant des autres moteurs (aucun import
# croisé) et duplique ces quelques helpers de nettoyage, minimes et sans état.

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


class _ShotPlanJsonError(RuntimeError):
    """Erreur typée pour classifier précisément la cause d'un échec de génération."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"[{reason}] {detail}" if detail else reason)
        self.reason = reason


class VisualDirector:
    """
    Director of Photography piloté par LLM — produit un ShotPlan (contrat
    universel Sprint 26) cohérent sur l'ensemble d'un script. Devient la
    source de vérité pour le cadrage : LLMImageGenerator et
    LLMAnimationGenerator reçoivent ces décisions au lieu de les inventer
    chacun de leur côté.

    Moteur totalement indépendant : aucun appel à LLMScriptGenerator,
    RewriteEngine, LLMImageGenerator, LLMAnimationGenerator ou
    LLMScriptEvaluator — ne dépend que des contrats de données publics
    ScriptScene / BrandProfile en entrée.

    Cohérence visuelle : une même instance accumule un historique des
    ShotPlan déjà établis (self._shot_plans_history) au fil des appels
    successifs sur les scènes d'UN MÊME script, dans l'ordre — instancier un
    nouveau VisualDirector par script (ou appeler reset_continuity()) pour
    repartir d'un historique vide.
    """

    def __init__(
        self,
        script: Optional[Script] = None,
        brand_profile: Optional[BrandProfile] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 768,
        max_retries: int = 2,
    ) -> None:
        self._script = script
        self._brand_profile = brand_profile
        self._provider_name = provider_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        self._provider = None  # lazy init
        self._shot_plans_history: Dict[int, ShotPlan] = {}
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
        return f"visual_director_{self._provider_name or 'auto'}{model_part}"

    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques cumulées du générateur (lecture seule)."""
        stats = dict(self._stats)
        stats["fallback_reasons"] = dict(self._stats["fallback_reasons"])
        return stats

    @property
    def shot_plans(self) -> Dict[int, ShotPlan]:
        """Historique des ShotPlan déjà établis, par ordre de scène (lecture seule)."""
        return dict(self._shot_plans_history)

    def reset_continuity(self) -> None:
        """Vide l'historique des ShotPlan — à appeler entre deux scripts distincts."""
        self._shot_plans_history = {}

    def _resolve_model(self) -> Optional[str]:
        """
        Résout le modèle à utiliser pour build_llm().

        Un `model=` explicite au constructeur est toujours prioritaire.
        Sinon, si le provider résolu (explicite ou auto-détecté) est
        DeepSeek, utilise _DEEPSEEK_VISUAL_DIRECTOR_MODEL.
        """
        if self._model is not None:
            return self._model

        provider = self._provider_name or (
            "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else None
        )
        if provider == "deepseek":
            return _DEEPSEEK_VISUAL_DIRECTOR_MODEL
        return None

    # ── Entrée riche — contrat ShotPlan (Sprint 26) ──────────────────────────

    def generate_shot_plan(
        self,
        script_scene: ScriptScene,
        brand_profile: BrandProfile,
        script: Optional[Script] = None,
    ) -> ShotPlan:
        """
        Génère un ShotPlan (contrat universel) pour une scène, avec
        cohérence de réalisation sur l'ensemble du `script` fourni.

        Args:
            script_scene: Scène du script (narration, contexte).
            brand_profile: Identité de marque (style, ton).
            script: Script complet, pour la continuité de réalisation (ordre
                des scènes, décisions déjà prises). Si omis, utilise
                self._script si disponible ; sinon la scène est traitée
                isolément (pas de continuité).

        Returns:
            ShotPlan — retombe sur une version déterministe dégradée si
            toutes les tentatives LLM échouent (aucune exception remontée).
        """
        effective_script = script if script is not None else self._script
        effective_brand = brand_profile if brand_profile is not None else self._brand_profile

        last_reason = "unknown"
        for attempt in range(1, self._max_retries + 1):
            try:
                shot_plan = self._try_generate_llm(effective_script, script_scene, effective_brand)
                self._shot_plans_history[script_scene.order] = shot_plan
                return shot_plan
            except Exception as exc:
                last_reason = getattr(exc, "reason", "unknown") or "unknown"
                logger.warning(
                    "Visual Director — tentative %d/%d échouée (scène #%d, raison=%s) : %s",
                    attempt, self._max_retries, script_scene.order, last_reason, exc,
                )
                self._stats["llm_failures"] += 1

        logger.info(
            "Visual Director — fallback pour la scène #%d (raison=%s)",
            script_scene.order, last_reason,
        )
        self._stats["fallbacks"] += 1
        self._stats["fallback_reasons"][last_reason] = self._stats["fallback_reasons"].get(last_reason, 0) + 1
        shot_plan = self._build_fallback_shot_plan(script_scene, effective_brand, reason=last_reason)
        self._shot_plans_history[script_scene.order] = shot_plan
        return shot_plan

    # ── Logique LLM ──────────────────────────────────────────────────────────

    def _try_generate_llm(
        self,
        script: Optional[Script],
        script_scene: ScriptScene,
        brand_profile: Optional[BrandProfile],
    ) -> ShotPlan:
        if self._provider is None:
            self._provider = build_llm(provider=self._provider_name, model=self._resolve_model())
            logger.info("VisualDirector utilise %s / %s", self._provider.name, self._provider.model)

        reasoning_enabled = supports_reasoning(self._provider.model)
        system_prompt = _build_system_prompt(reasoning_enabled)

        user_prompt = self._build_user_prompt(script, script_scene, brand_profile)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        response, elapsed_ms = self._call_llm(messages)
        self._raise_if_api_error(response)

        try:
            data = self._parse_and_validate(response)
        except _ShotPlanJsonError as first_err:
            logger.warning(
                "Visual Director — JSON invalide (raison=%s) pour la scène #%d — "
                "tentative de correction intelligente.",
                first_err.reason, script_scene.order,
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
            except _ShotPlanJsonError as second_err:
                logger.warning(
                    "Visual Director — correction JSON échouée (raison=%s) pour la scène #%d.",
                    second_err.reason, script_scene.order,
                )
                raise
            response, elapsed_ms = repair_response, repair_elapsed_ms
            self._stats["json_repairs_success"] += 1
            logger.info(
                "Visual Director — JSON corrigé avec succès pour la scène #%d.",
                script_scene.order,
            )

        shot_plan = self._build_shot_plan(data, response, elapsed_ms)

        self._stats["llm_success"] += 1
        logger.info(
            "Visual Director OK — scène #%d, %d tokens, $%.6f, %d ms, provider=%s, model=%s, reasoning=%s",
            script_scene.order, response.total_tokens, response.cost_usd,
            elapsed_ms, response.provider_name, response.model, reasoning_enabled,
        )
        return shot_plan

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
        raise _ShotPlanJsonError(reason, response.content[:200])

    @classmethod
    def _parse_and_validate(cls, response: Any) -> Dict[str, Any]:
        """Extrait, nettoie, parse et valide le JSON d'une réponse LLM —
        classifie précisément la cause d'un éventuel échec."""
        content = (response.content or "").strip()
        if not content:
            raise _ShotPlanJsonError("empty_response", "réponse vide")

        json_str = cls._extract_json(content)
        incomplete = getattr(response, "finish_reason", None) == "length"

        if not json_str:
            raise _ShotPlanJsonError(
                "json_incomplete" if incomplete else "json_invalid",
                "aucun objet JSON isolable dans la réponse",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise _ShotPlanJsonError(
                "json_incomplete" if incomplete else "json_invalid", str(exc)
            ) from exc

        try:
            cls._validate_json_structure(data)
        except ValueError as exc:
            raise _ShotPlanJsonError("validation_failed", str(exc)) from exc

        return data

    # ── Continuité de réalisation ─────────────────────────────────────────────

    def _build_continuity_block(self, script: Optional[Script], current_order: int) -> str:
        """
        Construit le bloc « CONTINUITE VISUELLE » injecté dans le prompt
        utilisateur : le script complet (dans l'ordre) pour situer la scène
        courante, et l'historique des ShotPlan déjà établis pour que le
        cadrage évolue naturellement au sein du même film.
        """
        if script is None:
            return ""

        lines: List[str] = ["", "=== VISUAL CONTINUITY (full script, in order) ==="]
        for scene in sorted(script.scenes, key=lambda s: s.order):
            marker = " <<< CURRENT SCENE" if scene.order == current_order else ""
            lines.append(f"  [{scene.order}] ({scene.scene.type}) {scene.narration_text}{marker}")

        if self._shot_plans_history:
            lines.append("")
            lines.append("=== SHOT PLANS ALREADY ESTABLISHED (evolve naturally, no abrupt break) ===")
            for order in sorted(self._shot_plans_history):
                plan = self._shot_plans_history[order]
                lines.append(
                    f"  - Scene {order} : {plan.shot_type}, {plan.camera_angle}, {plan.lens}, "
                    f"{plan.composition}, lighting={plan.lighting_style}, palette={plan.color_palette}, "
                    f"focal_point={plan.focal_point}"
                )
        else:
            lines.append("")
            lines.append("(No shot plan established yet — this scene may set the visual language.)")

        return "\n".join(lines)

    # ── Construction du prompt (en anglais, Sprint 32.1 — cf. dernière consigne) ─

    def _build_user_prompt(
        self,
        script: Optional[Script],
        script_scene: ScriptScene,
        brand_profile: Optional[BrandProfile],
    ) -> str:
        desc = script_scene.scene.description
        lines: List[str] = [
            "Make the cinematography decisions (ShotPlan) for the following scene.",
            "First reason silently about the emotion, the moment to freeze, and the visual hierarchy, "
            "then return ONLY the ShotPlan contract JSON, in the exact order of the schema.",
            "",
            "=== SCRIPT SCENE (current) — structured storyboard fields ===",
            f"  Type             : {script_scene.scene.type}",
            f"  Setting          : {desc.setting}",
            f"  Composition      : {desc.composition}",
            f"  Characters       : {desc.characters}",
            f"  Lighting         : {desc.lighting}",
            f"  Camera           : {desc.camera}",
            f"  Mood             : {desc.mood}",
            f"  Symbolism        : {desc.symbolism}",
            f"  Director's notes : {desc.director_notes}",
            f"  Viewer emotion   : {desc.viewer_emotion}",
            f"  Dialogues        : {script_scene.narration_text}",
            f"  Duration         : {script_scene.duration_seconds}s",
        ]
        if brand_profile is not None:
            lines += [
                "",
                "=== BRAND IDENTITY ===",
                f"  Brand       : {brand_profile.name}",
                f"  Tone        : {brand_profile.tone}",
                f"  Visual style: {brand_profile.visual_style}",
            ]
        lines.append(self._build_continuity_block(script, script_scene.order))
        lines += ["", "Generate the ShotPlan contract JSON now."]
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
        for field_name in _REQUIRED_LIST_FIELDS:
            if field_name not in data:
                raise ValueError(f"Champ obligatoire manquant dans la réponse : '{field_name}'")
            if not isinstance(data[field_name], list) or not data[field_name]:
                raise ValueError(f"Le champ '{field_name}' doit être une liste non vide")

    # ── Construction ShotPlan ─────────────────────────────────────────────────

    @staticmethod
    def _clamp_visual_priority(visual_priority: List[Any]) -> List[str]:
        cleaned = [str(item).strip() for item in visual_priority if str(item).strip()]
        return cleaned[:_MAX_VISUAL_PRIORITY_ITEMS] or ["Main Subject"]

    @classmethod
    def _build_shot_plan(cls, data: Dict[str, Any], response: Any, elapsed_ms: int) -> ShotPlan:
        """Construit le ShotPlan (contrat Sprint 26) à partir du JSON validé."""
        return ShotPlan(
            shot_type=data["shot_type"].strip(),
            camera_angle=data["camera_angle"].strip(),
            lens=data["lens"].strip(),
            composition=data["composition"].strip(),
            depth_of_field=data["depth_of_field"].strip(),
            lighting_style=data["lighting_style"].strip(),
            color_palette=data["color_palette"].strip(),
            focal_point=data["focal_point"].strip(),
            visual_priority=cls._clamp_visual_priority(data["visual_priority"]),
            thumbnail_moment=data["thumbnail_moment"].strip(),
            cinematic_goal=data["cinematic_goal"].strip(),
            metadata={
                "provider": response.provider_name,
                "model": response.model,
                "time_ms": elapsed_ms,
                "cost_usd": round(response.cost_usd, 6),
            },
        )

    @staticmethod
    def _build_fallback_shot_plan(
        script_scene: ScriptScene,
        brand_profile: Optional[BrandProfile],
        reason: str = "unknown",
    ) -> ShotPlan:
        """
        Construit un ShotPlan déterministe à partir des données déjà
        disponibles (ScriptScene + BrandProfile), sans appel LLM — utilisé
        quand toutes les tentatives LLM échouent. Garantit un contrat de
        sortie toujours valide et un comportement déterministe, sans jamais
        remonter d'exception à l'appelant.

        `reason` documente pourquoi le fallback a été déclenché (ex:
        "json_invalid", "json_incomplete", "timeout", "api_error",
        "validation_failed", "empty_response") — utile pour le diagnostic.
        """
        setting = script_scene.scene.description.setting.strip()
        focal_point = setting[:60] or "Main Subject"
        color_palette = _DEFAULT_COLOR_PALETTE
        if brand_profile is not None and brand_profile.visual_style:
            color_palette = f"{brand_profile.visual_style.strip()} tones"

        return ShotPlan(
            shot_type=_DEFAULT_SHOT_TYPE,
            camera_angle=_DEFAULT_CAMERA_ANGLE,
            lens=_DEFAULT_LENS,
            composition=_DEFAULT_COMPOSITION,
            depth_of_field=_DEFAULT_DEPTH_OF_FIELD,
            lighting_style=_DEFAULT_LIGHTING_STYLE,
            color_palette=color_palette,
            focal_point=focal_point,
            visual_priority=[focal_point],
            thumbnail_moment=setting or script_scene.narration_text.strip(),
            cinematic_goal="Communicate the core moment of the scene with instant clarity.",
            metadata={
                "provider": "fallback_heuristic",
                "model": "",
                "time_ms": 0,
                "cost_usd": 0.0,
                "fallback_reason": reason,
            },
        )
