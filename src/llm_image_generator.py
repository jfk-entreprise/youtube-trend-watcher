"""
LLM Image Generator — Contrat ImagePrompt universel via LLM, optimisé pour
Google Whisk / Nano Banana (Sprint 23 → 25, contrat ImagePrompt Sprint 24.1-24.3).

Objectif :
  Produire, pour chaque scène, un contrat d'image structuré et directement
  exploitable par un outil comme Google Whisk (qui sépare Subject / Scene /
  Style en entrées distinctes), tout en racontant une histoire visuelle
  CONTINUE sur l'ensemble du script (personnages, lieux, objets récurrents).

Contrat de sortie — ImagePrompt (Sprint 24.1) :
    {
      "subject": "...",            # sujet principal UNIQUEMENT
      "scene_description": "...",  # décor/ambiance/lumière/composition/détails
      "style": "...",              # rendu artistique UNIQUEMENT
      "prompt": "...",             # action / brief réalisateur UNIQUEMENT
      "negative_prompt": "...",
      "metadata": {
        "goal": "...", "emotion": "...", "characters": [...],
        "provider": "...", "model": "...", "time_ms": 0, "cost_usd": 0.0,
      },
    }

Architecture :
  1. Le LLM (via build_llm()) agit comme un Senior Hollywood Art Director +
     Prompt Engineer (Sprint 24.2) — raisonnement d'abord (goal, emotion,
     characters), puis les 5 champs du contrat, dans cet ordre.
  2. Cohérence visuelle (Sprint 24.3) : chaque appel reçoit le SCRIPT COMPLET
     (toutes les scènes, dans l'ordre) + une « bible de personnages » interne
     (self._characters_bible) qui verrouille, dès leur première apparition,
     la description physique (visage, vêtements, coiffure, accessoires, âge)
     de chaque personnage récurrent — réinjectée telle quelle dans les
     scènes suivantes pour que le LLM la réutilise à l'identique.
  3. Si le LLM échoue (JSON invalide, erreur API, timeout) : retry
     automatique (max_retries tentatives), puis fallback vers
     HeuristicImageGenerator — le résultat heuristique est alors enveloppé
     dans un ImagePrompt dégradé pour garder un contrat de sortie uniforme.
  4. Pour rester compatible avec ImageGenerator (generate(scene, plan) doit
     retourner un GeneratedImage — contrat d'un AUTRE moteur, non modifié),
     l'ImagePrompt est converti en GeneratedImage via _to_generated_image().
     Les dimensions/aspect ratio/seed restent calculés par les helpers
     déterministes de HeuristicImageGenerator (réutilisés, pas dupliqués).

Contrat :
  - Conserve `ImageGenerator` comme interface (generate(scene, plan) → GeneratedImage).
  - Entrée « riche » : generate_from_scenes(script_scene, visual_scene, brand_profile, script=None)
    → ImagePrompt (contrat universel Sprint 24.1).
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataclasses import dataclass, field

from src.brand_engine import BrandProfile
from src.image_engine import GeneratedImage, HeuristicImageGenerator, ImageGenerator
from src.llm import LLMMessage, build_llm, supports_reasoning
from src.script_engine import Script, ScriptScene
from src.visual_engine import VisualPlan, VisualScene

logger = logging.getLogger(__name__)


# ── Contrat ImagePrompt (Sprint 24.1) ────────────────────────────────────────

@dataclass(frozen=True)
class ImagePrompt:
    """
    Contrat universel de prompt d'image — directement exploitable par des
    outils qui séparent Subject / Scene / Style (ex: Google Whisk).

    Champs :
      subject          : sujet principal UNIQUEMENT (personnage, objet, animal...).
      scene_description: décor, environnement, ambiance, lumière, composition,
                          éléments visuels, détails réalistes, contexte narratif.
      style             : rendu artistique UNIQUEMENT (cinematic, photorealistic,
                          HDR, realistic textures, color grading, volumetric
                          lighting...).
      prompt            : action / résultat attendu UNIQUEMENT — le brief du
                          réalisateur ("que doit-il se passer dans l'image ?").
      negative_prompt   : éléments à éviter.
      metadata          : goal, emotion, characters (liste), provider, model,
                          time_ms, cost_usd.
    """
    subject: str
    scene_description: str
    style: str
    prompt: str
    negative_prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Modèle DeepSeek par défaut (Sprint 25, fiabilisé Sprint 24.5) ───────────
# deepseek-reasoner est réservé aux tâches de raisonnement (scripts,
# évaluation, réécriture) — les prompts d'image utilisent deepseek-chat par
# défaut, plus fiable en json_mode strict. Configurable via DEEPSEEK_IMAGE_MODEL.
_DEEPSEEK_IMAGE_MODEL = os.environ.get("DEEPSEEK_IMAGE_MODEL", "deepseek-chat")


# ── Champs requis dans la réponse LLM ────────────────────────────────────────
# L'ORDRE de ces champs EST le raisonnement demandé au LLM : objectif
# narratif, émotion et personnages présents d'abord, puis les 5 champs du
# contrat ImagePrompt.

_REQUIRED_STRING_FIELDS = (
    "goal", "emotion",
    "subject", "scene_description", "style", "prompt", "negative_prompt",
    # Sprint 34.6 — champs granulaires additionnels, utilisés pour construire
    # le "prompt" riche exporté dans image_prompts/scene_XX.json (voir
    # production_package_builder.py) sans dupliquer la logique de génération.
    "appearance", "clothing", "accessories", "pose", "facial_expression",
    "weather", "time_of_day", "background",
)
_REQUIRED_LIST_FIELDS = ("characters",)
_REQUIRED_FIELDS = _REQUIRED_STRING_FIELDS + _REQUIRED_LIST_FIELDS

# ── Garanties déterministes de rendu (Sprint 24) ────────────────────────────
# Appliquées au champ "style" (rendu artistique) — pas au champ "prompt"
# (qui ne décrit plus que l'action depuis le Sprint 24.1).

_RENDER_REQUIREMENTS = (
    (("9:16", "vertical", "portrait orientation"), "vertical 9:16 aspect ratio, portrait orientation"),
    (("8k", "hdr", "ultra-detailed", "ultra detailed"), "ultra-detailed, HDR, 8K resolution"),
    # Sprint 37.2 — le style visuel de marque (Sprint 34.6 : "Arcane
    # character design + Lord of Mysteries atmosphere") est stylisé/peint,
    # jamais photoréaliste. Cette garantie forçait auparavant "photorealistic"
    # sur CHAQUE style, en contradiction directe avec l'identité de marque —
    # elle force désormais l'inverse : le registre stylisé peint.
    (
        ("arcane", "painterly", "stylized illustration", "hand-painted"),
        "Arcane character design, painterly stylized illustration, hand-painted textures",
    ),
    (("cinematic",), "cinematic AI animation"),
)

_GENERIC_CHARACTER_LABELS = {"name", "nom", "personnage", "character", "characters"}

_NEGATIVE_PROMPT_BASELINE = (
    "text", "watermark", "logo", "signature", "blurry", "deformed",
    "distorted anatomy", "extra limbs", "low quality", "low resolution",
    # Sprint 37.2 — bannir "photorealistic" au lieu de "cartoon/illustration/
    # painting" : le style de marque EST une illustration peinte stylisée,
    # bannir ces mots poussait systématiquement le rendu vers le photoréalisme.
    "photorealistic", "photo-realistic", "photograph", "real photo", "realistic skin texture",
    "3d render", "oversaturated colors",
    "horizontal crop", "incorrect aspect ratio", "plastic skin", "jpeg artifacts",
    # Sprint 26 — l'image doit ressembler à une scène de film, jamais à un
    # document explicatif ou une slide de présentation.
    "infographic", "diagram", "schematic", "chart", "graph", "user interface",
    "ui elements", "app screenshot", "bullet points", "wall of text",
    "powerpoint slide", "presentation slide", "flowchart", "table layout",
)


# ── Prompt système (Sprint 24.2) ─────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_SYSTEM_PROMPT_BASE = (_PROMPTS_DIR / "image_system_prompt.txt").read_text(encoding="utf-8")


def _build_system_prompt(reasoning_enabled: bool) -> str:
    """
    Construit le prompt système, avec un raisonnement explicite étape par
    étape ajouté UNIQUEMENT quand le modèle actif n'a pas de mode Reasoning
    natif (voir src.llm.supports_reasoning()).
    """
    if reasoning_enabled:
        return _SYSTEM_PROMPT_BASE
    return _SYSTEM_PROMPT_BASE


# ── Garanties déterministes (Sprint 24) ─────────────────────────────────────

def _finalize_style_for_render(style: str) -> str:
    """
    Garantit systématiquement, sans dépendre de la discipline du LLM, que le
    champ "style" mentionne le format vertical 9:16, la qualité HDR/8K et le
    registre photoréaliste cinématographique.
    """
    result = style.strip()
    lowered = result.lower()
    for markers, suffix in _RENDER_REQUIREMENTS:
        if any(marker in lowered for marker in markers):
            continue
        result = f"{result.rstrip('. ').strip()}, {suffix}."
        lowered = result.lower()
    return result


def _finalize_negative_prompt(negative_prompt: str) -> str:
    """Complète le negative_prompt avec la base standard, sans dupliquer."""
    result = negative_prompt.strip()
    lowered = result.lower()
    missing = [term for term in _NEGATIVE_PROMPT_BASELINE if term not in lowered]
    if not missing:
        return result
    return f"{result.rstrip(', ').strip()}, {', '.join(missing)}"


# ── Extraction/nettoyage JSON robustes (Sprint 24.5) ────────────────────────
# Le LLM peut entourer le JSON de texte parasite, de balises <think>...</think>
# (traces de raisonnement qui fuitent dans le contenu malgré json_mode), de
# blocs Markdown, ou de petites erreurs de format (virgules trainantes,
# caractères de contrôle, faux commentaires). Ces helpers isolent et
# réparent le JSON avant json.loads(), pour rendre le fallback heuristique
# exceptionnel plutôt que systématique au moindre écart de formatage.

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
    """
    Isole le premier objet JSON complet en comptant les accolades (en
    ignorant celles qui apparaissent à l'intérieur de chaînes), pour gérer
    le texte parasite avant/après même quand il contient lui-même des
    accolades ou des fragments qui ressemblent à du JSON.
    """
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
    # Non équilibré (réponse tronquée) — renvoyé tel quel, json.loads échouera
    # proprement et sera classifié comme JSON incomplet par _parse_and_validate().
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
    "Respecte exactement le schema demande (tous les champs requis, dans le meme ordre, JSON valide et complet)."
)

# Sprint 30.2 — un "validation_failed" n'est jamais un problème de syntaxe
# JSON (le JSON était déjà valide et parsable) : le message générique
# ci-dessus ne dit rien du champ qui pose réellement problème, donc le LLM
# renvoie souvent exactement la même structure invalide. Cette instruction
# transmet le message de validation réel (quel champ, pourquoi) et rappelle
# explicitement le format attendu pour "characters" — seul champ liste du
# contrat — pour que la correction soit sémantique, pas seulement syntaxique.
_VALIDATION_REPAIR_INSTRUCTION_TEMPLATE = (
    "Validation failed.\n"
    "{detail}\n"
    "The field \"characters\" MUST be an array of plain strings — one complete "
    "descriptive sentence per character (combine name, age, hair, clothing, "
    "accessories and body type into a single string). Never return JSON objects "
    "inside this field.\n"
    "Do not modify any other field.\n"
    "Return valid JSON only, respecting exactly the same fields, in the same order."
)


def _build_repair_instruction(error: "_ImageJsonError") -> str:
    """
    Choisit l'instruction de réparation selon la cause réelle de l'échec
    (Sprint 30.2). Une correction de syntaxe JSON générique ne corrige rien
    quand le JSON était déjà valide — seule sa structure était non conforme.
    """
    if error.reason == "validation_failed":
        return _VALIDATION_REPAIR_INSTRUCTION_TEMPLATE.format(detail=str(error))
    return _JSON_REPAIR_INSTRUCTION


class _ImageJsonError(RuntimeError):
    """Erreur typée pour classifier précisément la cause d'un échec de génération."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"[{reason}] {detail}" if detail else reason)
        self.reason = reason


class LLMImageGenerator(ImageGenerator):
    """
    Générateur d'image piloté par LLM — produit un ImagePrompt (contrat
    universel Sprint 24.1) cohérent sur l'ensemble d'un script, optimisé
    pour Google Whisk / Nano Banana.

    Implémente l'interface ImageGenerator (generate(scene, plan) → GeneratedImage)
    pour rester interchangeable dans ImageEngine. Pour l'entrée « riche »
    explicite (contrat ImagePrompt complet), utiliser directement
    generate_from_scenes().

    Cohérence visuelle : une même instance accumule une « bible de
    personnages » (self._characters_bible) au fil des appels successifs sur
    les scènes d'UN MÊME script, dans l'ordre — instancier un nouveau
    générateur par script (ou appeler reset_continuity()) pour repartir
    d'une bible vide.
    """

    def __init__(
        self,
        script: Optional[Script] = None,
        brand_profile: Optional[BrandProfile] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1536,
        max_retries: int = 2,
        fallback_generator: Optional[ImageGenerator] = None,
    ) -> None:
        self._script = script
        self._brand_profile = brand_profile
        self._provider_name = provider_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._fallback = fallback_generator or HeuristicImageGenerator()

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
        return f"llm_image_{self._provider_name or 'auto'}{model_part}"

    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques cumulées du générateur (lecture seule)."""
        stats = dict(self._stats)
        stats["fallback_reasons"] = dict(self._stats["fallback_reasons"])
        return stats

    @property
    def characters_bible(self) -> Dict[str, str]:
        """Bible de personnages verrouillée (lecture seule) — Sprint 24.3."""
        return dict(self._characters_bible)

    def reset_continuity(self) -> None:
        """Vide la bible de personnages — à appeler entre deux scripts distincts."""
        self._characters_bible = {}

    def _resolve_model(self) -> Optional[str]:
        """
        Résout le modèle à utiliser pour build_llm().

        Un `model=` explicite au constructeur est toujours prioritaire.
        Sinon, si le provider résolu (explicite ou auto-détecté) est
        DeepSeek, utilise _DEEPSEEK_IMAGE_MODEL (par défaut deepseek-reasoner)
        pour bénéficier de son mode Reasoning natif.
        """
        if self._model is not None:
            return self._model

        provider = self._provider_name or (
            "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else None
        )
        if provider == "deepseek":
            return _DEEPSEEK_IMAGE_MODEL
        return None

    # ── Interface ImageGenerator ───────────────────────────────────────────────

    def generate(self, scene: VisualScene, plan: VisualPlan) -> GeneratedImage:
        """
        Point d'entrée conforme à ImageGenerator — retrouve la ScriptScene
        correspondante (via self._script) et l'identité de marque
        (via self._brand_profile), génère un ImagePrompt puis le convertit
        en GeneratedImage (contrat d'ImageEngine, non modifié).
        """
        script_scene = self._find_script_scene(scene.scene_order)
        if script_scene is None or self._brand_profile is None:
            logger.debug(
                "LLMImageGenerator : contexte insuffisant (script/brand non configurés) "
                "pour la scène #%d — fallback %s.",
                scene.scene_order, self._fallback.name,
            )
            return self._fallback.generate(scene, plan)

        image_prompt = self.generate_from_scenes(script_scene, scene, self._brand_profile, script=self._script)
        return self._to_generated_image(image_prompt, scene, self._brand_profile)

    def _find_script_scene(self, scene_order: int) -> Optional[ScriptScene]:
        if self._script is None:
            return None
        for scene in self._script.scenes:
            if scene.order == scene_order:
                return scene
        return None

    # ── Entrée riche — contrat ImagePrompt (Sprint 24.1) ─────────────────────────

    def generate_from_scenes(
        self,
        script_scene: ScriptScene,
        visual_scene: VisualScene,
        brand_profile: BrandProfile,
        script: Optional[Script] = None,
    ) -> ImagePrompt:
        """
        Génère un ImagePrompt (contrat universel) pour une scène, avec
        cohérence visuelle sur l'ensemble du `script` fourni (Sprint 24.3).

        Args:
            script_scene: Scène du script (narration, contexte).
            visual_scene: Scène visuelle (composition, éclairage de base, palette).
            brand_profile: Identité de marque (style, ton).
            script: Script complet, pour la continuité narrative (personnages,
                ordre des scènes). Si omis, utilise self._script si disponible ;
                sinon la scène est traitée isolément (pas de continuité).

        Returns:
            ImagePrompt — retombe sur une version dégradée construite à
            partir de HeuristicImageGenerator si toutes les tentatives LLM échouent.
        """
        effective_script = script if script is not None else self._script

        last_reason = "unknown"
        last_detail = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._try_generate_llm(effective_script, script_scene, visual_scene, brand_profile)
            except Exception as exc:
                last_reason = getattr(exc, "reason", "unknown") or "unknown"
                last_detail = str(exc)
                logger.warning(
                    "LLM Image Generator — tentative %d/%d échouée (scène #%d, raison=%s) : %s",
                    attempt, self._max_retries, visual_scene.scene_order, last_reason, exc,
                )
                self._stats["llm_failures"] += 1

        logger.info(
            "LLM Image Generator — fallback vers %s pour la scène #%d (raison=%s : %s)",
            self._fallback.name, visual_scene.scene_order, last_reason, last_detail,
        )
        self._stats["fallbacks"] += 1
        self._stats["fallback_reasons"][last_reason] = self._stats["fallback_reasons"].get(last_reason, 0) + 1
        plan = VisualPlan(
            title=script_scene.scene.description.setting[:60],
            style=brand_profile.visual_style, scenes=[visual_scene],
        )
        fallback_image = self._fallback.generate(visual_scene, plan)
        return self._from_generated_image(fallback_image, reason=last_reason, detail=last_detail)

    # ── Logique LLM ────────────────────────────────────────────────────────────

    def _try_generate_llm(
        self,
        script: Optional[Script],
        script_scene: ScriptScene,
        visual_scene: VisualScene,
        brand_profile: BrandProfile,
    ) -> ImagePrompt:
        if self._provider is None:
            self._provider = build_llm(provider=self._provider_name, model=self._resolve_model())
            logger.info("LLMImageGenerator utilise %s / %s", self._provider.name, self._provider.model)

        reasoning_enabled = supports_reasoning(self._provider.model)
        system_prompt = _build_system_prompt(reasoning_enabled)

        user_prompt = self._build_user_prompt(script, script_scene, visual_scene, brand_profile)
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]

        response, elapsed_ms = self._call_llm(messages)
        self._raise_if_api_error(response)

        try:
            data = self._parse_and_validate(response)
        except _ImageJsonError as first_err:
            logger.warning(
                "LLM Image Generator — JSON invalide (raison=%s) pour la scène #%d — "
                "tentative de correction intelligente.",
                first_err.reason, visual_scene.scene_order,
            )
            self._stats["json_repair_attempts"] += 1
            repair_messages = messages + [
                LLMMessage(role="assistant", content=response.content[:4000]),
                LLMMessage(role="user", content=_build_repair_instruction(first_err)),
            ]
            repair_response, repair_elapsed_ms = self._call_llm(repair_messages)
            self._raise_if_api_error(repair_response)
            try:
                data = self._parse_and_validate(repair_response)
            except _ImageJsonError as second_err:
                logger.warning(
                    "LLM Image Generator — correction JSON échouée (raison=%s) pour la scène #%d.",
                    second_err.reason, visual_scene.scene_order,
                )
                raise
            response, elapsed_ms = repair_response, repair_elapsed_ms
            self._stats["json_repairs_success"] += 1
            logger.info(
                "LLM Image Generator — JSON corrigé avec succès pour la scène #%d.",
                visual_scene.scene_order,
            )

        image_prompt = self._build_image_prompt(data, response, elapsed_ms)
        self._update_characters_bible(data["characters"])

        self._stats["llm_success"] += 1
        logger.info(
            "LLM Image OK — scène #%d, %d tokens, $%.6f, %d ms, provider=%s, model=%s, reasoning=%s",
            visual_scene.scene_order, response.total_tokens, response.cost_usd,
            elapsed_ms, response.provider_name, response.model, reasoning_enabled,
        )
        return image_prompt

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
        raise _ImageJsonError(reason, response.content[:200])

    @classmethod
    def _parse_and_validate(cls, response: Any) -> Dict[str, Any]:
        """
        Extrait, nettoie, parse et valide le JSON d'une réponse LLM —
        classifie précisément la cause d'un éventuel échec (Sprint 24.5).
        """
        content = (response.content or "").strip()
        if not content:
            raise _ImageJsonError("empty_response", "réponse vide")

        json_str = cls._extract_json(content)
        incomplete = getattr(response, "finish_reason", None) == "length"

        if not json_str:
            raise _ImageJsonError(
                "json_incomplete" if incomplete else "json_invalid",
                "aucun objet JSON isolable dans la réponse",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise _ImageJsonError(
                "json_incomplete" if incomplete else "json_invalid", str(exc)
            ) from exc

        try:
            cls._validate_json_structure(data)
        except ValueError as exc:
            raise _ImageJsonError("validation_failed", str(exc)) from exc

        return data

    # ── Continuité narrative (Sprint 24.3) ───────────────────────────────────────

    def _build_continuity_block(self, script: Optional[Script], current_order: int) -> str:
        """
        Construit le bloc « CONTINUITE NARRATIVE » injecté dans le prompt
        utilisateur : le script complet (dans l'ordre) pour situer la scène
        courante, et la bible de personnages déjà verrouillés.
        """
        if script is None:
            return ""

        lines: List[str] = ["", "=== NARRATIVE CONTINUITY (full script, in order) ==="]
        for scene in sorted(script.scenes, key=lambda s: s.order):
            marker = " <<< SCENE ACTUELLE" if scene.order == current_order else ""
            lines.append(f"  [{scene.order}] {scene.narration_text}{marker}")

        if self._characters_bible:
            lines.append("")
            lines.append("=== CHARACTERS ALREADY ESTABLISHED (reuse EXACTLY if they reappear) ===")
            for name, desc in self._characters_bible.items():
                lines.append(f"  - {name} : {desc}")
        else:
            lines.append("")
            lines.append("(No character established yet — if this scene introduces one, describe them precisely and stably for future scenes.)")

        return "\n".join(lines)

    def _update_characters_bible(self, characters: List[Any]) -> None:
        """
        Verrouille la description de chaque NOUVEAU personnage rencontré.
        Les personnages déjà connus conservent leur description d'origine
        (jamais écrasée), pour garantir une apparence stable dans le temps.

        Le LLM répond parfois par un label générique avant le premier ":"
        (ex: "Name: Young man. Description: ...") plutôt que par le nom du
        personnage lui-même. Dans ce cas, le label ("Name"/"Nom"/...) ferait
        une mauvaise clé de bible — on retombe alors sur la chaîne entière
        comme clé, pour éviter de polluer la bible avec des labels génériques.
        """
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

    # ── Construction du prompt ─────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        script: Optional[Script],
        script_scene: ScriptScene,
        visual_scene: VisualScene,
        brand_profile: BrandProfile,
    ) -> str:
        desc = script_scene.scene.description
        lines: List[str] = [
            "Write the ImagePrompt contract for the following scene.",
            "First reason about the narrative goal, the emotion, and the characters present, "
            "before writing subject/scene_description/style/prompt.",
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
            "",
            "=== VISUAL PLAN (base) ===",
            f"  Shot type      : {visual_scene.shot_type}",
            f"  Camera motion  : {visual_scene.camera_motion}",
            f"  Composition    : {visual_scene.composition}",
            f"  Lighting       : {visual_scene.lighting}",
            f"  Palette        : {', '.join(visual_scene.color_palette) or 'unspecified'}",
            "",
            "=== BRAND IDENTITY ===",
            f"  Brand       : {brand_profile.name}",
            f"  Tone        : {brand_profile.tone}",
            f"  Visual style: {brand_profile.visual_style}",
        ]
        lines.append(self._build_continuity_block(script, script_scene.order))
        lines += ["", "Generate the ImagePrompt contract JSON now."]
        return "\n".join(lines)

    # ── Extraction et validation JSON ──────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extrait un objet JSON exploitable d'une réponse LLM potentiellement
        « sale » (Sprint 24.5) : retire les balises <think>...</think>, les
        blocs Markdown ```json ... ```, isole le premier objet JSON complet
        par comptage d'accolades (robuste au texte parasite avant/après),
        puis nettoie les caractères de contrôle / faux commentaires /
        virgules traînantes.
        """
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
            if not isinstance(data[field_name], list):
                raise ValueError(f"Le champ '{field_name}' doit être une liste")
            if not all(isinstance(item, str) for item in data[field_name]):
                raise ValueError(
                    f"Le champ '{field_name}' doit être une liste de chaînes "
                    "(le LLM a renvoyé des objets structurés au lieu de descriptions textuelles)"
                )

    # ── Construction ImagePrompt / GeneratedImage ────────────────────────────────

    @staticmethod
    def _build_image_prompt(data: Dict[str, Any], response: Any, elapsed_ms: int) -> ImagePrompt:
        """Construit l'ImagePrompt (contrat Sprint 24.1) à partir du JSON validé."""
        return ImagePrompt(
            subject=data["subject"].strip(),
            scene_description=data["scene_description"].strip(),
            style=_finalize_style_for_render(data["style"]),
            prompt=data["prompt"].strip(),
            negative_prompt=_finalize_negative_prompt(data["negative_prompt"]),
            metadata={
                "goal": data["goal"].strip(),
                "emotion": data["emotion"].strip(),
                "characters": list(data["characters"]),
                "appearance": data["appearance"].strip(),
                "clothing": data["clothing"].strip(),
                "accessories": data["accessories"].strip(),
                "pose": data["pose"].strip(),
                "facial_expression": data["facial_expression"].strip(),
                "weather": data["weather"].strip(),
                "time_of_day": data["time_of_day"].strip(),
                "background": data["background"].strip(),
                "provider": response.provider_name,
                "model": response.model,
                "time_ms": elapsed_ms,
                "cost_usd": round(response.cost_usd, 6),
            },
        )

    @staticmethod
    def _from_generated_image(
        generated_image: GeneratedImage, reason: str = "unknown", detail: str = "",
    ) -> ImagePrompt:
        """
        Enveloppe un GeneratedImage (résultat du fallback heuristique) dans
        un ImagePrompt dégradé, pour garder un contrat de sortie uniforme.

        `reason` (Sprint 24.5) documente pourquoi le fallback a été déclenché
        (ex: "json_invalid", "json_incomplete", "timeout", "api_error",
        "validation_failed", "empty_response") — utile pour le diagnostic,
        sans altérer le contrat des scènes générées avec succès par le LLM.

        `detail` (Sprint 29.1) porte le message d'erreur précis à l'origine
        du fallback (ex: "Champ obligatoire manquant : 'prompt'", "Expecting
        value: line 1 column 1 (char 0)") — `reason` seul ne suffisait pas à
        comprendre immédiatement la cause réelle d'une validation échouée.

        Le générateur heuristique produit toujours un `prompt` complet et
        exploitable (Sprint 29.1) : on le réutilise ici comme champ `prompt`
        ET `scene_description` plutôt que de laisser `prompt` vide, pour
        qu'aucun ImagePrompt de secours ne soit inutilisable en production.
        """
        return ImagePrompt(
            subject="",
            scene_description=generated_image.prompt,
            style=generated_image.style,
            prompt=generated_image.prompt,
            negative_prompt=generated_image.negative_prompt,
            metadata={
                "goal": "",
                "emotion": "",
                "characters": [],
                "appearance": "", "clothing": "", "accessories": "", "pose": "",
                "facial_expression": "", "weather": "", "time_of_day": "", "background": "",
                "provider": generated_image.provider,
                "model": "",
                "time_ms": 0,
                "cost_usd": 0.0,
                "fallback_reason": reason,
                "fallback_detail": detail,
            },
        )

    @staticmethod
    def _to_generated_image(
        image_prompt: ImagePrompt,
        visual_scene: VisualScene,
        brand_profile: BrandProfile,
    ) -> GeneratedImage:
        """
        Convertit un ImagePrompt en GeneratedImage — nécessaire UNIQUEMENT
        pour rester compatible avec ImageGenerator/ImageEngine (contrat d'un
        autre moteur, non modifié). Les dimensions/aspect ratio/seed sont
        calculés par les helpers déterministes de HeuristicImageGenerator.
        """
        final_prompt = (
            f"{image_prompt.subject}. {image_prompt.scene_description} "
            f"{image_prompt.style}. {image_prompt.prompt}"
        ).strip()

        plan = VisualPlan(title="", style=brand_profile.visual_style, scenes=[visual_scene])
        aspect_ratio = HeuristicImageGenerator._resolve_aspect_ratio(plan.aspect_ratio, visual_scene)
        width, height = HeuristicImageGenerator._resolve_dimensions(aspect_ratio, visual_scene)
        seed = HeuristicImageGenerator._compute_seed(final_prompt, visual_scene.scene_order)
        quality = HeuristicImageGenerator._resolve_quality(plan, visual_scene)
        steps = HeuristicImageGenerator._resolve_steps(plan, visual_scene)

        return GeneratedImage(
            scene_order=visual_scene.scene_order,
            prompt=final_prompt,
            negative_prompt=image_prompt.negative_prompt,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            seed=seed,
            quality=quality,
            steps=steps,
            style=image_prompt.style,
            color_palette=list(visual_scene.color_palette),
            provider="llm_image_v1",
            metadata={
                "generator": "llm_image_v1",
                "image_prompt": {
                    "subject": image_prompt.subject,
                    "scene_description": image_prompt.scene_description,
                    "style": image_prompt.style,
                    "prompt": image_prompt.prompt,
                    "negative_prompt": image_prompt.negative_prompt,
                    "metadata": image_prompt.metadata,
                },
            },
        )
