
"""
LLM Script Generator — Générateur de scripts piloté par IA (Sprint 17,
fiabilisé Sprint 27 selon la philosophie Sprint 24.5).

Hérite de ScriptGenerator et utilise exclusivement le LLM Provider
pour produire des Scripts créatifs de haute qualité.

Architecture (identique à VisualDirector / LLMImageGenerator / LLMAnimationGenerator) :
  1. Prompt système + utilisateur envoyé au LLM (via build_llm()).
  2. Le LLM répond STRICTEMENT en JSON (json_mode=True).
  3. Extraction JSON robuste (balises <think>, blocs Markdown, texte parasite,
     caractères de contrôle, virgules traînantes) → validation → reconstruction
     de Script + ScriptScene.
  4. Si le JSON est invalide/incomplet : retry intelligent — un second appel
     avec une instruction de correction ciblée est tenté AVANT tout fallback.
  5. Si le LLM échoue malgré tout (JSON invalide, erreur API, timeout) :
     - 1er échec → retry automatique (2ᵉ tentative, si max_retries > 1).
     - dernier échec → fallback vers HeuristicScriptGenerator, avec la raison
       précise de l'échec enregistrée dans les statistiques.

Contrat :
  - Entrée : Opportunity + CreativeBrief + BrandProfile (inchangé).
  - Sortie  : Script (identique au contrat existant).
  - Dépend uniquement de LLMProvider (via build_llm()).
  - N'importe AUCUN moteur interne (ViralityEngine, KnowledgeEngine, etc.).
  - Ne modifie PAS le script_engine.py existant.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.brand_engine import BrandProfile
from src.creative_engine import CreativeBrief
from src.llm import LLMMessage, build_llm
from src.opportunity_engine import Opportunity
from src.script_engine import (
    MAX_SCENE_DURATION_SECONDS as MAX_SCENE_DURATION_SEC,
    Dialogue, Scene, SceneDescription, Script, ScriptGenerator, ScriptScene,
    cap_dialogues_to_duration, estimate_scene_duration,
)

logger = logging.getLogger(__name__)


# ── Modèle DeepSeek par défaut (Sprint 24, fiabilisé Sprint 27) ─────────────
# LLMScriptGenerator utilise systématiquement ce modèle quand le provider
# résolu est DeepSeek (explicite ou auto-détecté) et qu'aucun modèle n'est
# explicitement demandé. deepseek-reasoner s'est révélé peu fiable en
# json_mode strict pour une sortie structurée volumineuse (le raisonnement
# interne peut consommer tout le budget max_tokens avant même d'écrire le
# JSON, laissant une réponse vide) — comme constaté pour les 3 autres moteurs
# LLM au Sprint 24.5, deepseek-chat est le défaut. Configurable via la
# variable d'environnement DEEPSEEK_MODEL_SCRIPT (.env) — un `model=` explicite
# au constructeur reste toujours prioritaire.
_DEEPSEEK_SCRIPT_MODEL = os.environ.get("DEEPSEEK_MODEL_SCRIPT", "deepseek-chat")


# ── Format cible (Sprint 20.1 — qualite Shorts ; Sprint 37 — budget 60s) ────
# Format court et percutant impose pour tous les scripts generes par LLM,
# quelle que soit la duree suggeree par CreativeBrief.duration_seconds.
#
# Sprint 37 : la generation video (outil externe) coute cher par scene — le
# script cible 1 minute MAXIMUM au total. Sprint 37.3 : l'outil externe
# accepte desormais des clips de 10s (au lieu de 8s) — on privilegie donc
# MOINS de scenes, plus longues chacune (6 scenes x 10s = 60s pile), pour
# une histoire plus posee/cohérente, plutot que beaucoup de scenes tres
# courtes. Le plafond par scene protege toujours contre un echec de
# generation video (moins de details a faire tenir dans un seul clip).
#
# MAX_SCENE_DURATION_SEC est un ALIAS de src.script_engine.MAX_SCENE_DURATION_SECONDS
# (importe ci-dessus) — SOURCE UNIQUE, jamais redefinie ici, pour que le
# plafond demande au LLM et celui reellement applique par
# cap_dialogues_to_duration() ne puissent plus jamais diverger (bug
# Sprint 37.3 -> 37.5 : les scenes etaient tronquees a 6s alors que le LLM
# visait 10s).
_TARGET_DURATION_MIN_SEC = 40
_TARGET_DURATION_MAX_SEC = 60
_TARGET_DURATION_SEC = 55  # cible utilisee pour le calcul du nombre de mots
_TARGET_SCENES_MIN = 4
_TARGET_SCENES_MAX = 6

# Nom complet de chaque langue supportée — utilisé pour interpoler l'instruction
# de langue du prompt (Sprint 34 : la langue des repliques suit la marque,
# elle n'est plus hardcodée en français). Mêmes codes que brand_engine._VALID_LANGUAGES.
_LANGUAGE_NAMES = {
    "fr": "French", "en": "English", "es": "Spanish", "pt": "Portuguese",
    "de": "German", "it": "Italian", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean",
}


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


_BANNED_OPENERS = (
    "imagine", "imaginez",
    "dans cette video", "dans cette vidéo",
    "aujourd'hui nous allons", "aujourd'hui, nous allons",
    "bienvenue",
)
_BANNED_CTA_SNIPPETS = (
    "abonne-toi pour plus", "abonnez-vous pour plus", "abonne toi pour plus",
)


# ── Prompt système ───────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_SYSTEM_PROMPT = (_PROMPTS_DIR / "script_system_prompt.txt").read_text(encoding="utf-8")


# ── Extraction/nettoyage JSON robustes (Sprint 24.5, porté ici au Sprint 27) ──
# Le LLM peut entourer le JSON de texte parasite, de balises <think>...</think>
# (traces de raisonnement qui fuitent dans le contenu malgré json_mode), de
# blocs Markdown, ou de petites erreurs de format (virgules trainantes,
# caractères de contrôle, faux commentaires). Ce module reste volontairement
# indépendant des autres moteurs LLM (aucun import croisé) et duplique ces
# quelques helpers de nettoyage, minimes et sans état.

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
    "Respecte exactement le schema demande (memes champs, JSON valide et complet)."
)


class _ScriptJsonError(RuntimeError):
    """Erreur typée pour classifier précisément la cause d'un échec de génération."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"[{reason}] {detail}" if detail else reason)
        self.reason = reason


class LLMScriptGenerator(ScriptGenerator):
    """
    Générateur de scripts piloté par LLM.

    Utilise exclusivement le LLM Provider (via build_llm()) pour produire
    des scripts créatifs de haute qualité. Le LLM répond en JSON structuré
    qui est directement converti en objet Script.

    Mécanisme de résilience (identique aux 3 autres moteurs LLM du projet) :
      - Extraction JSON robuste (balises <think>, Markdown, texte parasite,
        caractères de contrôle, virgules traînantes).
      - Retry de réparation JSON intelligent avant tout fallback.
      - 2 tentatives complètes maximum (max_retries).
      - Fallback automatique vers HeuristicScriptGenerator si le LLM échoue,
        avec la raison précise de l'échec enregistrée dans les statistiques.
      - Validation stricte du JSON de réponse.
      - Logging de tous les échecs pour analyse.

    Utilisation :
        generator = LLMScriptGenerator()
        script = generator.generate(opportunity, brief, brand_profile)

    Ou avec ScriptEngine :
        engine = ScriptEngine(generator=LLMScriptGenerator())
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 2,
        fallback_generator: Optional[ScriptGenerator] = None,
    ) -> None:
        """
        Args:
            provider_name: Provider LLM à utiliser (None = auto-détection).
            model: Modèle spécifique (None = défaut du provider).
            temperature: Créativité [0.0 – 2.0].
            max_tokens: Tokens max en sortie (les scripts sont longs).
            max_retries: Nombre de tentatives LLM avant fallback (défaut: 2).
            fallback_generator: Générateur de repli (défaut: HeuristicScriptGenerator).
        """
        self._provider_name = provider_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        if fallback_generator is not None:
            self._fallback = fallback_generator
        else:
            from src.script_engine import HeuristicScriptGenerator
            self._fallback = HeuristicScriptGenerator()

        self._provider = None  # lazy init dans generate()
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
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        }

    @property
    def name(self) -> str:
        model_part = f"/{self._model}" if self._model else ""
        return f"llm_{self._provider_name or 'auto'}{model_part}"

    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques cumulées du générateur (lecture seule)."""
        stats = dict(self._stats)
        stats["fallback_reasons"] = dict(self._stats["fallback_reasons"])
        return stats

    def _resolve_model(self) -> Optional[str]:
        """
        Résout le modèle à utiliser pour build_llm().

        Un `model=` explicite au constructeur est toujours prioritaire.
        Sinon, si le provider résolu (explicite ou auto-détecté) est
        DeepSeek, utilise _DEEPSEEK_SCRIPT_MODEL (Sprint 24) au lieu du
        modèle par défaut du provider.
        """
        if self._model is not None:
            return self._model

        provider = self._provider_name or (
            "deepseek" if os.environ.get("DEEPSEEK_API_KEY") else None
        )
        if provider == "deepseek":
            return _DEEPSEEK_SCRIPT_MODEL
        return None

    # ── Interface ScriptGenerator ─────────────────────────────────────────────

    def generate(
        self,
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
    ) -> Script:
        """
        Génère un Script via LLM avec fallback automatique.

        Pipeline :
          1. Construire le prompt utilisateur avec les données d'entrée.
          2. Appeler le LLM (json_mode=True).
          3. Valider le JSON de réponse.
          4. Reconstruire Script + ScriptScene.
          5. En cas d'échec → retry (max 2).
          6. Si toujours en échec → fallback vers HeuristicScriptGenerator.

        Args:
            opportunity: Opportunité détectée.
            creative_brief: Brief créatif.
            brand_profile: Profil de marque.

        Returns:
            Script structuré (LLM ou heuristique).
        """
        # ── Phase 1 : Tentative LLM ────────────────────────────────────────────
        last_reason = "unknown"
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._try_generate_llm(
                    opportunity=opportunity,
                    creative_brief=creative_brief,
                    brand_profile=brand_profile,
                    attempt=attempt,
                )
            except Exception as exc:
                last_reason = getattr(exc, "reason", "unknown") or "unknown"
                logger.warning(
                    "LLM Script Generator — tentative %d/%d échouée (raison=%s) : %s",
                    attempt, self._max_retries, last_reason, exc,
                )
                self._stats["llm_failures"] += 1

        # ── Phase 2 : Fallback ─────────────────────────────────────────────────
        logger.info(
            "LLM Script Generator — fallback vers %s pour '%s' (raison=%s)",
            self._fallback.name,
            creative_brief.title[:50],
            last_reason,
        )
        self._stats["fallbacks"] += 1
        self._stats["fallback_reasons"][last_reason] = self._stats["fallback_reasons"].get(last_reason, 0) + 1
        return self._fallback.generate(opportunity, creative_brief, brand_profile)

    # ── Logique LLM ────────────────────────────────────────────────────────────

    def _try_generate_llm(
        self,
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
        attempt: int = 1,
    ) -> Script:
        """Tente une génération via LLM. Lève une exception en cas d'échec."""
        if self._provider is None:
            self._provider = build_llm(
                provider=self._provider_name,
                model=self._resolve_model(),
            )
            logger.info(
                "LLMScriptGenerator utilise %s / %s",
                self._provider.name, self._provider.model,
            )

        # ── Construire le prompt utilisateur ───────────────────────────────────
        user_prompt = self._build_user_prompt(
            opportunity=opportunity,
            creative_brief=creative_brief,
            brand_profile=brand_profile,
        )

        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        # ── Appel LLM ─────────────────────────────────────────────────────────
        response, elapsed_ms = self._call_llm(messages)
        self._raise_if_api_error(response)

        # ── Extraction, nettoyage, parsing et validation du JSON ───────────────
        # Sur échec (JSON vide/tronqué/invalide/structure incorrecte), un
        # second appel correctif ciblé est tenté AVANT tout fallback
        # (Sprint 24.5 — identique à VisualDirector / LLMImageGenerator /
        # LLMAnimationGenerator).
        try:
            data = self._parse_and_validate(response)
        except _ScriptJsonError as first_err:
            logger.warning(
                "LLM Script Generator — JSON invalide (raison=%s) — tentative de correction intelligente.",
                first_err.reason,
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
            except _ScriptJsonError as second_err:
                logger.warning(
                    "LLM Script Generator — correction JSON échouée (raison=%s).",
                    second_err.reason,
                )
                raise
            response, elapsed_ms = repair_response, repair_elapsed_ms
            self._stats["json_repairs_success"] += 1
            logger.info("LLM Script Generator — JSON corrigé avec succès.")

        # ── Validation de la durée (Sprint 20.1 — format Shorts fixe) ──────────
        # Sprint 32.1 : le LLM ne fournit plus duration_seconds — la durée
        # estimée ici vient de estimate_scene_duration() (même logique que
        # la reconstruction finale), à partir des repliques de chaque scène.
        target_sec = _TARGET_DURATION_SEC
        estimated_dur = sum(
            estimate_scene_duration(
                [Dialogue(personnage=str(d.get("personnage", "")), replique=str(d.get("replique", "")))
                 for d in s.get("dialogues", [])]
            )
            for s in data["scenes"]
        )
        total_words = sum(
            len(d.get("replique", "").split())
            for s in data["scenes"]
            for d in s.get("dialogues", [])
        )

        dur_diff_pct = abs(estimated_dur - target_sec) / max(target_sec, 1) * 100
        dur_out_of_range = not (_TARGET_DURATION_MIN_SEC <= estimated_dur <= _TARGET_DURATION_MAX_SEC)
        expected_words = round(target_sec * 150 / 60)
        word_diff_pct = abs(total_words - expected_words) / max(expected_words, 1) * 100

        logger.info(
            "Validation durée : cible=%ds [%d-%d], estimée=%ds (écart=%.1f%%), mots=%d (attendu=%d, écart=%.1f%%)",
            target_sec, _TARGET_DURATION_MIN_SEC, _TARGET_DURATION_MAX_SEC,
            estimated_dur, dur_diff_pct, total_words, expected_words, word_diff_pct,
        )

        # Si hors de la fourchette [100, 130]s ET qu'on n'a pas déjà fait une tentative de correction
        if (dur_out_of_range or word_diff_pct > 30.0) and attempt == 1:
            logger.warning(
                "Validation durée échouée (%ds hors de [%d, %d]s) — seconde génération corrective",
                estimated_dur, _TARGET_DURATION_MIN_SEC, _TARGET_DURATION_MAX_SEC,
            )
            # Seconde génération avec un message correctif — Sprint 32.1 : la
            # duree n'est plus un champ que le LLM ecrit, seul le nombre de
            # mots des dialogues la determine (estimate_scene_duration()).
            correction_msg = (
                f"⚠️ CORRECTION : Le script precedent representait environ {estimated_dur} "
                f"secondes de parole ({total_words} mots au total), mais la duree cible doit etre "
                f"entre {_TARGET_DURATION_MIN_SEC} et {_TARGET_DURATION_MAX_SEC} secondes "
                f"(environ {expected_words} mots).\n"
                f"Reecris le script en respectant STRICTEMENT ces contraintes :\n"
                f"- Total de mots parles (toutes repliques confondues) ≈ {expected_words}\n"
                f"- Ajuste le nombre de scenes ({_TARGET_SCENES_MIN}-{_TARGET_SCENES_MAX}) et la longueur des dialogues.\n"
                f"- Ne fournis toujours PAS de champ duration_seconds — il est calcule automatiquement."
            )
            # Ajouter le message de correction au prompt
            messages.append(LLMMessage(role="user", content=correction_msg))

            # Ré-appel LLM avec une temperature legerement plus creative pour debloquer
            response2, elapsed_ms2 = self._call_llm(
                messages, temperature=self._temperature + 0.1,
            )
            self._raise_if_api_error(response2)
            data = self._parse_and_validate(response2)
            elapsed_ms = elapsed_ms + elapsed_ms2

            # Re-vérifier la durée après correction
            estimated_dur2 = sum(
                estimate_scene_duration(
                    [Dialogue(personnage=str(d.get("personnage", "")), replique=str(d.get("replique", "")))
                     for d in s.get("dialogues", [])]
                )
                for s in data["scenes"]
            )
            total_words2 = sum(
                len(d.get("replique", "").split())
                for s in data["scenes"]
                for d in s.get("dialogues", [])
            )

            dur_diff_pct2 = abs(estimated_dur2 - target_sec) / max(target_sec, 1) * 100
            logger.info(
                "Correction durée : cible=%ds, estimée=%ds (écart=%.1f%%), mots=%d",
                target_sec, estimated_dur2, dur_diff_pct2, total_words2,
            )

        # ── Reconstruction du Script ───────────────────────────────────────────
        try:
            script = self._build_script_from_json(
                data=data,
                opportunity=opportunity,
                creative_brief=creative_brief,
                brand_profile=brand_profile,
                response_time_ms=elapsed_ms,
                response_tokens=response.total_tokens,
                response_cost=response.cost_usd,
                llm_provider=response.provider_name,
                llm_model=response.model,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _ScriptJsonError(
                "validation_failed",
                f"Erreur de reconstruction Script : {exc}\n"
                f"Données JSON : {json.dumps(data, ensure_ascii=False)[:500]}",
            )

        self._stats["llm_success"] += 1
        logger.info(
            "LLM Script OK — '%s' (%d scènes, %d tokens, $%.6f, %d ms, provider=%s, model=%s) — tentative %d",
            script.title[:50],
            len(script.scenes),
            response.total_tokens,
            response.cost_usd,
            elapsed_ms,
            response.provider_name,
            response.model,
            attempt,
        )
        return script

    def _call_llm(self, messages: List[LLMMessage], temperature: Optional[float] = None):
        """Appelle le LLM et accumule les statistiques (temps, coût, tokens, nombre d'appels)."""
        start = time.time()
        self._stats["llm_calls"] += 1
        response = self._provider.generate(
            messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=self._max_tokens,
            json_mode=True,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        self._stats["total_time_ms"] += elapsed_ms
        self._stats["total_cost_usd"] += response.cost_usd
        self._stats["total_prompt_tokens"] += response.prompt_tokens
        self._stats["total_completion_tokens"] += response.completion_tokens
        return response, elapsed_ms

    @staticmethod
    def _raise_if_api_error(response: Any) -> None:
        if response.finish_reason != "error":
            return
        reason = "timeout" if "timeout" in response.content.lower() else "api_error"
        raise _ScriptJsonError(reason, response.content[:200])

    @classmethod
    def _parse_and_validate(cls, response: Any) -> Dict[str, Any]:
        """
        Extrait, nettoie, parse et valide le JSON d'une réponse LLM —
        classifie précisément la cause d'un éventuel échec (Sprint 24.5).
        """
        content = (response.content or "").strip()
        if not content:
            raise _ScriptJsonError("empty_response", "réponse vide")

        json_str = cls._extract_json(content)
        incomplete = getattr(response, "finish_reason", None) == "length"

        if not json_str:
            raise _ScriptJsonError(
                "json_incomplete" if incomplete else "json_invalid",
                "aucun objet JSON isolable dans la réponse",
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise _ScriptJsonError(
                "json_incomplete" if incomplete else "json_invalid", str(exc)
            ) from exc

        try:
            cls._validate_json_structure(data)
        except ValueError as exc:
            raise _ScriptJsonError("validation_failed", str(exc)) from exc

        return data

    # ── Construction du prompt ─────────────────────────────────────────────────

    def _build_user_prompt(
        self,
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
    ) -> str:
        """Construit le prompt utilisateur à partir des données d'entrée (en anglais, Sprint 32.1)."""
        # Sprint 20.1 : cible fixe format Shorts (Sprint 37.3 : 40-60s / 4-6 scenes),
        # independamment de creative_brief.duration_seconds (format long/standard).
        target_sec = _TARGET_DURATION_SEC
        target_words = round(target_sec * 150 / 60)
        language_name = _language_name(brand_profile.primary_language)

        lines: List[str] = [
            "Create an original, captivating YouTube Shorts script. All production fields "
            f"(title, scene descriptions, transitions) must be written in ENGLISH — ONLY the "
            f"spoken dialogues/repliques must be written in {language_name}.",
            "DRAW INSPIRATION from the elements below but DO NOT COPY THEM. Be creative!",
            "",
            f"### TARGET TOTAL DURATION: {target_sec} seconds (Shorts format, {_TARGET_DURATION_MIN_SEC}-{_TARGET_DURATION_MAX_SEC}s) ###",
            f"The TOTAL spoken word count across every replique must be approximately {target_words} words "
            f"(150 words/minute = 2.5 words per second). You do NOT provide duration_seconds — it is computed automatically afterward.",
            "",
            "Here is an example breakdown to guide you (adapt to your actual scene count):",
            self._build_duration_breakdown(target_sec),
            "",
            "=== NICHE / TOPIC ===",
            f"  Topic          : {opportunity.niche}",
            f"  Source video title : {opportunity.title}",
            f"  Potential score    : {opportunity.overall_score}/100",
            "",
        ]

        sequel_hint = opportunity.metadata.get("sequel_of")
        if sequel_hint:
            lines += [
                "=== CONTINUATION CONTEXT (Sprint 33 — topic_history) ===",
                f"A closely related topic was already covered recently: \"{sequel_hint.get('title', '')}\" "
                f"(published {sequel_hint.get('date', 'recently')}).",
                "Do NOT repeat that video. Write this one as an explicit CONTINUATION or a genuinely NEW ANGLE: "
                "reference what was already covered only briefly (one sentence at most), then go further — "
                "new facts, the next chapter, a deeper level, or a twist the previous video did not cover.",
                "Make the continuation clear from the hook itself, without generic transition phrases like "
                "'as promised' or 'as we said before'.",
                "",
            ]

        lines += [
            "=== INSPIRATION (adapt freely) ===",
            f"  Suggested title : {creative_brief.title}",
            f"  Suggested angle : {creative_brief.angle} (you MAY choose a different one if relevant)",
            f"  Hook            : {creative_brief.hook}",
            f"  Promise         : {creative_brief.promise}",
            f"  Audience        : {creative_brief.audience}",
            f"  Suggested CTA   : {creative_brief.cta} (rephrase it to be specific to the topic if it is generic)",
            f"  Emotion         : {creative_brief.emotion}",
            "",
            "=== BRAND PROFILE ===",
            f"  Brand          : {brand_profile.name}",
            f"  Tone           : {brand_profile.tone}",
            f"  Language       : {brand_profile.primary_language}",
            f"  Preferred duration (long format, not applicable here): {brand_profile.preferred_video_duration} seconds",
            f"  Audience       : {brand_profile.target_audience}",
            "",
            "=== MANDATORY REQUIREMENTS (Shorts format) ===",
            f"  - SCENE COUNT: between {_TARGET_SCENES_MIN} and {_TARGET_SCENES_MAX} scenes MAXIMUM. Fewer is fine, never more.",
            f"  - WORD COUNT: About {target_words} words total across all repliques combined (~{round(target_words / 2.5)}s of speech), "
            f"NEVER above {round(_TARGET_DURATION_MAX_SEC * 2.5)} words (that would exceed the 1-minute hard limit).",
            f"  - MAX {MAX_SCENE_DURATION_SEC} SECONDS PER SCENE (HARD LIMIT): the repliques of ANY SINGLE scene must never "
            f"add up to more than ~{round(MAX_SCENE_DURATION_SEC * 2.5)} words (150 words/minute). This is a production "
            "constraint — the video-generation tool can fail or need a costly retry if one scene carries too much dialogue. "
            "Split a long beat into two shorter scenes instead of writing one long scene.",
            "  - HOOK IN 3 SECONDS: one short line that opens a curiosity gap from the very first word.",
            "  - STRICTLY FORBIDDEN to open with 'Imagine', 'In this video', 'Today we are going to', or 'Welcome' (in any language).",
            "  - FAST PACE: one idea per scene, no filler lines.",
            "  - RETENTION: use pattern interrupts, tension, payoff, and escalation across scenes.",
            "  - TOPIC-SPECIFIC CTA: FORBIDDEN to write a generic 'Subscribe for more' or any variant.",
            f"  - Respect the brand's tone: '{brand_profile.tone}'",
            "  - INVENT an original title (do not copy the suggested title) — write it in ENGLISH",
            "  - Structure the scenes in a surprising, natural way",
            "  - Every 'scene.description' field must be extremely rich (setting, composition, characters, lighting, camera, mood, symbolism, director_notes, viewer_emotion) — write it in ENGLISH",
            "  - Every 'transition' field must be written in ENGLISH",
            f"  - Every 'replique' must sound natural, credible, and cinematic — write it in {language_name.upper()}",
            "  - Do NOT include a 'duration_seconds' field anywhere",
            "  - Respond ONLY with valid JSON, no text before/after",
            "",
            "Generate the script JSON now.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_duration_breakdown(target_sec: int) -> str:
        """
        Génère une répartition indicative de la durée entre les scenes —
        Sprint 37 : chaque scene (hook et CTA inclus, ce sont juste la
        premiere et la derniere scene) est plafonnee a MAX_SCENE_DURATION_SEC.
        """
        n_scenes = min(_TARGET_SCENES_MAX, max(_TARGET_SCENES_MIN, -(-target_sec // MAX_SCENE_DURATION_SEC)))
        return (
            f"Hook (scene 1): {MAX_SCENE_DURATION_SEC}s max | "
            f"{n_scenes - 2} development scenes: {MAX_SCENE_DURATION_SEC}s max EACH | "
            f"CTA (last scene): {MAX_SCENE_DURATION_SEC}s max "
            f"| TOTAL ≈ {target_sec}s across {n_scenes} scenes (never exceed {MAX_SCENE_DURATION_SEC}s on any single scene)"
        )

    # ── Extraction et validation JSON ──────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extrait un objet JSON exploitable d'une réponse LLM potentiellement
        « sale » (Sprint 24.5) : retire les balises <think>...</think>, les
        blocs Markdown ```json ... ```, isole le premier objet JSON complet
        par comptage d'accolades (robuste au texte parasite avant/après,
        y compris s'il contient lui-même des accolades), puis nettoie les
        caractères de contrôle / faux commentaires / virgules traînantes.
        """
        text = text.strip()
        text = _strip_think_tags(text)
        text = _strip_code_fence(text)
        text = _isolate_json_object(text)
        text = _clean_json_text(text)
        return text.strip()

    # Les 9 champs obligatoires de SceneDescription (Sprint 32.1).
    _REQUIRED_DESCRIPTION_FIELDS = (
        "setting", "composition", "characters", "lighting", "camera",
        "mood", "symbolism", "director_notes", "viewer_emotion",
    )

    @classmethod
    def _validate_json_structure(cls, data: Dict[str, Any]) -> None:
        """
        Valide la structure du JSON de réponse — storyboard cinématographique
        (Sprint 32.1) : chaque scène porte un objet "scene" imbriqué
        {number, type, description{9 champs}}, des "dialogues", et une
        "transition". Le LLM ne fournit plus "duration_seconds" — elle est
        calculée après coup par estimate_scene_duration().
        """
        required_fields = ["title", "scenes"]

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Champ obligatoire manquant dans la réponse : '{field}'")

        if not isinstance(data["scenes"], list):
            raise ValueError("Le champ 'scenes' doit être une liste")

        if len(data["scenes"]) < _TARGET_SCENES_MIN:
            raise ValueError(
                f"Trop peu de scènes : {len(data['scenes'])} (minimum {_TARGET_SCENES_MIN})"
            )

        if len(data["scenes"]) > _TARGET_SCENES_MAX:
            raise ValueError(
                f"Trop de scènes : {len(data['scenes'])} (maximum {_TARGET_SCENES_MAX})"
            )

        for i, storyboard_scene in enumerate(data["scenes"]):
            if not isinstance(storyboard_scene, dict):
                raise ValueError(f"La scène {i} n'est pas un dictionnaire")
            for field in ("scene", "dialogues", "transition"):
                if field not in storyboard_scene:
                    raise ValueError(f"Champ '{field}' manquant dans la scène {i}")
            if not isinstance(storyboard_scene.get("transition"), str):
                raise ValueError(f"Le champ 'transition' de la scène {i} doit être une chaîne")

            scene_obj = storyboard_scene["scene"]
            if not isinstance(scene_obj, dict):
                raise ValueError(f"Le champ 'scene' de la scène {i} doit être un objet")
            if not isinstance(scene_obj.get("number"), int) or scene_obj["number"] < 1:
                raise ValueError(f"'scene.number' de la scène {i} doit être un entier >= 1")
            if not isinstance(scene_obj.get("type"), str) or not scene_obj["type"].strip():
                raise ValueError(f"'scene.type' de la scène {i} doit être une chaîne non vide")

            description = scene_obj.get("description")
            if not isinstance(description, dict):
                raise ValueError(f"'scene.description' de la scène {i} doit être un objet")
            for field in cls._REQUIRED_DESCRIPTION_FIELDS:
                if field not in description:
                    raise ValueError(
                        f"Champ 'scene.description.{field}' manquant dans la scène {i}"
                    )
                if not isinstance(description[field], str) or not description[field].strip():
                    raise ValueError(
                        f"'scene.description.{field}' de la scène {i} doit être une chaîne non vide"
                    )

            dialogues = storyboard_scene.get("dialogues")
            if not isinstance(dialogues, list) or not dialogues:
                raise ValueError(
                    f"Le champ 'dialogues' de la scène {i} doit être une liste non vide"
                )
            for j, dlg in enumerate(dialogues):
                if not isinstance(dlg, dict):
                    raise ValueError(f"Le dialogue {j} de la scène {i} n'est pas un dictionnaire")
                if not isinstance(dlg.get("personnage"), str) or not dlg["personnage"].strip():
                    raise ValueError(
                        f"Le champ 'personnage' du dialogue {j} de la scène {i} "
                        "doit être une chaîne non vide"
                    )
                if not isinstance(dlg.get("replique"), str) or not dlg["replique"].strip():
                    raise ValueError(
                        f"Le champ 'replique' du dialogue {j} de la scène {i} "
                        "doit être une chaîne non vide"
                    )

    # ── Reconstruction Script ──────────────────────────────────────────────────

    @staticmethod
    def _build_script_from_json(
        data: Dict[str, Any],
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
        response_time_ms: int,
        response_tokens: int,
        response_cost: float,
        llm_provider: str = "",
        llm_model: str = "",
    ) -> Script:
        """
        Reconstruit un objet Script à partir du JSON validé (Sprint 32.1) —
        `duration_seconds` n'est JAMAIS lu depuis le JSON : il est calculé
        ici par estimate_scene_duration(), seule source de vérité.
        """
        scenes_raw: List[Dict[str, Any]] = data["scenes"]

        scenes: List[ScriptScene] = []
        for scene_data in scenes_raw:
            dialogues = [
                Dialogue(personnage=str(d["personnage"]), replique=str(d["replique"]))
                for d in scene_data["dialogues"]
            ]
            scene_obj = scene_data["scene"]
            description = SceneDescription(**{
                field: str(scene_obj["description"][field])
                for field in (
                    "setting", "composition", "characters", "lighting", "camera",
                    "mood", "symbolism", "director_notes", "viewer_emotion",
                )
            })
            scene_number = int(scene_obj["number"])
            capped_dialogues = cap_dialogues_to_duration(dialogues)
            if len(capped_dialogues) != len(dialogues) or any(
                a.replique != b.replique for a, b in zip(capped_dialogues, dialogues)
            ):
                logger.warning(
                    "Scène %d : le LLM a dépassé le budget de %ds — répliques tronquées "
                    "(%ds -> %ds).",
                    scene_number, MAX_SCENE_DURATION_SEC,
                    estimate_scene_duration(dialogues), estimate_scene_duration(capped_dialogues),
                )

            scene = ScriptScene(
                scene=Scene(
                    number=scene_number,
                    type=str(scene_obj["type"]),
                    description=description,
                ),
                dialogues=capped_dialogues,
                transition=str(scene_data["transition"]),
                duration_seconds=estimate_scene_duration(capped_dialogues),
            )
            scenes.append(scene)

        estimated_duration = sum(s.duration_seconds for s in scenes)

        # Métadonnées enrichies — le provider/model réel sont passés en paramètre
        # (remplis dans _try_generate_llm après reconstruction)
        metadata: Dict[str, Any] = {
            "generator": "llm_v1",
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "angle": creative_brief.angle,
            "niche": opportunity.niche,
            "brand_id": brand_profile.id,
            "brand_name": brand_profile.name,
            "opportunity_score": opportunity.overall_score,
            "urgency": opportunity.urgency,
            "scene_count": len(scenes),
            "opportunity_id": opportunity.source_video_id,
            "llm_time_ms": response_time_ms,
            "llm_tokens": response_tokens,
            "llm_cost_usd": round(response_cost, 6),
        }

        script = Script(
            title=str(data["title"]),
            scenes=scenes,
            estimated_duration=estimated_duration,
            language=str(data.get("language", brand_profile.primary_language)),
            target_audience=str(data.get("target_audience", creative_brief.audience)),
            style=str(data.get("style", brand_profile.tone)),
            metadata=metadata,
        )

        return script
