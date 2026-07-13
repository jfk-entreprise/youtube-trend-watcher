"""
Rewrite Engine — Réécriture ciblée d'un Script à partir d'une évaluation LLM
(Sprint 22).

Objectif :
  Demander à un LLM de réécrire UNIQUEMENT les points faibles d'un Script
  (hook, rythme, storytelling, CTA, rétention), sans jamais toucher au
  sujet, à la marque, à la durée ou au nombre de scènes.

Pipeline :
    Script → Evaluation (LLMScriptEvaluator) → Rewrite → Nouvelle évaluation
        → si le score augmente : garder la nouvelle version
        → sinon : conserver l'ancienne

Ce que le Rewrite Engine PEUT changer (texte uniquement) :
  - les répliques (`replique`) de chaque dialogue de chaque scène — la
    première scène joue le rôle du hook, la dernière celui du CTA (Sprint
    31.1 : plus de champs hook/introduction/conclusion/call_to_action
    séparés, ce sont de simples scènes)

Ce que le Rewrite Engine NE CHANGE JAMAIS :
  - le sujet (title, `scene` — description visuelle —, `transition` de
    chaque scène restent identiques)
  - la marque (language, style, target_audience du Script)
  - la durée (estimated_duration du Script ET duration_seconds de chaque
    scène — jamais envoyés au LLM, toujours recopiés depuis l'original)
  - le nombre de scènes et leur ordre (order de chaque scène est fixé et
    vérifié après coup ; toute divergence invalide la réécriture)
  - le nombre de dialogues par scène et le personnage de chacun (seule la
    réplique est réécrite, jamais qui parle ni combien de fois)

Contrat :
  - Entrée : Script + LLMScriptScore (rapport de LLMScriptEvaluator).
  - Sortie  : Script (identique au contrat existant — la version réécrite
    si elle score mieux, sinon le Script original inchangé).
  - Dépend uniquement de Script (contrat), LLMScriptScore/LLMScriptEvaluator
    (pour ré-évaluer) et de LLMProvider (via build_llm()).
  - N'importe AUCUN moteur interne (ViralityEngine, KnowledgeEngine,
    ContentUnderstandingEngine, OpportunityEngine, CreativeEngine,
    BrandEngine, Collector, Storage, Agents, NicheAnalyzer).
"""

import json
import logging
import time
from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, List, Optional

from src.llm import LLMMessage, build_llm
from src.llm_script_evaluator import LLMScriptEvaluator, LLMScriptScore
from src.script_engine import Script, ScriptScene

logger = logging.getLogger(__name__)

# Nom complet de chaque langue supportée (même table que llm_script_generator._LANGUAGE_NAMES) —
# Sprint 34 : la langue des repliques réécrites suit script.language, jamais hardcodée.
_LANGUAGE_NAMES = {
    "fr": "French", "en": "English", "es": "Spanish", "pt": "Portuguese",
    "de": "German", "it": "Italian", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean",
}


def _language_name(code: str) -> str:
    return _LANGUAGE_NAMES.get(code, code)


# ── Prompt système ───────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_SYSTEM_PROMPT = (_PROMPTS_DIR / "rewrite_system_prompt.txt").read_text(encoding="utf-8")


class RewriteEngine:
    """
    Réécrit un Script pour corriger ses points faibles, sans jamais
    modifier le sujet, la marque, la durée ou le nombre de scènes.

    Utilisation :
        engine = RewriteEngine()
        evaluation = LLMScriptEvaluator().evaluate(script)
        improved = engine.rewrite(script, evaluation)
    """

    def __init__(
        self,
        evaluator: Optional[LLMScriptEvaluator] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_retries: int = 1,
    ) -> None:
        """
        Args:
            evaluator: Évaluateur utilisé pour noter la version réécrite
                (défaut : nouvelle instance de LLMScriptEvaluator).
            provider_name: Provider LLM pour la réécriture (None = auto-détection).
            model: Modèle spécifique (None = défaut du provider).
            temperature: Créativité de la réécriture.
            max_tokens: Tokens max en sortie.
            max_retries: Nombre de tentatives LLM avant d'abandonner la réécriture.
        """
        self._evaluator = evaluator or LLMScriptEvaluator()
        self._provider_name = provider_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        self._provider = None  # lazy init dans rewrite()
        self._stats: Dict[str, Any] = {
            "rewrite_attempts": 0,
            "rewrites_applied": 0,
            "rewrites_rejected": 0,
            "llm_failures": 0,
            "total_time_ms": 0,
            "total_cost_usd": 0.0,
        }

    @property
    def name(self) -> str:
        model_part = f"/{self._model}" if self._model else ""
        return f"rewrite_engine_{self._provider_name or 'auto'}{model_part}"

    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques cumulées du moteur (lecture seule)."""
        return dict(self._stats)

    # ── Interface publique ─────────────────────────────────────────────────────

    def rewrite(self, script: Script, evaluation: LLMScriptScore) -> Script:
        """
        Réécrit un Script à partir de son évaluation, puis garde la version
        qui obtient le meilleur score.

        Args:
            script: Script à améliorer.
            evaluation: Rapport de LLMScriptEvaluator sur `script`.

        Returns:
            Le Script réécrit si son score global augmente, sinon `script`
            inchangé. Ne lève jamais d'exception : tout échec (LLM, JSON,
            validation) retombe sur la version originale.
        """
        self._stats["rewrite_attempts"] += 1

        try:
            candidate = self._try_rewrite(script, evaluation)
        except Exception as exc:
            logger.warning("RewriteEngine — réécriture abandonnée : %s", exc)
            self._stats["llm_failures"] += 1
            self._stats["rewrites_rejected"] += 1
            return script

        try:
            new_score = self._evaluator.evaluate(candidate)
        except Exception as exc:
            logger.warning("RewriteEngine — ré-évaluation impossible, version conservée : %s", exc)
            self._stats["rewrites_rejected"] += 1
            return script

        if new_score.global_score > evaluation.global_score:
            logger.info(
                "RewriteEngine — amélioration retenue : %.1f → %.1f/80",
                evaluation.global_score, new_score.global_score,
            )
            self._stats["rewrites_applied"] += 1
            return replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "rewritten": True,
                    "rewrite_score_before": evaluation.global_score,
                    "rewrite_score_after": new_score.global_score,
                },
            )

        logger.info(
            "RewriteEngine — pas d'amélioration (%.1f → %.1f/80), version originale conservée",
            evaluation.global_score, new_score.global_score,
        )
        self._stats["rewrites_rejected"] += 1
        return script

    # ── Logique de réécriture ───────────────────────────────────────────────────

    def _try_rewrite(self, script: Script, evaluation: LLMScriptScore) -> Script:
        """Tente une réécriture via LLM. Lève une exception en cas d'échec."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._call_llm_and_build(script, evaluation)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "RewriteEngine — tentative %d/%d échouée : %s",
                    attempt, self._max_retries, exc,
                )
        raise RuntimeError(f"RewriteEngine : échec après {self._max_retries} tentative(s) : {last_exc}")

    def _call_llm_and_build(self, script: Script, evaluation: LLMScriptScore) -> Script:
        if self._provider is None:
            self._provider = build_llm(provider=self._provider_name, model=self._model)
            logger.info("RewriteEngine utilise %s / %s", self._provider.name, self._provider.model)

        user_prompt = self._build_user_prompt(script, evaluation)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

        start = time.time()
        response = self._provider.generate(
            messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_mode=True,
        )
        elapsed_ms = int((time.time() - start) * 1000)
        self._stats["total_time_ms"] += elapsed_ms
        self._stats["total_cost_usd"] += response.cost_usd

        if response.finish_reason == "error":
            raise RuntimeError(
                f"API {self._provider.name} error après {response.time_ms}ms: {response.content[:200]}"
            )

        json_str = self._extract_json(response.content.strip())
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON invalide : {exc}\nContenu reçu (début) : {response.content[:300]}")

        return self._build_script_from_json(data, script)

    # ── Construction du prompt ─────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(script: Script, evaluation: LLMScriptScore) -> str:
        language_name = _language_name(script.language)
        lines: List[str] = [
            "Here is the script to improve, and its critique.",
            "The first scene plays the role of the hook, the last one the CTA.",
            "",
            "=== CURRENT SCRIPT ===",
            f"Title: {script.title}",
            "Scenes:",
        ]
        for scene in script.scenes:
            dialogues_str = " / ".join(f"{d.personnage}: {d.replique}" for d in scene.dialogues)
            lines.append(f"  [order={scene.order}] ({scene.scene.type}) {scene.scene.description.setting}")
            lines.append(f"    Dialogues: {dialogues_str}")
        lines += [
            "",
            "=== CRITIQUE (LLMScriptEvaluator) ===",
            f"Current global score: {evaluation.global_score}/80",
            f"Weaknesses: {', '.join(evaluation.weaknesses) or '(none reported)'}",
            f"Suggestions: {', '.join(evaluation.suggestions) or '(none)'}",
            "",
            f"Rewrite ONLY the 'replique' of every dialogue in each of the "
            f"{len(script.scenes)} scenes above (keeping EXACTLY the same "
            f"\"order\" values: {[s.order for s in script.scenes]}, the same "
            f"number of dialogues per scene, and the same 'personnage' for "
            f"each dialogue — only the replique changes). Write repliques in {language_name}.",
            "Do not change the subject, the scene count, their order, or who speaks.",
            "Respond now with the rewrite JSON.",
        ]
        return "\n".join(lines)

    # ── Extraction et validation JSON ──────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extrait le JSON d'une réponse texte (gère les blocs ```json ... ```)."""
        text = text.strip()
        if text.startswith("```"):
            start = text.find("\n")
            end = text.rfind("```")
            if start != -1 and end != -1:
                text = text[start:end].strip()
            elif start != -1:
                text = text[start:].strip()
            else:
                text = text.replace("```json", "").replace("```", "").strip()

        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start:brace_end + 1]

        return text.strip()

    @staticmethod
    def _build_script_from_json(data: Dict[str, Any], original: Script) -> Script:
        """
        Reconstruit un Script à partir du JSON de réécriture, en préservant
        strictement le sujet, la marque, la durée et la structure de scènes
        du script original — seules les répliques (`replique`) changent
        (Sprint 31.1 : plus de hook/introduction/conclusion/call_to_action
        séparés, la première/dernière scène en jouent le rôle).

        Raises:
            ValueError: si un champ requis manque ou si le nombre/ordre des
                scènes, ou le nombre/personnage des dialogues, ne
                correspondent pas exactement à l'original.
        """
        if "scenes" not in data:
            raise ValueError("Champ obligatoire manquant dans la réécriture : 'scenes'")

        rewritten_scenes = data["scenes"]
        if not isinstance(rewritten_scenes, list):
            raise ValueError("Le champ 'scenes' doit être une liste")
        if len(rewritten_scenes) != len(original.scenes):
            raise ValueError(
                f"Nombre de scènes modifié : {len(rewritten_scenes)} reçu, "
                f"{len(original.scenes)} attendu — réécriture rejetée"
            )

        repliques_by_order: Dict[int, List[str]] = {}
        for entry in rewritten_scenes:
            if not isinstance(entry, dict) or "order" not in entry or "dialogues" not in entry:
                raise ValueError(f"Entrée de scène invalide dans la réécriture : {entry!r}")
            dialogues = entry["dialogues"]
            if not isinstance(dialogues, list):
                raise ValueError(f"'dialogues' doit être une liste pour la scène order={entry['order']}")
            repliques_by_order[int(entry["order"])] = [str(d.get("replique", "")) for d in dialogues]

        original_orders = {s.order for s in original.scenes}
        if set(repliques_by_order.keys()) != original_orders:
            raise ValueError(
                f"Ordre des scènes modifié : {sorted(repliques_by_order.keys())} reçu, "
                f"{sorted(original_orders)} attendu — réécriture rejetée"
            )

        new_scenes: List[ScriptScene] = []
        for scene in original.scenes:
            new_repliques = repliques_by_order[scene.order]
            if len(new_repliques) != len(scene.dialogues):
                raise ValueError(
                    f"Nombre de dialogues modifié pour la scène order={scene.order} : "
                    f"{len(new_repliques)} reçu, {len(scene.dialogues)} attendu — réécriture rejetée"
                )
            new_dialogues = [
                replace(dialogue, replique=new_replique)
                for dialogue, new_replique in zip(scene.dialogues, new_repliques)
            ]
            new_scenes.append(replace(scene, dialogues=new_dialogues))

        return replace(
            original,
            scenes=new_scenes,
            # Champs volontairement PRÉSERVÉS (jamais lus depuis `data`) :
            # title, estimated_duration, language, target_audience, style,
            # scene (description visuelle), transition, duration_seconds,
            # personnage de chaque dialogue — sujet, durée et identité de
            # marque restent inchangés.
        )
