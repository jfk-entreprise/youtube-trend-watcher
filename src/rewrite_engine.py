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
  - hook, introduction, conclusion, call_to_action
  - narration de chaque scène (le texte parlé)

Ce que le Rewrite Engine NE CHANGE JAMAIS :
  - le sujet (title, visual_description, image_prompt, animation_notes,
    sound_effects de chaque scène restent identiques)
  - la marque (language, style, target_audience du Script)
  - la durée (estimated_duration du Script ET duration_seconds de chaque
    scène — jamais envoyés au LLM, toujours recopiés depuis l'original)
  - le nombre de scènes et leur ordre (order de chaque scène est fixé et
    vérifié après coup ; toute divergence invalide la réécriture)

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
        lines: List[str] = [
            "Voici le script a ameliorer, et sa critique.",
            "",
            "=== SCRIPT ACTUEL ===",
            f"Titre : {script.title}",
            f"Hook : {script.hook}",
            f"Introduction : {script.introduction}",
            "Scenes :",
        ]
        for scene in script.scenes:
            lines.append(f"  [order={scene.order}] {scene.title} : {scene.narration}")
        lines += [
            f"Conclusion : {script.conclusion}",
            f"CTA : {script.call_to_action}",
            "",
            "=== CRITIQUE (LLMScriptEvaluator) ===",
            f"Score global actuel : {evaluation.global_score}/80",
            f"Points faibles : {', '.join(evaluation.weaknesses) or '(aucun signale)'}",
            f"Suggestions : {', '.join(evaluation.suggestions) or '(aucune)'}",
            "",
            f"Reecris uniquement hook, introduction, conclusion, call_to_action, "
            f"et la narration de chacune des {len(script.scenes)} scenes ci-dessus "
            f"(en conservant EXACTEMENT les memes valeurs de \"order\" : "
            f"{[s.order for s in script.scenes]}).",
            "Ne change ni le sujet, ni le nombre de scenes, ni leur ordre.",
            "Reponds maintenant avec le JSON de reecriture.",
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
        du script original.

        Raises:
            ValueError: si un champ requis manque ou si le nombre/ordre des
                scènes ne correspond pas exactement à l'original.
        """
        for required in ("hook", "introduction", "scenes", "conclusion", "call_to_action"):
            if required not in data:
                raise ValueError(f"Champ obligatoire manquant dans la réécriture : '{required}'")

        rewritten_scenes = data["scenes"]
        if not isinstance(rewritten_scenes, list):
            raise ValueError("Le champ 'scenes' doit être une liste")
        if len(rewritten_scenes) != len(original.scenes):
            raise ValueError(
                f"Nombre de scènes modifié : {len(rewritten_scenes)} reçu, "
                f"{len(original.scenes)} attendu — réécriture rejetée"
            )

        narration_by_order: Dict[int, str] = {}
        for entry in rewritten_scenes:
            if not isinstance(entry, dict) or "order" not in entry or "narration" not in entry:
                raise ValueError(f"Entrée de scène invalide dans la réécriture : {entry!r}")
            narration_by_order[int(entry["order"])] = str(entry["narration"])

        original_orders = {s.order for s in original.scenes}
        if set(narration_by_order.keys()) != original_orders:
            raise ValueError(
                f"Ordre des scènes modifié : {sorted(narration_by_order.keys())} reçu, "
                f"{sorted(original_orders)} attendu — réécriture rejetée"
            )

        new_scenes: List[ScriptScene] = [
            replace(scene, narration=narration_by_order[scene.order])
            for scene in original.scenes
        ]

        return replace(
            original,
            hook=str(data["hook"]),
            introduction=str(data["introduction"]),
            scenes=new_scenes,
            conclusion=str(data["conclusion"]),
            call_to_action=str(data["call_to_action"]),
            # Champs volontairement PRÉSERVÉS (jamais lus depuis `data`) :
            # title, estimated_duration, language, target_audience, style —
            # sujet, durée et identité de marque restent inchangés.
        )
