"""
LLM Script Evaluator — Notation d'un Script par un LLM-juge (Sprint 21).

Objectif :
  Faire évaluer un Script par un LLM (via build_llm()) sur 8 critères
  qualitatifs, en complément (ou en comparaison) du ScriptEvaluator
  heuristique existant.

Architecture :
  1. Prompt système + utilisateur envoyé au LLM (via build_llm()).
  2. Le LLM répond STRICTEMENT en JSON (json_mode=True).
  3. Validation du JSON → reconstruction de LLMScriptScore.
  4. Si le LLM échoue (JSON invalide, erreur API, timeout) :
     - retry automatique (max_retries tentatives).
     - au-delà → RuntimeError (pas de fallback heuristique : ce module
       EST la variante LLM, à comparer explicitement au ScriptEvaluator
       heuristique par l'appelant).

Contrat :
  - Entrée : Script (contrat existant, inchangé).
  - Sortie  : LLMScriptScore (8 critères /10 + global_score /80 +
    forces/faiblesses/suggestions qualitatives).
  - Hérite de BaseEvaluator (src/script_evaluator.py) — polymorphe avec
    ScriptEvaluator (heuristique) via `.global_score` et `.name`.
  - Dépend uniquement de Script (contrat) et de LLMProvider (via build_llm()).
  - N'importe AUCUN moteur interne (ViralityEngine, KnowledgeEngine,
    ContentUnderstandingEngine, OpportunityEngine, CreativeEngine,
    BrandEngine, Collector, Storage, Agents, NicheAnalyzer).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm import LLMMessage, build_llm
from src.script_engine import Script
from src.script_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


# ── Critères ─────────────────────────────────────────────────────────────────

_CRITERIA = ("hook", "curiosity", "storytelling", "rhythm", "clarity", "cta", "retention", "viral")


# ── Prompt système ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """Tu es un juge expert en scripts YouTube (format Shorts). Tu reponds UNIQUEMENT en JSON valide, sans texte avant ni apres le bloc JSON.

Tu dois evaluer objectivement un script video sur 8 criteres, chacun note de 0 a 10 (entiers ou decimales a 1 chiffre) :

1. hook         : la premiere phrase capte-t-elle l'attention en moins de 3 secondes ? Ouvre-t-elle une boucle de curiosite ?
2. curiosity    : le script donne-t-il envie de savoir la suite tout du long ?
3. storytelling : progression narrative, tension, payoff, montee progressive
4. rhythm       : rythme de la narration, absence de remplissage, une idee par scene
5. clarity      : le message est-il facile a comprendre, sans jargon inutile ?
6. cta          : l'appel a l'action est-il specifique au sujet (jamais generique type 'Abonne-toi pour plus') ?
7. retention    : potentiel de retention du spectateur jusqu'a la fin
8. viral        : potentiel de partage / viralite du sujet et de l'angle choisi

Sois strict et exigeant : un script moyen doit obtenir des notes autour de 5-6, pas 8-9.

FORMAT DE REPONSE (respecte STRICTEMENT cette structure JSON) :

```json
{
  "hook": 8,
  "curiosity": 9,
  "storytelling": 8,
  "rhythm": 7,
  "clarity": 9,
  "cta": 8,
  "retention": 9,
  "viral": 8,
  "global_score": 66,
  "strengths": ["point fort 1", "point fort 2"],
  "weaknesses": ["point faible 1", "point faible 2"],
  "suggestions": ["suggestion concrete 1", "suggestion concrete 2"]
}
```

REGLES :
1. Le JSON doit etre VALIDE et directement utilisable par json.loads()
2. Les 8 criteres sont des nombres entre 0 et 10
3. global_score = somme des 8 criteres (sur 80)
4. strengths, weaknesses, suggestions sont des listes de chaines courtes et concretes (2-5 elements chacune)
5. weaknesses et suggestions doivent porter en priorite sur : hook, rythme, storytelling, CTA, retention
6. NE RIEN ECRIRE en dehors du bloc JSON
7. Si json_mode est actif, reponds UNIQUEMENT avec l'objet JSON brut"""


# ── LLMScriptScore ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMScriptScore:
    """Score d'un script par un LLM-juge sur 8 critères + score global."""
    hook: float             # /10
    curiosity: float        # /10
    storytelling: float     # /10
    rhythm: float           # /10
    clarity: float          # /10
    cta: float              # /10
    retention: float        # /10
    viral: float            # /10
    global_score: float     # /80 (somme des 8)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── LLMScriptEvaluator ───────────────────────────────────────────────────────

class LLMScriptEvaluator(BaseEvaluator):
    """
    Évaluateur de scripts piloté par LLM (LLM-as-judge).

    Utilise exclusivement le LLM Provider (via build_llm()) pour noter un
    Script sur 8 critères qualitatifs et produire des points forts/faibles/
    suggestions exploitables (notamment par RewriteEngine, Sprint 22).

    Utilisation :
        evaluator = LLMScriptEvaluator()
        score = evaluator.evaluate(script)
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        max_retries: int = 2,
    ) -> None:
        """
        Args:
            provider_name: Provider LLM à utiliser (None = auto-détection).
            model: Modèle spécifique (None = défaut du provider).
            temperature: Faible par défaut — on veut un jugement consistant.
            max_tokens: Tokens max en sortie (score + justifications courtes).
            max_retries: Nombre de tentatives LLM avant échec définitif.
        """
        self._provider_name = provider_name
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        self._provider = None  # lazy init dans evaluate()
        self._stats: Dict[str, Any] = {
            "llm_calls": 0,
            "llm_success": 0,
            "llm_failures": 0,
            "total_time_ms": 0,
            "total_cost_usd": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
        }

    @property
    def name(self) -> str:
        model_part = f"/{self._model}" if self._model else ""
        return f"llm_judge_{self._provider_name or 'auto'}{model_part}"

    @property
    def stats(self) -> Dict[str, Any]:
        """Statistiques cumulées de l'évaluateur (lecture seule)."""
        return dict(self._stats)

    # ── Interface BaseEvaluator ────────────────────────────────────────────────

    def evaluate(self, script: Script) -> LLMScriptScore:
        """
        Évalue un Script via LLM.

        Args:
            script: Script à évaluer.

        Returns:
            LLMScriptScore.

        Raises:
            RuntimeError: si toutes les tentatives LLM échouent.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._try_evaluate(script, attempt=attempt)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "LLM Script Evaluator — tentative %d/%d échouée : %s",
                    attempt, self._max_retries, exc,
                )
                self._stats["llm_failures"] += 1

        raise RuntimeError(
            f"LLMScriptEvaluator : échec après {self._max_retries} tentative(s) : {last_exc}"
        )

    # ── Logique LLM ────────────────────────────────────────────────────────────

    def _try_evaluate(self, script: Script, attempt: int = 1) -> LLMScriptScore:
        """Tente une évaluation via LLM. Lève une exception en cas d'échec."""
        if self._provider is None:
            self._provider = build_llm(
                provider=self._provider_name,
                model=self._model,
            )
            logger.info(
                "LLMScriptEvaluator utilise %s / %s",
                self._provider.name, self._provider.model,
            )

        user_prompt = self._build_user_prompt(script)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_prompt),
        ]

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
        self._stats["total_prompt_tokens"] += response.prompt_tokens
        self._stats["total_completion_tokens"] += response.completion_tokens

        if response.finish_reason == "error":
            raise RuntimeError(
                f"API {self._provider.name} error après {response.time_ms}ms: "
                f"{response.content[:200]}"
            )

        json_str = self._extract_json(response.content.strip())
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JSON invalide après tentative {attempt} : {exc}\n"
                f"Contenu reçu (début) : {response.content[:300]}"
            )

        score = self._build_score_from_json(
            data,
            llm_provider=response.provider_name,
            llm_model=response.model,
            response_time_ms=elapsed_ms,
            response_tokens=response.total_tokens,
            response_cost=response.cost_usd,
        )

        self._stats["llm_success"] += 1
        logger.info(
            "LLM Script Evaluator OK — global_score=%.1f/80 (%d tokens, $%.6f, %d ms, provider=%s, model=%s)",
            score.global_score, response.total_tokens, response.cost_usd,
            elapsed_ms, response.provider_name, response.model,
        )
        return score

    # ── Construction du prompt ─────────────────────────────────────────────────

    @staticmethod
    def _build_user_prompt(script: Script) -> str:
        """Construit le prompt utilisateur à partir du Script à évaluer."""
        lines: List[str] = [
            "Evalue le script YouTube suivant selon les 8 criteres definis dans tes instructions.",
            "",
            f"=== TITRE === \n{script.title}",
            "",
            f"=== HOOK ({script.scenes[0].duration_seconds if script.scenes else '?'}s premiere scene) === \n{script.hook}",
            "",
            f"=== INTRODUCTION === \n{script.introduction}",
            "",
            "=== SCENES ===",
        ]
        for scene in script.scenes:
            lines.append(f"  [{scene.order}] ({scene.duration_seconds}s) {scene.title} : {scene.narration}")
        lines.extend([
            "",
            f"=== CONCLUSION === \n{script.conclusion}",
            "",
            f"=== CTA === \n{script.call_to_action}",
            "",
            f"=== METADONNEES === Duree totale: {script.estimated_duration}s | Style: {script.style} | Langue: {script.language}",
            "",
            "Reponds maintenant avec le JSON d'evaluation.",
        ])
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
    def _validate_json_structure(data: Dict[str, Any]) -> None:
        """Valide la structure du JSON de réponse."""
        for criterion in _CRITERIA:
            if criterion not in data:
                raise ValueError(f"Critère obligatoire manquant dans la réponse : '{criterion}'")
            value = data[criterion]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Le critère '{criterion}' doit être un nombre, reçu : {value!r}")
            if not (0 <= float(value) <= 10):
                raise ValueError(f"Le critère '{criterion}' doit être entre 0 et 10, reçu : {value}")

        for list_field in ("strengths", "weaknesses", "suggestions"):
            if list_field not in data:
                raise ValueError(f"Champ obligatoire manquant dans la réponse : '{list_field}'")
            if not isinstance(data[list_field], list):
                raise ValueError(f"Le champ '{list_field}' doit être une liste")

    @classmethod
    def _build_score_from_json(
        cls,
        data: Dict[str, Any],
        llm_provider: str = "",
        llm_model: str = "",
        response_time_ms: int = 0,
        response_tokens: int = 0,
        response_cost: float = 0.0,
    ) -> LLMScriptScore:
        """Reconstruit un LLMScriptScore à partir du JSON validé."""
        cls._validate_json_structure(data)

        values = {c: float(data[c]) for c in _CRITERIA}
        # global_score toujours recalculé côté serveur (ne pas faire confiance
        # à l'arithmétique du LLM) — garde-fou contre une somme incohérente.
        computed_global = round(sum(values.values()), 1)
        llm_global = data.get("global_score")
        if isinstance(llm_global, (int, float)) and abs(float(llm_global) - computed_global) > 1.0:
            logger.debug(
                "global_score LLM (%s) diffère de la somme recalculée (%s) — la somme fait foi.",
                llm_global, computed_global,
            )

        return LLMScriptScore(
            hook=values["hook"],
            curiosity=values["curiosity"],
            storytelling=values["storytelling"],
            rhythm=values["rhythm"],
            clarity=values["clarity"],
            cta=values["cta"],
            retention=values["retention"],
            viral=values["viral"],
            global_score=computed_global,
            strengths=[str(s) for s in data.get("strengths", [])],
            weaknesses=[str(s) for s in data.get("weaknesses", [])],
            suggestions=[str(s) for s in data.get("suggestions", [])],
            metadata={
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "llm_time_ms": response_time_ms,
                "llm_tokens": response_tokens,
                "llm_cost_usd": round(response_cost, 6),
            },
        )
