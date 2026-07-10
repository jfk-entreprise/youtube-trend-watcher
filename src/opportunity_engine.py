"""
Opportunity Engine v1 — Moteur de détection d'opportunités de contenu.

Transforme des ContentProfile + KnowledgeBase en opportunités concrètes de création,
sans jamais lire directement les VideoSnapshot.

Composants :
  - Opportunity            : contrat officiel avec les moteurs créatifs.
  - OpportunityCriterion   : interface modulaire (direction maximize/minimize).
  - ViralityCriterion      : score ViralityEngine normalisé.
  - GrowthCriterion        : croissance temporelle mesurée (vues/heure).
  - EvergreenCriterion     : pérennité du contenu.
  - TrendCriterion         : dynamique de tendance.
  - CompetitionCriterion   : densité de la niche dans le corpus (heuristique V1).
  - DifficultyCriterion    : complexité estimée de production.
  - KnowledgeCriterion     : bonus combinaison validée par la KnowledgeBase.
  - OpportunityEngine      : orchestrateur.

Extensibilité :
  - LLM    : sous-classer OpportunityCriterion → score() via API.
  - Adaptatif : passer des weights personnalisés dans DEFAULT_CRITERIA.
  - Feedback : ajouter un FeedbackCriterion branché sur les retours utilisateurs.
"""

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from src.content_understanding import ContentProfile
from src.knowledge_engine import KnowledgeBase
from src.virality_engine import VideoTimeline
from src.virality_engine import DEFAULT_CRITERIA as VIRALITY_CRITERIA

logger = logging.getLogger(__name__)


# ── Opportunity ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Opportunity:
    """
    Opportunité concrète de création de contenu.

    Contrat officiel entre l'Opportunity Engine et les moteurs créatifs
    (Creative Engine, Script Engine, Learning Engine).

    Champs de score :
      - virality_score, growth_score, evergreen_score, trend_score : [0, 1] → plus haut = mieux
      - competition_score   : [0, 1] → plus bas = moins de concurrence = mieux
      - production_difficulty : [0, 1] → plus bas = plus facile à produire
      - overall_score       : [0, 1] → score composite (ranking principal)
      - urgency             : [0, 1] → temporalité de l'action recommandée
    """
    title: str
    niche: str
    source_video_id: str

    overall_score: float

    virality_score: float
    growth_score: float
    evergreen_score: float
    trend_score: float

    competition_score: float       # 0 = faible concurrence, 1 = niche saturée
    production_difficulty: float   # 0 = facile, 1 = très exigeant

    urgency: float

    recommendation: str
    rationale: List[str]
    metadata: dict[str, Any]


# ── OpportunityCriterion ──────────────────────────────────────────────────────

class OpportunityCriterion(ABC):
    """
    Interface modulaire pour les critères de scoring.

    direction = "maximize" (défaut) : score élevé → bonne opportunité.
    direction = "minimize"           : score élevé → mauvaise opportunité.

    L'orchestrateur applique automatiquement l'inversion pour les critères
    de type "minimize" avant de calculer le score global.

    Pour un critère LLM : sous-classer et implémenter score() avec un appel API.
    L'orchestrateur l'accepte sans modification.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def direction(self) -> str:
        return "maximize"

    @abstractmethod
    def score(
        self,
        profile: ContentProfile,
        timeline: Optional[VideoTimeline],
        kb: KnowledgeBase,
    ) -> float:
        """Retourne un score brut dans [0.0, 1.0]."""
        ...


# ── Critères ──────────────────────────────────────────────────────────────────

class ViralityCriterion(OpportunityCriterion):
    """
    Potentiel viral issu du ViralityEngine (scores multi-critères : vélocité,
    engagement, format, source, croissance temporelle, accélération).
    Normalisé sur le score brut p90 observé dans le corpus (~30).
    """

    _NORM = 40.0   # p99 observé ~38 — laisse de la marge pour différencier le top

    @property
    def name(self) -> str:
        return "virality"

    def score(self, profile, timeline, kb):
        if timeline is None:
            topic_fact = kb.topics.get(profile.primary_topic)
            return topic_fact.value.get("avg_trend_score", 0.5) if topic_fact else 0.5
        raw = sum(c.score(timeline) * c.weight for c in VIRALITY_CRITERIA)
        return round(min(raw / self._NORM, 1.0), 4)


class GrowthCriterion(OpportunityCriterion):
    """
    Croissance observée entre snapshots (vues/heure).
    Actif uniquement si ≥ 2 snapshots disponibles — retourne 0.0 sinon.
    Normalisé en log sur 50 000 vues/heure.
    """

    _LOG_REF = math.log1p(50_000)

    @property
    def name(self) -> str:
        return "growth"

    def score(self, profile, timeline, kb):
        if timeline is None or timeline.metrics is None:
            return 0.0
        return round(min(math.log1p(timeline.metrics.views_per_hour) / self._LOG_REF, 1.0), 4)


class EvergreenCriterion(OpportunityCriterion):
    """Pérennité du contenu (transmis directement depuis ContentProfile)."""

    @property
    def name(self) -> str:
        return "evergreen"

    def score(self, profile, timeline, kb):
        return profile.evergreen_score


class TrendCriterion(OpportunityCriterion):
    """Dynamique de tendance actuelle (transmis directement depuis ContentProfile)."""

    @property
    def name(self) -> str:
        return "trend"

    def score(self, profile, timeline, kb):
        return profile.trend_score


class CompetitionCriterion(OpportunityCriterion):
    """
    Niveau de concurrence estimé dans la niche (direction=minimize).

    V1 heuristique : densité du sujet dans le corpus × performance tendance.
    Moins la niche est représentée dans le corpus, plus l'opportunité est rare.

    V2 (prochains sprints) : LLM analyse les titres concurrents sur YouTube.
    """

    @property
    def name(self) -> str:
        return "competition"

    @property
    def direction(self) -> str:
        return "minimize"

    def score(self, profile, timeline, kb):
        fact = kb.topics.get(profile.primary_topic)
        if fact is None:
            return 0.5
        share = fact.observations / max(kb.total_profiles, 1)
        avg_trend = fact.value.get("avg_trend_score", 0.5)
        # Niche populaire ET très tendance = plus compétitive
        raw = share * 4.0 * (1.0 + avg_trend * 0.25)
        return round(min(raw, 1.0), 4)


class DifficultyCriterion(OpportunityCriterion):
    """
    Difficulté estimée de production (direction=minimize).

    Basée sur le format du contenu (ContentProfile.content_type).
    Un Short est simple à produire ; un Long ou un Clip Musical est exigeant.
    """

    _MAP: dict[str, float] = {
        "Short": 0.15,
        "Divertissement": 0.30,
        "Gameplay": 0.35,
        "Actualité": 0.40,
        "Standard": 0.45,
        "Tutorial": 0.55,
        "Analyse": 0.60,
        "Long": 0.75,
        "Clip Musical": 0.85,
    }

    @property
    def name(self) -> str:
        return "difficulty"

    @property
    def direction(self) -> str:
        return "minimize"

    def score(self, profile, timeline, kb):
        base = self._MAP.get(profile.content_type, 0.45)
        return round(base, 3)


class KnowledgeCriterion(OpportunityCriterion):
    """
    Bonus de connaissance marché : récompense les vidéos dont la combinaison
    (sujet, format) figure parmi les co-occurrences fréquentes de la KnowledgeBase.
    Une combinaison validée = format éprouvé dans cette niche.
    """

    @property
    def name(self) -> str:
        return "knowledge"

    def score(self, profile, timeline, kb):
        for combo in kb.top_combinations(25):
            keys = combo.value.get("keys", {})
            if (
                keys.get("primary_topic") == profile.primary_topic
                and keys.get("content_type") == profile.content_type
            ):
                freq = combo.value.get("frequency", 0)
                return round(min(freq / max(kb.total_profiles, 1) * 10, 1.0), 4)
        return 0.0


# ── Critères par défaut ────────────────────────────────────────────────────────
#
# Poids calqués sur l'esprit du ViralityEngine :
#   - La croissance temporelle domine (×2.0) quand disponible.
#   - La concurrence impacte significativement (×1.2, direction=minimize).
#   - La difficulté de production est un signal secondaire (×0.5).
#
DEFAULT_CRITERIA: List[OpportunityCriterion] = [
    ViralityCriterion(weight=1.5),
    GrowthCriterion(weight=2.0),
    EvergreenCriterion(weight=0.8),
    TrendCriterion(weight=1.0),
    CompetitionCriterion(weight=1.2),   # direction=minimize
    DifficultyCriterion(weight=0.5),    # direction=minimize
    KnowledgeCriterion(weight=0.8),
]


# ── OpportunityEngine ─────────────────────────────────────────────────────────

class OpportunityEngine:
    """
    Détecte et classe les meilleures opportunités de contenu.

    Exemple minimal :
        engine = OpportunityEngine()
        opps = engine.build(profiles, timelines, kb, top_n=10)

    Avec critères personnalisés (ex. LLM) :
        engine = OpportunityEngine(criteria=[
            LLMViralityCriterion(weight=2.0),
            GrowthCriterion(weight=2.0),
            ...
        ])

    Préparation Sprint 12 (Creative / Script Engine) :
        Le moteur retourne des Opportunity. Le Creative Engine consomme
        uniquement des Opportunity — jamais des ContentProfile ou VideoTimeline.
    """

    def __init__(self, criteria: Optional[List[OpportunityCriterion]] = None) -> None:
        self._criteria = criteria if criteria is not None else DEFAULT_CRITERIA

    # ── Interface publique ─────────────────────────────────────────────────────

    def build(
        self,
        profiles: List[ContentProfile],
        timelines: List[VideoTimeline],
        kb: KnowledgeBase,
        top_n: int = 10,
    ) -> List[Opportunity]:
        """
        Produit les top_n opportunités triées par overall_score décroissant.

        Args:
            profiles  : ContentProfile à évaluer (contrat Sprint 9).
            timelines : VideoTimeline pour les métriques temporelles (Sprint 5).
            kb        : KnowledgeBase — contexte marché (contrat Sprint 10).
            top_n     : nombre d'opportunités à retourner.
        """
        if not profiles:
            logger.warning("Aucun ContentProfile fourni.")
            return []

        timeline_map = {tl.video_id: tl for tl in timelines}
        opportunities = []

        for profile in profiles:
            tl = timeline_map.get(profile.video_id)
            try:
                opportunities.append(self._build_opportunity(profile, tl, kb))
            except Exception as exc:
                logger.debug("Échec sur %s : %s", profile.video_id, exc)

        ranked = sorted(opportunities, key=lambda o: o.overall_score, reverse=True)

        if ranked:
            logger.info(
                "%d opportunités évaluées — top : %.4f | moy : %.4f | avec données temp. : %d",
                len(ranked),
                ranked[0].overall_score,
                sum(o.overall_score for o in ranked) / len(ranked),
                sum(1 for o in ranked if o.growth_score > 0),
            )

        return ranked[:top_n]

    # ── Construction d'une opportunité ─────────────────────────────────────────

    def _build_opportunity(
        self,
        profile: ContentProfile,
        timeline: Optional[VideoTimeline],
        kb: KnowledgeBase,
    ) -> Opportunity:
        raw: dict[str, float] = {
            c.name: c.score(profile, timeline, kb) for c in self._criteria
        }

        # Score composite normalisé [0, 1]
        total_w = sum(c.weight for c in self._criteria)
        weighted = sum(
            (raw[c.name] if c.direction == "maximize" else 1.0 - raw[c.name]) * c.weight
            for c in self._criteria
        )
        base_score = weighted / total_w

        urgency = self._compute_urgency(profile, timeline)
        overall_score = round(base_score * (0.85 + urgency * 0.15), 4)

        title = timeline.latest.title if timeline else profile.video_id

        return Opportunity(
            title=title,
            niche=profile.primary_topic,
            source_video_id=profile.video_id,
            overall_score=overall_score,
            virality_score=round(raw.get("virality", 0.0), 4),
            growth_score=round(raw.get("growth", 0.0), 4),
            evergreen_score=profile.evergreen_score,
            trend_score=profile.trend_score,
            competition_score=round(raw.get("competition", 0.5), 4),
            production_difficulty=round(raw.get("difficulty", 0.45), 4),
            urgency=round(urgency, 4),
            recommendation=self._build_recommendation(raw, profile),
            rationale=self._build_rationale(raw, profile, timeline, kb),
            metadata={
                "raw_scores": {k: round(v, 4) for k, v in raw.items()},
                "base_score": round(base_score, 4),
                "content_type": profile.content_type,
                "language": profile.language,
                "target_audience": profile.target_audience,
                "emotion": profile.emotion,
            },
        )

    # ── Urgence ────────────────────────────────────────────────────────────────

    def _compute_urgency(
        self,
        profile: ContentProfile,
        timeline: Optional[VideoTimeline],
    ) -> float:
        urgency = profile.trend_score * 0.50

        if timeline and timeline.metrics:
            log_ref = math.log1p(50_000)
            velocity_signal = min(math.log1p(timeline.metrics.views_per_hour) / log_ref, 1.0)
            urgency += velocity_signal * 0.35

        if profile.metadata.get("trend", {}).get("source_bonus"):
            urgency += 0.15

        return round(min(urgency, 1.0), 4)

    # ── Recommandation textuelle ───────────────────────────────────────────────

    def _build_recommendation(
        self,
        raw: dict[str, float],
        profile: ContentProfile,
    ) -> str:
        parts = []
        trend = raw.get("trend", 0)
        growth = raw.get("growth", 0)
        competition = raw.get("competition", 0.5)
        difficulty = raw.get("difficulty", 0.5)
        evergreen = profile.evergreen_score

        # Timing
        if trend > 0.75 and growth > 0.45:
            parts.append("Produire aujourd'hui — tendance active et croissance mesurée.")
        elif trend > 0.65:
            parts.append("Produire rapidement — tendance en cours.")
        elif evergreen > 0.70:
            parts.append("Contenu pérenne — peut être produit sans urgence.")
        else:
            parts.append("Opportunité stable — pas de délai critique.")

        # Format
        if profile.content_type == "Long" and competition > 0.55:
            parts.append("Envisager une version Short pour réduire la barrière d'entrée.")
        elif profile.content_type == "Short" and evergreen > 0.65:
            parts.append("Un format Long apporterait plus de valeur durable sur ce sujet.")

        # Concurrence
        if competition < 0.20:
            parts.append("Niche peu exploitée — positionnement privilégié disponible.")
        elif competition > 0.65:
            parts.append("Niche saturée — différencier l'angle ou le format.")

        # Difficulté
        if difficulty < 0.25:
            parts.append("Production accessible — idéal pour tester rapidement.")
        elif difficulty > 0.70:
            parts.append("Production exigeante — prévoir les ressources adaptées.")

        return " ".join(parts)

    # ── Justification ──────────────────────────────────────────────────────────

    def _build_rationale(
        self,
        raw: dict[str, float],
        profile: ContentProfile,
        timeline: Optional[VideoTimeline],
        kb: KnowledgeBase,
    ) -> List[str]:
        rationale = []

        if raw.get("virality", 0) > 0.55:
            rationale.append(f"Potentiel viral élevé (score composite {raw['virality']:.2f})")

        if timeline and timeline.metrics:
            vph = timeline.metrics.views_per_hour
            if vph > 50:
                rationale.append(f"Croissance mesurée : {vph:,.0f} vues/heure entre snapshots")
        elif raw.get("growth", 0) == 0:
            rationale.append("Données temporelles insuffisantes — un seul snapshot disponible")

        if profile.evergreen_score > 0.65:
            rationale.append(
                f"Sujet pérenne — audience durable garantie (evergreen {profile.evergreen_score:.2f})"
            )

        if profile.trend_score > 0.75:
            rationale.append(f"En tendance active (trend {profile.trend_score:.2f})")

        topic_fact = kb.topics.get(profile.primary_topic)
        if topic_fact:
            n = topic_fact.observations
            pct = topic_fact.value.get("pct", 0)
            avg_t = topic_fact.value.get("avg_trend_score", 0)
            rationale.append(
                f"Niche '{profile.primary_topic}' : {n} vidéos ({pct:.1f}% du corpus)"
                f" — trend moyen {avg_t:.2f}"
            )

        if raw.get("knowledge", 0) > 0.25:
            rationale.append(
                f"Combinaison format/niche validée par la KnowledgeBase"
                f" (signal {raw['knowledge']:.2f})"
            )

        if raw.get("competition", 0.5) < 0.25:
            rationale.append("Faible concurrence dans cette niche — opportunité rare")

        if timeline and timeline.metrics and timeline.metrics.acceleration:
            acc = timeline.metrics.acceleration
            if acc > 0:
                rationale.append(f"Accélération de croissance positive ({acc:+.1f} vues/h²)")

        return rationale or ["Signal composite positif — scoring multi-critères."]
