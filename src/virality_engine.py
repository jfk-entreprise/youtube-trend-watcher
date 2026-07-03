"""
Virality Engine v2 — Moteur d'analyse + Time Engine.

Architecture à responsabilité unique :
  - Les agents collectent.
  - Ce moteur analyse (statique ET temporel).
  - L'IA expliquera.

Comportement adaptatif selon la disponibilité des données historiques :
  - 1 snapshot  : scoring statique (Sprint 4 — baseline)
  - ≥2 snapshots : scoring temporel prioritaire (vélocité de croissance)
  - ≥3 snapshots : +accélération (signal viral le plus fort)

Extensibilité : sous-classer ScoringCriterion, brancher sur la liste du constructeur.
"""

import csv
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.models import VideoSnapshot

logger = logging.getLogger(__name__)


# ── Métriques temporelles ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class TemporalMetrics:
    """Indicateurs d'évolution calculés à partir de l'historique des snapshots."""
    view_growth: int              # croissance absolue des vues (premier → dernier snapshot)
    like_growth: Optional[int]    # croissance absolue des likes (None si stats masquées)
    comment_growth: Optional[int] # croissance absolue des commentaires
    hours_elapsed: float          # durée de l'observation en heures
    views_per_hour: float         # vélocité de croissance (vues/heure)
    acceleration: Optional[float] # variation de la vélocité en (vues/h)/h — None si < 3 snapshots


# ── Timeline (regroupement chronologique) ──────────────────────────────────────

class VideoTimeline:
    """
    Regroupe tous les snapshots d'une même vidéo, triés chronologiquement.

    C'est l'unité d'analyse du moteur :
      - timeline.latest    → snapshot le plus récent (pour les métriques statiques)
      - timeline.metrics   → TemporalMetrics calculées une seule fois (None si 1 seul snapshot)
      - timeline.has_history → True si plusieurs snapshots disponibles
    """

    __slots__ = ("video_id", "snapshots", "_metrics_cache", "_metrics_computed")

    def __init__(self, video_id: str, snapshots: list[VideoSnapshot]) -> None:
        self.video_id = video_id
        self.snapshots: list[VideoSnapshot] = sorted(snapshots, key=lambda s: s.collected_at)
        self._metrics_cache: Optional[TemporalMetrics] = None
        self._metrics_computed: bool = False

    @property
    def latest(self) -> VideoSnapshot:
        return self.snapshots[-1]

    @property
    def oldest(self) -> VideoSnapshot:
        return self.snapshots[0]

    @property
    def has_history(self) -> bool:
        return len(self.snapshots) >= 2

    @property
    def metrics(self) -> Optional[TemporalMetrics]:
        """Calcul paresseux des métriques temporelles (résultat mis en cache)."""
        if not self._metrics_computed:
            self._metrics_cache = self._compute_metrics()
            self._metrics_computed = True
        return self._metrics_cache

    def _compute_metrics(self) -> Optional[TemporalMetrics]:
        if not self.has_history:
            return None

        first = self.snapshots[0]
        last = self.snapshots[-1]

        t0 = _parse_dt(first.collected_at)
        t1 = _parse_dt(last.collected_at)
        hours_total = max((t1 - t0).total_seconds() / 3600, 0.001)

        # Croissance des vues
        v0 = first.view_count or 0
        v1 = last.view_count or 0
        view_growth = max(v1 - v0, 0)
        views_per_hour = view_growth / hours_total

        # Croissance des likes
        like_growth: Optional[int] = None
        if first.like_count is not None and last.like_count is not None:
            like_growth = max(last.like_count - first.like_count, 0)

        # Croissance des commentaires
        comment_growth: Optional[int] = None
        if first.comment_count is not None and last.comment_count is not None:
            comment_growth = max(last.comment_count - first.comment_count, 0)

        # Accélération — nécessite ≥ 3 snapshots
        # Approche : comparer la vélocité de la 1ère moitié vs la 2ème moitié.
        acceleration: Optional[float] = None
        if len(self.snapshots) >= 3:
            mid = self.snapshots[len(self.snapshots) // 2]
            t_mid = _parse_dt(mid.collected_at)

            h_early = max((t_mid - t0).total_seconds() / 3600, 0.001)
            h_late = max((t1 - t_mid).total_seconds() / 3600, 0.001)

            v_mid = mid.view_count or 0
            vel_early = max(v_mid - v0, 0) / h_early
            vel_late = max(v1 - v_mid, 0) / h_late

            # (vues/heure) / heure — mesure la tendance d'accélération
            acceleration = (vel_late - vel_early) / hours_total

        return TemporalMetrics(
            view_growth=view_growth,
            like_growth=like_growth,
            comment_growth=comment_growth,
            hours_elapsed=hours_total,
            views_per_hour=views_per_hour,
            acceleration=acceleration,
        )


# ── Critères de scoring (modulaires) ──────────────────────────────────────────

class ScoringCriterion(ABC):
    """
    Interface commune à tout critère de scoring.

    Reçoit un VideoTimeline (qui contient le snapshot et l'historique).
    Retourne un score brut dans [0, +∞).
    Le poids est appliqué par le moteur lors de l'agrégation.

    Les critères statiques utilisent timeline.latest.
    Les critères temporels utilisent timeline.metrics (None si 1 seul snapshot → retourne 0).
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def description(self) -> str:
        return ""

    @abstractmethod
    def score(self, timeline: VideoTimeline) -> float: ...


# ── Critères statiques (toujours disponibles) ──────────────────────────────────

class VelocityCriterion(ScoringCriterion):
    """
    Vélocité statique = vues / âge_depuis_publication (log-normalisée).
    Baseline toujours disponible, même sans historique de collecte.
    """

    @property
    def name(self) -> str:
        return "velocity"

    @property
    def description(self) -> str:
        return "Vues/jour depuis publication (log)"

    def score(self, timeline: VideoTimeline) -> float:
        video = timeline.latest
        if not video.view_count or video.view_count <= 0:
            return 0.0
        age = max(_age_days(video), 0.5)
        return math.log1p(video.view_count / age)


class EngagementCriterion(ScoringCriterion):
    """Taux d'engagement = (likes + commentaires) / vues, log-normalisé."""

    @property
    def name(self) -> str:
        return "engagement"

    @property
    def description(self) -> str:
        return "(Likes + commentaires) / vues (log)"

    def score(self, timeline: VideoTimeline) -> float:
        video = timeline.latest
        if not video.view_count or video.view_count <= 0:
            return 0.0
        interactions = (video.like_count or 0) + (video.comment_count or 0)
        return math.log1p((interactions / video.view_count) * 100)


class FormatCriterion(ScoringCriterion):
    """Bonus pour le format Short (≤ 60s) — diffusion algorithmique amplifiée."""

    @property
    def name(self) -> str:
        return "format"

    @property
    def description(self) -> str:
        return "Bonus Short ≤ 60s"

    def score(self, timeline: VideoTimeline) -> float:
        return 1.0 if timeline.latest.duration_seconds <= 60 else 0.0


class SourceCriterion(ScoringCriterion):
    """Bonus pour les vidéos validées par le chart YouTube (source=trending)."""

    @property
    def name(self) -> str:
        return "source"

    @property
    def description(self) -> str:
        return "Bonus vidéo trending (chart=mostPopular)"

    def score(self, timeline: VideoTimeline) -> float:
        return 1.0 if timeline.latest.source == "trending" else 0.0


# ── Critères temporels (actifs uniquement si ≥ 2 snapshots) ───────────────────

class GrowthVelocityCriterion(ScoringCriterion):
    """
    Vélocité mesurée = vues gagnées / heures écoulées entre snapshots.
    Nettement plus précis que la vélocité statique car basé sur de vraies données
    de croissance observées, et non sur une estimation depuis la publication.
    Retourne 0 si un seul snapshot est disponible (aucun impact sur le score statique).
    """

    @property
    def name(self) -> str:
        return "growth_velocity"

    @property
    def description(self) -> str:
        return "Vues/heure mesurées entre snapshots (log) [temporel]"

    def score(self, timeline: VideoTimeline) -> float:
        m = timeline.metrics
        if m is None:
            return 0.0
        return math.log1p(m.views_per_hour)


class AccelerationCriterion(ScoringCriterion):
    """
    Accélération = variation de la vélocité dans le temps.
    Signal viral le plus fort : une vidéo qui accélère est en train de percer.
    Nécessite ≥ 3 snapshots. Retourne 0 en dessous de ce seuil.
    """

    @property
    def name(self) -> str:
        return "acceleration"

    @property
    def description(self) -> str:
        return "Accélération de croissance (Δvues/h)/h [≥ 3 snapshots]"

    def score(self, timeline: VideoTimeline) -> float:
        m = timeline.metrics
        if m is None or m.acceleration is None:
            return 0.0
        # On ne récompense que l'accélération positive
        return math.log1p(max(m.acceleration, 0.0))


# ── Critères par défaut ────────────────────────────────────────────────────────
#
# Composition conçue pour être adaptative :
#   - Sans historique : seuls les 4 critères statiques contribuent → comportement Sprint 4
#   - Avec historique  : les critères temporels (poids ×2.0 et ×1.5) prennent la main
#
DEFAULT_CRITERIA: list[ScoringCriterion] = [
    VelocityCriterion(weight=1.0),          # statique — toujours présent
    EngagementCriterion(weight=0.5),         # statique — qualité du contenu
    FormatCriterion(weight=0.3),             # statique — bonus format viral
    SourceCriterion(weight=0.2),             # statique — bonus validation algo
    GrowthVelocityCriterion(weight=2.0),     # temporel — signal dominant si historique
    AccelerationCriterion(weight=1.5),       # temporel — signal de percée virale
]


# ── Moteur principal ───────────────────────────────────────────────────────────

class ViralityEngine:
    """
    Analyse les données collectées et produit un classement par potentiel viral.

    Exemple minimal :
        engine = ViralityEngine(csv_path=Path("data/videos.csv"))
        report = engine.run()

    Exemple avec critères personnalisés :
        engine = ViralityEngine(
            csv_path=Path("data/videos.csv"),
            criteria=[GrowthVelocityCriterion(weight=3.0), FormatCriterion(weight=0.5)],
            top_n=10,
        )
    """

    def __init__(
        self,
        csv_path: Path,
        criteria: Optional[list[ScoringCriterion]] = None,
        top_n: int = 20,
    ) -> None:
        self._csv_path = csv_path
        self._criteria = criteria if criteria is not None else DEFAULT_CRITERIA
        self._top_n = top_n

    # ── Interface publique ─────────────────────────────────────────────────────

    def run(self) -> str:
        """Charge → groupe → score → tri → rapport."""
        timelines = self._load_timelines()
        if not timelines:
            return "Aucune donnée disponible dans le fichier CSV."

        scored = sorted(
            ((tl, self._compute_score(tl)) for tl in timelines),
            key=lambda pair: pair[1],
            reverse=True,
        )

        n_with_history = sum(1 for tl in timelines if tl.has_history)
        n_with_accel = sum(
            1 for tl in timelines if tl.metrics and tl.metrics.acceleration is not None
        )
        logger.info(
            "%d vidéos uniques | %d avec historique | %d avec accélération",
            len(timelines), n_with_history, n_with_accel,
        )
        logger.info(
            "Top score : %.2f | Score moyen : %.2f",
            scored[0][1],
            sum(s for _, s in scored) / len(scored),
        )
        return self._generate_report(scored, n_with_history, n_with_accel)

    # ── Chargement & regroupement ──────────────────────────────────────────────

    def _load_timelines(self) -> list[VideoTimeline]:
        """Charge le CSV, reconstruit les snapshots et les groupe par video_id."""
        if not self._csv_path.exists():
            logger.error("Fichier CSV introuvable : %s", self._csv_path)
            return []

        buckets: dict[str, list[VideoSnapshot]] = {}
        skipped = 0

        with open(self._csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    snap = VideoSnapshot(
                        video_id=row["video_id"],
                        title=row["title"],
                        channel_id=row["channel_id"],
                        channel_title=row["channel_title"],
                        published_at=row["published_at"],
                        description=row.get("description", ""),
                        duration_iso=row["duration_iso"],
                        duration_seconds=int(row["duration_seconds"] or 0),
                        view_count=_to_int(row.get("view_count")),
                        like_count=_to_int(row.get("like_count")),
                        comment_count=_to_int(row.get("comment_count")),
                        keyword=row["keyword"],
                        source=row.get("source", "keyword"),
                        collected_at=row["collected_at"],
                    )
                    buckets.setdefault(snap.video_id, []).append(snap)
                except Exception as exc:
                    skipped += 1
                    logger.debug("Ligne ignorée (%s)", exc)

        if skipped:
            logger.warning("%d ligne(s) ignorée(s) lors du chargement.", skipped)

        timelines = [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]
        logger.info(
            "%d snapshots → %d vidéos uniques (depuis %s)",
            sum(len(tl.snapshots) for tl in timelines),
            len(timelines),
            self._csv_path.name,
        )
        return timelines

    # ── Score composite ────────────────────────────────────────────────────────

    def _compute_score(self, timeline: VideoTimeline) -> float:
        return sum(c.score(timeline) * c.weight for c in self._criteria)

    def _score_breakdown(self, timeline: VideoTimeline) -> dict[str, float]:
        return {c.name: round(c.score(timeline) * c.weight, 3) for c in self._criteria}

    # ── Rapport ────────────────────────────────────────────────────────────────

    def _generate_report(
        self,
        scored: list[tuple[VideoTimeline, float]],
        n_with_history: int,
        n_with_accel: int,
    ) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        SEP = "=" * 72
        THIN = "-" * 72

        total = len(scored)
        top = scored[: self._top_n]
        all_scores = [s for _, s in scored]
        avg_score = sum(all_scores) / total

        sources: dict[str, int] = {}
        for tl, _ in scored:
            src = tl.latest.source
            sources[src] = sources.get(src, 0) + 1

        # ── En-tête ──
        lines = [
            SEP,
            "  VIRALITY ENGINE v2 — RAPPORT D'ANALYSE (Time Engine)",
            f"  Généré le         : {now_str}",
            f"  Vidéos uniques    : {total}",
            f"  Avec historique   : {n_with_history} vidéos (≥ 2 snapshots)",
            f"  Avec accélération : {n_with_accel} vidéos (≥ 3 snapshots)",
            f"  Score moyen       : {avg_score:.2f}  |  Max : {all_scores[0]:.2f}",
            f"  Top N affiché     : {self._top_n}",
            SEP,
            "",
            "  CRITÈRES & POIDS DU SCORE COMPOSITE",
            THIN,
            f"  {'Critère':<25} {'Poids':>6}  Description",
            f"  {'─'*25} {'─'*6}  {'─'*35}",
            *[f"  {c.name:<25} {'×'+str(c.weight):>6}  {c.description}" for c in self._criteria],
            "",
            "  RÉPARTITION DES SOURCES",
            THIN,
        ]
        for src, count in sorted(sources.items()):
            lines.append(f"  {src:<12} {count:>5} vidéos ({count/total*100:.0f}%)")

        # ── Contexte temporel ──
        lines += ["", "  ANALYSE TEMPORELLE", THIN]
        if n_with_history == 0:
            lines += [
                "  Aucun historique disponible — un seul snapshot par vidéo.",
                "  Le moteur opère en mode statique (Sprint 4).",
                "  Relancez test_agents.py pour accumuler des snapshots supplémentaires.",
                "  Les critères temporels (growth_velocity, acceleration) sont prêts",
                "  et s'activeront automatiquement dès le prochain cycle de collecte.",
            ]
        else:
            lines += [
                f"  {n_with_history}/{total} vidéos disposent d'un historique de snapshots.",
                f"  {n_with_accel}/{total} disposent de données d'accélération (≥ 3 snapshots).",
                "",
                "  TOP CROISSEURS — vidéos par vélocité mesurée (vues/heure)",
                THIN,
            ]
            growth_leaders = sorted(
                [(tl, tl.metrics) for tl in [t for t, _ in scored] if tl.metrics],
                key=lambda x: x[1].views_per_hour,  # type: ignore[union-attr]
                reverse=True,
            )[:10]
            for i, (tl, m) in enumerate(growth_leaders, 1):
                accel_str = (
                    f"  accel: {m.acceleration:+.1f} vues/h²"  # type: ignore[union-attr]
                    if m.acceleration is not None  # type: ignore[union-attr]
                    else ""
                )
                lines.append(
                    f"  {i:2}. {_fmt_views(int(m.views_per_hour)):>8}/h  "  # type: ignore[arg-type]
                    f"+{_fmt_views(m.view_growth)} vues  "
                    f"{m.hours_elapsed:.1f}h  "
                    f"{accel_str}  "
                    f"{tl.latest.title[:35]}"
                )

        # ── Top N ──
        lines += [
            "",
            SEP,
            f"  TOP {self._top_n} — VIDÉOS AU PLUS FORT POTENTIEL VIRAL",
            SEP,
            "",
        ]

        for rank, (timeline, score) in enumerate(top, 1):
            video = timeline.latest
            m = timeline.metrics
            age = _age_days(video)
            is_short = video.duration_seconds <= 60
            breakdown = self._score_breakdown(timeline)
            breakdown_str = "  ".join(f"{k}:{v:.2f}" for k, v in breakdown.items())

            temporal_line = ""
            if m is not None:
                accel_part = (
                    f"  accél: {m.acceleration:+.1f}/h²" if m.acceleration is not None else ""
                )
                temporal_line = (
                    f"       Croissance : +{_fmt_views(m.view_growth)} vues"
                    f"  en {m.hours_elapsed:.1f}h"
                    f"  ({_fmt_views(int(m.views_per_hour))}/h)"
                    f"{accel_part}"
                )

            snapshot_badge = f"[{len(timeline.snapshots)} snap]"

            lines += [
                f"  #{rank:02d}  ▸ Score : {score:.2f}  {snapshot_badge}",
                f"       {video.title[:66]}",
                f"       Chaîne    : {video.channel_title[:50]}",
                f"       Vues      : {_fmt_views(video.view_count):<10}"
                f"  Durée : {_fmt_dur(video.duration_seconds):<9}"
                + ("  [SHORT]" if is_short else ""),
                f"       Âge pub.  : {age:.1f}j"
                f"  Source : {video.source:<10}"
                f"  Mot-clé : {video.keyword}",
            ]
            if temporal_line:
                lines.append(temporal_line)
            lines += [
                f"       Détail    : {breakdown_str}",
                f"       ↳ https://youtu.be/{video.video_id}",
                "",
            ]

        lines += [SEP, "  FIN DU RAPPORT", SEP]
        return "\n".join(lines)


# ── Utilitaires (privés au module) ────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _age_days(video: VideoSnapshot) -> float:
    try:
        return max((datetime.now(timezone.utc) - _parse_dt(video.published_at)).total_seconds() / 86400, 0.0)
    except Exception:
        return 30.0


def _to_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def _fmt_dur(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_views(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)
