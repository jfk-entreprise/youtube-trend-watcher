"""
Niche Intelligence Engine v1 — Moteur d'analyse sectorielle.

Architecture :
  - Niche         : entité thématique avec métriques agrégées.
  - NicheAnalyzer : regroupe les VideoTimeline par mot-clé et calcule
                    les indicateurs d'attractivité de chaque niche.

Comportement adaptatif selon la disponibilité des données historiques :
  - 1 snapshot par vidéo  : croissance = N/A, score basé sur vues + engagement.
  - ≥ 2 snapshots/vidéo   : avg_growth_speed activé, dominant dans le score (×2.0).

V1 : classification par mot-clé de collecte (champ `keyword`).
Évolution future : classification sémantique via LLM.

Extensibilité : surcharger _compute_niche_score() ou injecter des critères supplémentaires.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.virality_engine import VideoTimeline
from src.utils import fmt_views as _fmt_views, csv_snapshots_to_timelines

logger = logging.getLogger(__name__)


# ── Entité Niche ──────────────────────────────────────────────────────────────

@dataclass
class Niche:
    """
    Entité thématique avec métriques agrégées sur l'ensemble de ses vidéos.

    Champs calculés par NicheAnalyzer._compute_niche() ; ne pas instancier manuellement.
    """
    name: str
    volume: int                        # vidéos uniques dans la niche
    avg_views: float                   # vues moyennes (dernier snapshot)
    avg_engagement: float              # taux d'engagement moyen en % [(likes+comments)/views]
    avg_growth_speed: float            # vues/heure moyennes (0.0 si données temporelles absentes)
    niche_score: float                 # score synthétique d'attractivité
    timelines: list[VideoTimeline]     # accès aux VideoTimeline pour usage programmatique


# ── Moteur d'analyse ──────────────────────────────────────────────────────────

class NicheAnalyzer:
    """
    Charge les snapshots, regroupe les vidéos par niche et calcule les
    métriques d'attractivité de chaque secteur.

    Exemple minimal :
        analyzer = NicheAnalyzer(csv_path=Path("data/videos.csv"))
        report = analyzer.run()

    Usage programmatique :
        niches = analyzer.analyze()   # list[Niche] triée par score décroissant
    """

    def __init__(self, csv_path: Path, top_n: int = 10) -> None:
        self._csv_path = csv_path
        self._top_n = top_n

    # ── Interface publique ─────────────────────────────────────────────────────

    def run(self) -> str:
        """Charge → groupe → métriques → rapport texte."""
        timelines = self._load_timelines()
        if not timelines:
            return "Aucune donnée disponible dans le fichier CSV."

        niches = sorted(self._build_niches(timelines), key=lambda n: n.niche_score, reverse=True)
        logger.info("%d vidéos uniques → %d niches identifiées", len(timelines), len(niches))
        return self._generate_report(niches, len(timelines))

    def analyze(self) -> list[Niche]:
        """Retourne la liste des Niche triées par score décroissant (usage programmatique)."""
        timelines = self._load_timelines()
        if not timelines:
            return []
        return sorted(self._build_niches(timelines), key=lambda n: n.niche_score, reverse=True)

    # ── Chargement ─────────────────────────────────────────────────────────────

    def _load_timelines(self) -> list[VideoTimeline]:
        """Charge le CSV et regroupe les snapshots en VideoTimeline par video_id."""
        buckets = csv_snapshots_to_timelines(self._csv_path)
        return [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]

    # ── Regroupement & calcul des niches ──────────────────────────────────────

    def _build_niches(self, timelines: list[VideoTimeline]) -> list[Niche]:
        buckets: dict[str, list[VideoTimeline]] = {}
        for tl in timelines:
            buckets.setdefault(_niche_name(tl), []).append(tl)
        return [self._compute_niche(name, tls) for name, tls in buckets.items()]

    def _compute_niche(self, name: str, timelines: list[VideoTimeline]) -> Niche:
        volume = len(timelines)

        # Vues moyennes (dernier snapshot disponible)
        views = [tl.latest.view_count for tl in timelines if tl.latest.view_count is not None]
        avg_views = sum(views) / len(views) if views else 0.0

        # Engagement moyen en % — (likes + commentaires) / vues × 100
        eng_rates = []
        for tl in timelines:
            v = tl.latest
            if v.view_count and v.view_count > 0:
                interactions = (v.like_count or 0) + (v.comment_count or 0)
                eng_rates.append(interactions / v.view_count * 100)
        avg_engagement = sum(eng_rates) / len(eng_rates) if eng_rates else 0.0

        # Vitesse de croissance moyenne (vues/heure — actif uniquement si ≥ 2 snapshots/vidéo)
        growth_speeds = [
            tl.metrics.views_per_hour
            for tl in timelines
            if tl.metrics is not None
        ]
        avg_growth_speed = sum(growth_speeds) / len(growth_speeds) if growth_speeds else 0.0

        return Niche(
            name=name,
            volume=volume,
            avg_views=avg_views,
            avg_engagement=avg_engagement,
            avg_growth_speed=avg_growth_speed,
            niche_score=_compute_niche_score(avg_views, avg_engagement, avg_growth_speed, volume),
            timelines=timelines,
        )

    # ── Rapport ────────────────────────────────────────────────────────────────

    def _generate_report(self, niches: list[Niche], total_videos: int) -> str:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        SEP = "=" * 72
        THIN = "-" * 72

        n_with_growth = sum(1 for n in niches if n.avg_growth_speed > 0)

        lines = [
            SEP,
            "  NICHE INTELLIGENCE ENGINE v1 — RAPPORT D'ANALYSE SECTORIELLE",
            f"  Généré le            : {now_str}",
            f"  Vidéos totales       : {total_videos}",
            f"  Niches identifiées   : {len(niches)}",
            f"  Niches avec données de croissance : {n_with_growth}",
            SEP,
            "",
            "  CLASSEMENT DES NICHES (score décroissant)",
            THIN,
            f"  {'#':<4} {'Niche':<18} {'Vol':>5} {'Vues moy.':>10} {'Engagement':>11} {'Crois./h':>9} {'Score':>7}",
            f"  {'─'*4} {'─'*18} {'─'*5} {'─'*10} {'─'*11} {'─'*9} {'─'*7}",
        ]

        for rank, niche in enumerate(niches, 1):
            growth_str = f"{niche.avg_growth_speed:>7.0f}" if niche.avg_growth_speed > 0 else "    N/A"
            lines.append(
                f"  {rank:<4} {niche.name:<18} {niche.volume:>5} "
                f"{_fmt_views(int(niche.avg_views)):>10} "
                f"{niche.avg_engagement:>10.2f}% "
                f"{growth_str} "
                f"{niche.niche_score:>7.2f}"
            )

        lines += ["", SEP, "  DÉTAIL PAR NICHE", SEP]

        for rank, niche in enumerate(niches, 1):
            top_tls = sorted(
                niche.timelines,
                key=lambda tl: tl.latest.view_count or 0,
                reverse=True,
            )[: self._top_n]

            lines += [
                "",
                f"  #{rank}  ▸ Niche : {niche.name.upper()}  (score : {niche.niche_score:.2f})",
                THIN,
                f"  Volume        : {niche.volume} vidéo(s)",
                f"  Vues moyennes : {_fmt_views(int(niche.avg_views))}",
                f"  Engagement    : {niche.avg_engagement:.2f}%",
                f"  Croissance    : "
                + (f"{niche.avg_growth_speed:.0f} vues/heure (moy. observée)"
                   if niche.avg_growth_speed > 0
                   else "N/A — un seul snapshot par vidéo"),
                f"  Top {self._top_n} vidéos par vues :",
            ]
            for i, tl in enumerate(top_tls, 1):
                v = tl.latest
                snap_badge = f"[{len(tl.snapshots)} snap]" if len(tl.snapshots) > 1 else ""
                lines.append(
                    f"    {i:>2}. {_fmt_views(v.view_count):>8}  "
                    f"{v.title[:50]:<50}  "
                    f"{v.channel_title[:28]:<28}  {snap_badge}"
                )

        lines += ["", SEP, "  FIN DU RAPPORT", SEP]
        return "\n".join(lines)


# ── Utilitaires (privés au module) ────────────────────────────────────────────

def _niche_name(timeline: VideoTimeline) -> str:
    """Dérive le nom de niche depuis le mot-clé (V1). Vidéos trending → '(trending)'."""
    kw = timeline.latest.keyword.strip()
    return kw if kw else "(trending)"


def _compute_niche_score(
    avg_views: float,
    avg_engagement: float,
    avg_growth_speed: float,
    volume: int,
) -> float:
    """
    Score synthétique d'attractivité d'une niche, inspiré des poids DEFAULT_CRITERIA.

    - avg_views (×1.0) : demande de marché, normalisée par tranche de 1k vues.
    - avg_engagement (×0.5) : qualité de l'audience (signal organique).
    - avg_growth_speed (×2.0) : momentum — dominant si données temporelles disponibles.
    - volume (×0.3) : confiance statistique — plus de vidéos = signal plus robuste.
    """
    return (
        math.log1p(avg_views / 1_000) * 1.0
        + math.log1p(avg_engagement) * 0.5
        + math.log1p(avg_growth_speed) * 2.0
        + math.log1p(volume) * 0.3
    )
