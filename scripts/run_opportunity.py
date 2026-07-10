"""
Opportunity Engine — Script de lancement.
Usage : python scripts/run_opportunity.py

Pipeline complet :
  1. Snapshots via build_storage()
  2. ContentProfile via ContentUnderstandingEngine (Sprint 9)
  3. KnowledgeBase via KnowledgeEngine (Sprint 10)
  4. Opportunités via OpportunityEngine (Sprint 11)
  5. Rapport texte + sauvegarde
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.models import VideoSnapshot
from src.storage import build_storage
from src.virality_engine import VideoTimeline
from src.content_understanding import ContentUnderstandingEngine
from src.knowledge_engine import KnowledgeEngine
from src.opportunity_engine import Opportunity, OpportunityEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"
TOP_N = 10


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _build_timelines(snapshots: list[VideoSnapshot]) -> list[VideoTimeline]:
    buckets: dict[str, list[VideoSnapshot]] = {}
    for snap in snapshots:
        buckets.setdefault(snap.video_id, []).append(snap)
    return [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]


# ── Formatage du rapport ──────────────────────────────────────────────────────

def _score_bar(score: float, width: int = 20) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _label_competition(s: float) -> str:
    if s < 0.20:
        return f"{s:.2f}  ✓ Faible"
    if s < 0.55:
        return f"{s:.2f}  ~ Modérée"
    return f"{s:.2f}  ✗ Élevée"


def _label_difficulty(s: float) -> str:
    if s < 0.30:
        return f"{s:.2f}  ✓ Facile"
    if s < 0.60:
        return f"{s:.2f}  ~ Modérée"
    return f"{s:.2f}  ✗ Exigeante"


def _label_urgency(u: float) -> str:
    if u > 0.75:
        return f"{u:.2f}  🔴 Produire maintenant"
    if u > 0.45:
        return f"{u:.2f}  🟡 Cette semaine"
    return f"{u:.2f}  🟢 Pas d'urgence"


def build_report(opportunities: list[Opportunity]) -> str:
    SEP = "=" * 72
    THIN = "-" * 72
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        SEP,
        "  OPPORTUNITY ENGINE v1 — DÉTECTION D'OPPORTUNITÉS DE CONTENU",
        f"  Généré le              : {now_str}",
        f"  Opportunités détectées : {len(opportunities)}",
        SEP,
    ]

    for rank, opp in enumerate(opportunities, 1):
        score_bar = _score_bar(opp.overall_score)
        lines += [
            "",
            f"  #{rank:02d} ─── Score global : {opp.overall_score:.4f}  [{score_bar}]",
            THIN,
            f"  Titre    : {opp.title[:68]}",
            f"  Niche    : {opp.niche}",
            f"  Format   : {opp.metadata.get('content_type', '?')}",
            f"  Audience : {opp.metadata.get('target_audience', '?')}",
            f"  Langue   : {opp.metadata.get('language', '?')}",
            f"  Émotion  : {opp.metadata.get('emotion', '?')}",
            "",
            "  Scores détaillés :",
            f"  ├─ Viralité           {opp.virality_score:>6.3f}  {_score_bar(opp.virality_score, 15)}",
            f"  ├─ Croissance         {opp.growth_score:>6.3f}  {_score_bar(opp.growth_score, 15)}",
            f"  ├─ Evergreen          {opp.evergreen_score:>6.3f}  {_score_bar(opp.evergreen_score, 15)}",
            f"  ├─ Tendance           {opp.trend_score:>6.3f}  {_score_bar(opp.trend_score, 15)}",
            f"  ├─ Concurrence      : {_label_competition(opp.competition_score)}",
            f"  └─ Difficulté prod. : {_label_difficulty(opp.production_difficulty)}",
            "",
            f"  Urgence    : {_label_urgency(opp.urgency)}",
            "",
            "  Pourquoi cette opportunité :",
        ]
        for reason in opp.rationale:
            lines.append(f"    • {reason}")

        lines += [
            "",
            "  Recommandation :",
            f"    → {opp.recommendation}",
            "",
            f"  Source : https://youtu.be/{opp.source_video_id}",
        ]

    # ── Résumé global ──────────────────────────────────────────────────────────
    if opportunities:
        avg_score = sum(o.overall_score for o in opportunities) / len(opportunities)
        niches = {}
        for o in opportunities:
            niches[o.niche] = niches.get(o.niche, 0) + 1

        lines += [
            "",
            SEP,
            "  RÉSUMÉ",
            THIN,
            f"  Score moyen           : {avg_score:.4f}",
            f"  Meilleure opportunité : {opportunities[0].niche} — {opportunities[0].title[:40]}",
            "  Répartition par niche :",
        ]
        for niche, count in sorted(niches.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"    {niche:<30} {count} opportunité(s)")

    lines += ["", SEP, "  FIN DU RAPPORT", SEP]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Données
    backend = build_storage(CSV_PATH)
    snapshots = backend.load()
    if not snapshots:
        logger.error("Aucun snapshot chargé.")
        sys.exit(1)

    timelines = _build_timelines(snapshots)
    logger.info("%d snapshots → %d vidéos uniques", len(snapshots), len(timelines))

    # 2. ContentProfile
    profiles = ContentUnderstandingEngine().analyze_all(timelines)

    # 3. KnowledgeBase
    kb = KnowledgeEngine().build(profiles)

    # 4. Opportunités
    engine = OpportunityEngine()
    opportunities = engine.build(profiles, timelines, kb, top_n=TOP_N)

    # 5. Rapport
    report = build_report(opportunities)
    print("\n" + report)

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"opportunities_{timestamp}.txt"
    path.write_text(report, encoding="utf-8")
    print(f"\nRapport sauvegardé → {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
