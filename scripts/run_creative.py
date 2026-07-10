"""
Creative Engine — Script de lancement.
Usage : python scripts/run_creative.py

Pipeline complet :
  1. Snapshots via build_storage()
  2. ContentProfile via ContentUnderstandingEngine (Sprint 9)
  3. KnowledgeBase via KnowledgeEngine (Sprint 10)
  4. Opportunités via OpportunityEngine (Sprint 11)
  5. CreativeBrief via CreativeEngine (Sprint 12)
  6. Rapport texte + sauvegarde
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
from src.creative_engine import CreativeBrief, CreativeEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"
TOP_N_OPP = 5    # opportunités à développer créativement
TOP_N_BRIEFS = 5  # briefs max par opportunité


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _build_timelines(snapshots: list[VideoSnapshot]) -> list[VideoTimeline]:
    buckets: dict[str, list[VideoSnapshot]] = {}
    for snap in snapshots:
        buckets.setdefault(snap.video_id, []).append(snap)
    return [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]


# ── Formatage ─────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _score_bar(score: float, width: int = 12) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _originality_label(score: float) -> str:
    if score >= 0.85:
        return "Très original"
    if score >= 0.65:
        return "Original"
    if score >= 0.45:
        return "Modéré"
    return "Similaire aux autres"


def build_report(
    opportunities: list[Opportunity],
    briefs_map: dict[str, list[CreativeBrief]],
    generator_name: str,
) -> str:
    SEP = "=" * 76
    THIN = "-" * 76
    DOTTED = "·" * 76
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_briefs = sum(len(v) for v in briefs_map.values())

    lines = [
        SEP,
        "  CREATIVE ENGINE v1 — CONCEPTS ÉDITORIAUX",
        f"  Généré le              : {now_str}",
        f"  Générateur             : {generator_name}",
        f"  Opportunités traitées  : {len(opportunities)}",
        f"  CreativeBrief produits : {total_briefs}",
        SEP,
    ]

    for opp_rank, opp in enumerate(opportunities, 1):
        briefs = briefs_map.get(opp.source_video_id, [])

        lines += [
            "",
            f"  ╔══ OPPORTUNITÉ #{opp_rank:02d} {'═' * 50}",
            f"  ║  Titre    : {opp.title[:68]}",
            f"  ║  Niche    : {opp.niche}  |  Score : {opp.overall_score:.4f}",
            f"  ║  Format   : {opp.metadata.get('content_type', '?')}  "
            f"|  Audience : {opp.metadata.get('target_audience', '?')}",
            f"  ║  Briefs   : {len(briefs)} variante(s) générée(s)",
            f"  ╚{'═' * 60}",
        ]

        for b_rank, brief in enumerate(briefs, 1):
            dur_label = _fmt_duration(brief.duration_seconds)
            orig_bar = _score_bar(brief.originality_score, 10)
            orig_label = _originality_label(brief.originality_score)

            lines += [
                "",
                f"  ┌─── Variante {b_rank} / {len(briefs)} — [{brief.angle.upper()}]",
                DOTTED,
                f"  │  Titre proposé  : {brief.title}",
                f"  │  Angle          : {brief.angle}",
                f"  │  Format         : {brief.format}  |  Durée cible : {dur_label}",
                f"  │  Audience       : {brief.audience}  |  Émotion : {brief.emotion}",
                f"  │  Originalité    : {brief.originality_score:.3f}  {orig_bar}  ({orig_label})",
                "  │",
                f"  │  HOOK",
                f"  │  → {brief.hook}",
                "  │",
                f"  │  PROMESSE",
                f"  │  → {brief.promise}",
                "  │",
                "  │  STRUCTURE NARRATIVE",
            ]

            for step_i, step in enumerate(brief.structure):
                connector = "└─" if step_i == len(brief.structure) - 1 else "├─"
                lines.append(f"  │    {connector} {step}")

            lines += [
                "  │",
                f"  │  STYLE VISUEL",
                f"  │  → {brief.visual_style}",
                "  │",
                f"  │  CTA",
                f"  │  → {brief.cta}",
                "  │",
                "  │  NOTES DE PRODUCTION",
            ]
            for note in brief.production_notes:
                lines.append(f"  │    • {note}")

            lines += [
                "  │",
                "  │  JUSTIFICATION",
            ]
            for reason in brief.rationale:
                lines.append(f"  │    → {reason}")

            lines.append(f"  └{'─' * 64}")

    # ── Résumé global ──────────────────────────────────────────────────────────
    all_briefs = [b for briefs in briefs_map.values() for b in briefs]

    if all_briefs:
        avg_orig = sum(b.originality_score for b in all_briefs) / len(all_briefs)
        avg_dur = sum(b.duration_seconds for b in all_briefs) / len(all_briefs)

        angle_counts: dict[str, int] = {}
        for b in all_briefs:
            angle_counts[b.angle] = angle_counts.get(b.angle, 0) + 1

        lines += [
            "",
            SEP,
            "  RÉSUMÉ GLOBAL",
            THIN,
            f"  Total CreativeBrief      : {len(all_briefs)}",
            f"  Originalité moyenne      : {avg_orig:.3f}",
            f"  Durée cible moyenne      : {_fmt_duration(int(avg_dur))}",
            "  Répartition par angle :",
        ]
        for angle, count in sorted(angle_counts.items(), key=lambda x: x[1], reverse=True):
            bar = _score_bar(count / len(all_briefs), 15)
            lines.append(f"    {angle:<25} {count:>3} brief(s)  {bar}")

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
    opportunities = OpportunityEngine().build(profiles, timelines, kb, top_n=TOP_N_OPP)
    logger.info("%d opportunités sélectionnées", len(opportunities))

    # 5. CreativeBrief
    engine = CreativeEngine()
    briefs_map = engine.generate_all(opportunities)

    # 6. Rapport
    report = build_report(opportunities, briefs_map, engine.generator_name)
    print("\n" + report)

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"creative_{timestamp}.txt"
    path.write_text(report, encoding="utf-8")
    print(f"\nRapport sauvegardé → {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
