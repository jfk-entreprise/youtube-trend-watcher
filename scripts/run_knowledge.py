"""
Knowledge Engine — Script de lancement.
Usage : python scripts/run_knowledge.py

Pipeline complet :
  1. Chargement des snapshots via build_storage() (Supabase ou CSV).
  2. Génération des ContentProfile via ContentUnderstandingEngine.
  3. Construction de la KnowledgeBase via KnowledgeEngine.
  4. Rapport texte + export JSON.
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
from src.knowledge_engine import KnowledgeBase, KnowledgeEngine, KnowledgeFact

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"


# ── Pipeline helpers ──────────────────────────────────────────────────────────

def _build_timelines(snapshots: list[VideoSnapshot]) -> list[VideoTimeline]:
    buckets: dict[str, list[VideoSnapshot]] = {}
    for snap in snapshots:
        buckets.setdefault(snap.video_id, []).append(snap)
    return [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]


# ── Rapport texte ─────────────────────────────────────────────────────────────

def _fmt_dur(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    return f"{m}m{s:02d}s" if m < 60 else f"{m // 60}h{m % 60:02d}m{s:02d}s"


def _bar(n: int, total: int, width: int = 28) -> str:
    filled = int(n / max(total, 1) * width)
    return "█" * filled


def _fmt_pct(n: int, total: int) -> str:
    return f"{n / max(total, 1) * 100:.1f}%"


def _section_distribution(
    title: str,
    facts: list[KnowledgeFact],
    total: int,
    show_scores: bool = True,
) -> list[str]:
    SEP = "-" * 72
    lines = ["", f"  {title}", SEP]
    for i, f in enumerate(facts, 1):
        v = f.value
        freq = v.get("frequency", f.observations)
        pct = v.get("pct", round(freq / total * 100, 1))
        bar = _bar(freq, total)
        score_str = ""
        if show_scores:
            avg_t = v.get("avg_trend_score", "")
            avg_e = v.get("avg_evergreen_score", "")
            if avg_t != "" and avg_e != "":
                score_str = f"  trend:{avg_t:.2f}  green:{avg_e:.2f}"
        lines.append(
            f"  {i:>2}. {f.description:<28} {freq:>4}  ({pct:>5.1f}%)  {bar}{score_str}"
        )
    return lines


def build_report(kb: KnowledgeBase) -> str:
    now_str = kb.generated_at.strftime("%Y-%m-%d %H:%M:%S")
    SEP = "=" * 72
    THIN = "-" * 72
    T = kb.total_profiles

    lines = [
        SEP,
        "  KNOWLEDGE ENGINE v1 — BASE DE CONNAISSANCES MARCHÉ",
        f"  Générée le           : {now_str}",
        f"  ContentProfile traités : {T}",
        f"  Sujets détectés      : {len(kb.topics)}",
        f"  Combinaisons trouvées : {len(kb.combinations)}",
        SEP,
    ]

    # ── Distributions catégorielles ───────────────────────────────────────────
    lines += _section_distribution("TOP SUJETS", kb.top("topics", 10), T)
    lines += _section_distribution("TOP ÉMOTIONS", kb.top("emotions", 10), T)
    lines += _section_distribution("TOP AUDIENCES", kb.top("audiences", 10), T)
    lines += _section_distribution("TOP FORMATS DE CONTENU", kb.top("content_types", 10), T)
    lines += _section_distribution("TOP LANGUES", kb.top("languages", 10), T, show_scores=False)

    # ── Analyse Evergreen ─────────────────────────────────────────────────────
    ev = kb.evergreen.value
    lines += [
        "", "  ANALYSE EVERGREEN (pérennité du contenu)", THIN,
        f"  Score moyen   : {ev['mean']:.3f}",
        f"  Score médian  : {ev['median']:.3f}",
        "  Sujets les plus pérennes :",
    ]
    for item in ev.get("top_topics", []):
        lines.append(f"    {item['topic']:<30} {item['avg_score']:.3f}")

    # ── Analyse Trend ─────────────────────────────────────────────────────────
    tr = kb.trend.value
    lines += [
        "", "  ANALYSE TREND (dynamique de tendance)", THIN,
        f"  Score moyen           : {tr['mean']:.3f}",
        f"  Vidéos haute tendance (≥ 0.7) : {tr['high_trend_count']}",
        f"  Vidéos basse tendance (< 0.4) : {tr['low_trend_count']}",
        "  Sujets les plus en tendance :",
    ]
    for item in tr.get("top_topics", []):
        lines.append(f"    {item['topic']:<30} {item['avg_score']:.3f}")

    # ── Analyse Durées ────────────────────────────────────────────────────────
    dur = kb.durations.value
    if "note" in dur:
        lines += ["", "  ANALYSE DES DURÉES", THIN, f"  {dur['note']}"]
    else:
        lines += [
            "", "  ANALYSE DES DURÉES", THIN,
            f"  Durée moyenne       : {dur['mean_fmt']} ({dur['mean_seconds']}s)",
            f"  Durée médiane       : {dur['median_fmt']} ({dur['median_seconds']}s)",
            f"  Plage dominante     : {dur['best_range_by_volume']}",
            "",
            f"  {'Plage':<20} {'Volume':>7}  {'Trend moyen':>12}",
            f"  {'─'*20} {'─'*7}  {'─'*12}",
        ]
        for label, count in dur["ranges"].items():
            trend = dur["avg_trend_by_range"].get(label, 0.0)
            bar = _bar(count, T, 15)
            lines.append(
                f"  {label:<20} {count:>7}  {trend:>12.3f}  {bar}"
            )

    # ── Combinaisons ──────────────────────────────────────────────────────────
    top_combos = kb.top_combinations(15)
    lines += ["", "  TOP COMBINAISONS (co-occurrences fréquentes)", THIN]
    if not top_combos:
        lines.append("  Aucune combinaison trouvée (données insuffisantes).")
    else:
        lines += [
            f"  {'#':<4} {'Combinaison':<52} {'Vol':>5}  {'%':>5}",
            f"  {'─'*4} {'─'*52} {'─'*5}  {'─'*5}",
        ]
        for i, combo in enumerate(top_combos, 1):
            v = combo.value
            label = combo.description[:50]
            scope_badge = "[3]" if combo.metadata.get("scope") == "triple" else "[2]"
            lines.append(
                f"  {i:<4} {scope_badge} {label:<50} {v['frequency']:>5}  {v['pct']:>4.1f}%"
            )

    lines += ["", SEP, "  FIN DU RAPPORT", SEP]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Chargement des données
    backend = build_storage(CSV_PATH)
    snapshots = backend.load()
    if not snapshots:
        logger.error("Aucun snapshot chargé.")
        sys.exit(1)

    timelines = _build_timelines(snapshots)
    logger.info("%d snapshots → %d vidéos uniques", len(snapshots), len(timelines))

    # 2. ContentProfile via ContentUnderstandingEngine
    cue = ContentUnderstandingEngine()
    profiles = cue.analyze_all(timelines)

    # 3. KnowledgeBase via KnowledgeEngine
    engine = KnowledgeEngine()
    kb = engine.build(profiles)

    # 4. Rapport texte
    report = build_report(kb)
    print("\n" + report)

    # 5. Sauvegarde
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    txt_path = REPORTS_DIR / f"knowledge_{timestamp}.txt"
    txt_path.write_text(report, encoding="utf-8")

    json_path = REPORTS_DIR / f"knowledge_{timestamp}.json"
    json_path.write_text(kb.to_json(), encoding="utf-8")

    print(f"\nRapport texte → {txt_path.relative_to(ROOT)}")
    print(f"Export JSON   → {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
