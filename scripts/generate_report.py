"""
Pipeline CI — Génération du rapport de synthèse Markdown.
Sprint 7.1 : source de données Supabase en priorité, CSV en fallback,
via build_storage() — aucune modification des agents ni du Virality Engine.

Usage : python scripts/generate_report.py
Produit : reports/YYYY-MM-DD_HH-MM.md
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
from src.virality_engine import DEFAULT_CRITERIA, VideoTimeline, _fmt_dur, _fmt_views

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"
TOP_N = 10


def _build_timelines(snapshots: list[VideoSnapshot]) -> list[VideoTimeline]:
    buckets: dict[str, list[VideoSnapshot]] = {}
    for snap in snapshots:
        buckets.setdefault(snap.video_id, []).append(snap)
    return [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]


def _build_top10_table(timelines: list[VideoTimeline], warnings: list[str]) -> str:
    if not timelines:
        warnings.append("Aucune donnée disponible pour le classement.")
        return "_Aucune donnée disponible._"

    scored = sorted(
        ((tl, sum(c.score(tl) * c.weight for c in DEFAULT_CRITERIA)) for tl in timelines),
        key=lambda pair: pair[1],
        reverse=True,
    )[:TOP_N]

    rows = [
        "| # | Titre | Chaîne | Score | Vues | Durée | Source |",
        "|:-:|-------|--------|------:|-----:|-------|--------|",
    ]
    for rank, (tl, score) in enumerate(scored, 1):
        v = tl.latest
        title_link = f"[{v.title[:55]}](https://youtu.be/{v.video_id})"
        rows.append(
            f"| {rank} | {title_link} | {v.channel_title[:35]} "
            f"| {score:.2f} | {_fmt_views(v.view_count)} "
            f"| {_fmt_dur(v.duration_seconds)} | {v.source} |"
        )
    return "\n".join(rows)


def main() -> None:
    warnings: list[str] = []
    now = datetime.now()
    timestamp_label = now.strftime("%Y-%m-%d %H:%M")
    filename_ts = now.strftime("%Y-%m-%d_%H-%M")

    logger.info("=== Rapport de synthèse CI — %s ===", timestamp_label)

    backend = build_storage(CSV_PATH)
    snapshots = backend.load()

    raw_count = len(snapshots)
    unique_count = len({s.video_id for s in snapshots})
    logger.info("Snapshots chargés : %d bruts | %d vidéos uniques", raw_count, unique_count)

    if raw_count == 0:
        warnings.append("Aucun snapshot chargé — base de données vide ou inaccessible.")

    timelines = _build_timelines(snapshots)
    top10_md = _build_top10_table(timelines, warnings)
    logger.info("Top %d construit.", TOP_N)

    warnings_section = (
        "\n".join(f"- {w}" for w in warnings)
        if warnings
        else "_Aucune erreur ni avertissement._"
    )

    report = f"""\
# Rapport de veille YouTube — {timestamp_label}

## Résumé de l'exécution

| Paramètre | Valeur |
|:----------|-------:|
| Date / Heure | {timestamp_label} |
| Snapshots collectés (bruts) | {raw_count} |
| Vidéos uniques | {unique_count} |
| Doublons supprimés | {raw_count - unique_count} |

## Top {TOP_N} — Virality Engine

{top10_md}

## Avertissements & Erreurs

{warnings_section}
"""

    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"{filename_ts}.md"
    out_path.write_text(report, encoding="utf-8")
    logger.info("Rapport sauvegardé → %s", out_path.relative_to(ROOT))
    print(f"\nRapport : {out_path}")


if __name__ == "__main__":
    main()
