"""
Content Understanding Engine — Script de démonstration.
Usage : python scripts/run_content_understanding.py

Charge les données via build_storage(), génère un ContentProfile pour chaque
vidéo, affiche un rapport sémantique et sauvegarde un résumé agrégé.
"""

import logging
import sys
from collections import Counter
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
from src.content_understanding import ContentProfile, ContentUnderstandingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"
SAMPLE_SIZE = 20   # nombre de profils affichés en détail


def _build_timelines(snapshots: list[VideoSnapshot]) -> list[VideoTimeline]:
    buckets: dict[str, list[VideoSnapshot]] = {}
    for snap in snapshots:
        buckets.setdefault(snap.video_id, []).append(snap)
    return [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]


def _fmt_profile(profile: ContentProfile, title: str) -> str:
    SEP = "=" * 56
    THIN = "-" * 56
    secondary = ", ".join(profile.secondary_topics) if profile.secondary_topics else "—"
    return "\n".join([
        SEP,
        f"Titre : {title[:70]}",
        THIN,
        f"Sujet principal     : {profile.primary_topic}",
        f"Sujets secondaires  : {secondary}",
        f"Langue              : {profile.language}",
        f"Audience            : {profile.target_audience}",
        f"Type de contenu     : {profile.content_type}",
        f"Émotion             : {profile.emotion}",
        f"Evergreen Score     : {profile.evergreen_score:.2f}",
        f"Trend Score         : {profile.trend_score:.2f}",
        f"Confiance           : {profile.confidence:.2f}",
        SEP,
    ])


def _build_summary_report(
    profiles: list[ContentProfile],
    timelines: list[VideoTimeline],
    elapsed_ms: float,
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SEP = "=" * 72
    THIN = "-" * 72

    # Distributions
    topic_dist = Counter(p.primary_topic for p in profiles)
    lang_dist = Counter(p.language for p in profiles)
    type_dist = Counter(p.content_type for p in profiles)
    audience_dist = Counter(p.target_audience for p in profiles)
    emotion_dist = Counter(p.emotion for p in profiles)

    avg_evergreen = sum(p.evergreen_score for p in profiles) / len(profiles)
    avg_trend = sum(p.trend_score for p in profiles) / len(profiles)
    avg_confidence = sum(p.confidence for p in profiles) / len(profiles)

    def _dist_lines(counter: Counter, label: str) -> list[str]:
        total = sum(counter.values())
        lines = [f"  {label}", THIN]
        for val, count in counter.most_common(8):
            bar = "█" * int(count / total * 30)
            lines.append(f"  {val:<28} {count:>4}  {bar}")
        return lines

    lines = [
        SEP,
        "  CONTENT UNDERSTANDING ENGINE v1 — RAPPORT AGRÉGÉ",
        f"  Généré le          : {now_str}",
        f"  Vidéos analysées   : {len(profiles)}",
        f"  Temps d'analyse    : {elapsed_ms:.0f} ms",
        f"  Confiance moyenne  : {avg_confidence:.2f}",
        SEP,
        "",
        "  MÉTRIQUES GLOBALES",
        THIN,
        f"  Evergreen Score moyen : {avg_evergreen:.3f}",
        f"  Trend Score moyen     : {avg_trend:.3f}",
        "",
        *_dist_lines(topic_dist, "DISTRIBUTION DES SUJETS"),
        "",
        *_dist_lines(lang_dist, "DISTRIBUTION DES LANGUES"),
        "",
        *_dist_lines(type_dist, "DISTRIBUTION DES TYPES DE CONTENU"),
        "",
        *_dist_lines(audience_dist, "DISTRIBUTION DES AUDIENCES"),
        "",
        *_dist_lines(emotion_dist, "DISTRIBUTION DES ÉMOTIONS"),
        "",
        SEP,
        "  FIN DU RAPPORT",
        SEP,
    ]
    return "\n".join(lines)


def main() -> None:
    backend = build_storage(CSV_PATH)
    snapshots = backend.load()

    if not snapshots:
        logger.error("Aucun snapshot chargé. Vérifiez la source de données.")
        sys.exit(1)

    timelines = _build_timelines(snapshots)
    logger.info(
        "%d snapshots → %d vidéos uniques", len(snapshots), len(timelines)
    )

    # Analyse complète
    engine = ContentUnderstandingEngine()
    t_start = datetime.now()
    profiles = engine.analyze_all(timelines)
    elapsed_ms = (datetime.now() - t_start).total_seconds() * 1000

    # ── Affichage des profils détaillés (sample) ──
    # Sélection : vidéos avec le plus de vues pour un échantillon représentatif
    sorted_tls = sorted(timelines, key=lambda tl: tl.latest.view_count or 0, reverse=True)
    sample_tls = sorted_tls[:SAMPLE_SIZE]
    profile_by_id = {p.video_id: p for p in profiles}

    print()
    print(f"{'=' * 56}")
    print(f"  CONTENT UNDERSTANDING ENGINE v1 — {SAMPLE_SIZE} profils détaillés")
    print(f"{'=' * 56}")

    for tl in sample_tls:
        profile = profile_by_id.get(tl.latest.video_id)
        if profile:
            print(_fmt_profile(profile, tl.latest.title))

    # ── Rapport agrégé ──
    summary = _build_summary_report(profiles, timelines, elapsed_ms)
    print("\n" + summary)

    # ── Sauvegarde ──
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"content_{timestamp}.txt"
    report_path.write_text(summary, encoding="utf-8")
    print(f"\nRapport agrégé sauvegardé → {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
