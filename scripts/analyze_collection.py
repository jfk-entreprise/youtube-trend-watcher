"""
Analyse du fichier data/videos.csv et génération d'un rapport textuel.
Usage : python scripts/analyze_collection.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import csv
from collections import Counter

CSV_PATH = ROOT / "data" / "videos.csv"
REPORTS_DIR = ROOT / "reports"

SHORT_MAX_SECONDS = 60       # YouTube Shorts : ≤ 60 s
LONG_MIN_SECONDS = 1200      # Vidéo longue   : ≥ 20 min
OLD_DAYS_THRESHOLD = 10      # Anomalie "trop ancienne" si > 10 jours
NOW = datetime.now(timezone.utc)


# ------------------------------------------------------------------
# Lecture
# ------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=","))


def _parse_dt(value: str) -> datetime | None:
    """Parse une chaîne ISO 8601 en datetime UTC — retourne None si invalide."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------
# Calculs
# ------------------------------------------------------------------

def analyse(rows: list[dict]) -> dict:
    total = len(rows)

    # Dédoublonnage : une même vidéo peut apparaître plusieurs fois (snapshots)
    unique_ids = {r["video_id"] for r in rows}
    unique_channels = {r["channel_id"] for r in rows}

    # Durées
    durations = [_int_or_none(r["duration_seconds"]) for r in rows]
    valid_durations = [d for d in durations if d is not None]
    avg_duration = sum(valid_durations) / len(valid_durations) if valid_durations else 0
    shorts = [d for d in valid_durations if d <= SHORT_MAX_SECONDS]
    long_videos = [d for d in valid_durations if d >= LONG_MIN_SECONDS]
    missing_duration = durations.count(None) + valid_durations.count(0)

    # Vues
    views = [_int_or_none(r["view_count"]) for r in rows]
    missing_views = views.count(None)

    # Top 20 par vues
    rows_with_views = [(r, v) for r, v in zip(rows, views) if v is not None]
    top20 = sorted(rows_with_views, key=lambda x: x[1], reverse=True)[:20]

    # Top 10 chaînes les plus représentées
    channel_counter = Counter(r["channel_title"] for r in rows)
    top10_channels = channel_counter.most_common(10)

    # Répartition par mot-clé
    keyword_counter = Counter(r["keyword"] for r in rows)

    # Âge moyen des vidéos
    pub_dates = [_parse_dt(r["published_at"]) for r in rows]
    valid_pub = [d for d in pub_dates if d is not None]
    ages_days = [(NOW - d).total_seconds() / 86400 for d in valid_pub]
    avg_age_days = sum(ages_days) / len(ages_days) if ages_days else 0

    # Anomalies
    anomalies: list[str] = []

    # Vidéos trop anciennes
    old_videos = [
        rows[i] for i, d in enumerate(pub_dates)
        if d is not None and (NOW - d).total_seconds() / 86400 > OLD_DAYS_THRESHOLD
    ]
    if old_videos:
        for v in old_videos:
            anomalies.append(
                f"  [TROP ANCIENNE] {v['title'][:60]!r} — publiée le {v['published_at'][:10]}"
            )

    # Durées nulles (hors Shorts légitimes)
    zero_dur = [
        r for r, d in zip(rows, durations)
        if d is not None and d == 0
    ]
    if zero_dur:
        for v in zero_dur:
            anomalies.append(
                f"  [DURÉE ZÉRO]   {v['title'][:60]!r} (video_id={v['video_id']})"
            )

    # Vues manquantes
    if missing_views:
        anomalies.append(
            f"  [VUES MANQUANTES] {missing_views} vidéo(s) sans statistiques de vues"
        )

    # Durées manquantes / non parsées
    if missing_duration > len(shorts):
        anomalies.append(
            f"  [DURÉE MANQUANTE] {missing_duration} durée(s) à 0 ou None"
        )

    return {
        "total": total,
        "unique_ids": len(unique_ids),
        "unique_channels": len(unique_channels),
        "avg_duration": avg_duration,
        "shorts_count": len(shorts),
        "long_count": len(long_videos),
        "missing_views": missing_views,
        "missing_duration": missing_duration,
        "top20": top20,
        "top10_channels": top10_channels,
        "keyword_dist": keyword_counter,
        "avg_age_days": avg_age_days,
        "anomalies": anomalies,
    }


# ------------------------------------------------------------------
# Mise en forme
# ------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sc = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{sc:02d}s"
    return f"{m}m{sc:02d}s"


def render_report(stats: dict, csv_path: Path) -> str:
    lines: list[str] = []
    add = lines.append

    sep = "=" * 70
    thin = "-" * 70

    add(sep)
    add("  RAPPORT D'ANALYSE — YouTube Trend Watcher")
    add(f"  Fichier : {csv_path}")
    add(f"  Généré  : {NOW.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    add(sep)

    # --- Vue d'ensemble ---
    add("")
    add("VUE D'ENSEMBLE")
    add(thin)
    add(f"  Snapshots totaux          : {stats['total']}")
    add(f"  Vidéos uniques            : {stats['unique_ids']}")
    add(f"  Chaînes uniques           : {stats['unique_channels']}")
    add(f"  Durée moyenne             : {_fmt_duration(stats['avg_duration'])}")
    add(f"  Shorts (≤ 60 s)           : {stats['shorts_count']}")
    add(f"  Vidéos longues (≥ 20 min) : {stats['long_count']}")
    add(f"  Âge moyen des vidéos      : {stats['avg_age_days']:.1f} jours")

    # --- Top 20 par vues ---
    add("")
    add("TOP 20 — VIDÉOS PAR NOMBRE DE VUES")
    add(thin)
    for rank, (row, views) in enumerate(stats["top20"], 1):
        title = row["title"][:55]
        add(f"  {rank:>2}. {views:>12,} vues  {title!r}  [{row['keyword']}]")

    # --- Top 10 chaînes ---
    add("")
    add("TOP 10 — CHAÎNES LES PLUS REPRÉSENTÉES")
    add(thin)
    for rank, (channel, count) in enumerate(stats["top10_channels"], 1):
        add(f"  {rank:>2}. {count:>3} vidéo(s)  {channel}")

    # --- Répartition par mot-clé ---
    add("")
    add("RÉPARTITION PAR MOT-CLÉ")
    add(thin)
    total_kw = sum(stats["keyword_dist"].values())
    for kw, count in sorted(stats["keyword_dist"].items(), key=lambda x: -x[1]):
        pct = count / total_kw * 100 if total_kw else 0
        bar = "#" * int(pct / 2)
        add(f"  {kw:<15} {count:>3}  ({pct:5.1f}%)  {bar}")

    # --- Anomalies ---
    add("")
    add("ANOMALIES DÉTECTÉES")
    add(thin)
    if stats["anomalies"]:
        for line in stats["anomalies"]:
            add(line)
    else:
        add("  Aucune anomalie détectée.")

    add("")
    add(sep)
    add("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    if not CSV_PATH.exists():
        print(f"[ERREUR] Fichier introuvable : {CSV_PATH}")
        sys.exit(1)

    rows = load_csv(CSV_PATH)
    if not rows:
        print("[ERREUR] Le CSV est vide.")
        sys.exit(1)

    stats = analyse(rows)
    report = render_report(stats, CSV_PATH)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"analysis_{NOW.strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"Rapport sauvegardé → {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
