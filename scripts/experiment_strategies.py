"""
Laboratoire d'expérimentation — Comparaison de 5 stratégies API YouTube Data v3.

Usage : python scripts/experiment_strategies.py

Lecture seule : ce script ne modifie pas data/videos.csv ni aucun fichier existant.
Il génère uniquement un rapport dans reports/experiment_<timestamp>.txt.

Coût quota estimé : ~2 560 unités (5 stratégies × 5 mots-clés × 100 + batches videos.list)
"""

import logging
import os
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from googleapiclient.discovery import build

from src.models import VideoSnapshot

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

KEYWORDS = ["IA", "Business", "Argent", "Histoire", "Technologie"]
MAX_RESULTS = 20       # par mot-clé, max API = 50
DAYS_BACK = 7          # fenêtre pour les stratégies C et E
REGION_CODE = os.getenv("YOUTUBE_REGION_CODE", "FR")
LANGUAGE = os.getenv("YOUTUBE_LANGUAGE", "fr")

_ISO_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")

# ── Méta-données des stratégies (pros/cons) ─────────────────────────────────────

STRATEGIES_META: dict[str, dict] = {
    "A": {
        "name": "Pertinence YouTube (baseline)",
        "description": "order=relevance | sans filtre de date ni de durée",
        "pros": [
            "Reflète le classement algorithmique natif de YouTube",
            "Diversité maximale : mélange de formats et de créateurs",
            "Capture des contenus populaires toutes périodes confondues",
        ],
        "cons": [
            "Mélange contenus récents et anciens (parfois plusieurs années)",
            "L'algorithme de pertinence YouTube est opaque et variable",
            "Faible contrôle sur la distribution temporelle des résultats",
            "Peu adapté à la détection de tendances actuelles",
        ],
    },
    "B": {
        "name": "Tri par popularité (viewCount)",
        "description": "order=viewCount | sans filtre de date",
        "pros": [
            "Retourne directement les vidéos au plus fort impact sur le sujet",
            "Vues élevées = signal de qualité et d'intérêt public validé",
            "Idéal pour cartographier les créateurs dominants d'une niche",
            "Benchmark solide pour calibrer un seuil de viralité",
        ],
        "cons": [
            "Biaisé vers les contenus anciens (plus de temps pour accumuler des vues)",
            "Rate totalement les vidéos virales naissantes (< 48h)",
            "Sur-représente le contenu 'evergreen' au détriment des tendances",
            "Peu utile pour un système de détection précoce",
        ],
    },
    "C": {
        "name": "Récence stricte — 7 derniers jours",
        "description": "publishedAfter=7j | order=date (méthode la plus proche de la collecte actuelle)",
        "pros": [
            "Garantit la fraîcheur absolue du contenu collecté (≤ 7 jours)",
            "Capture les tendances et sujets en cours de discussion",
            "Adapté à un monitoring hebdomadaire régulier",
            "Idéal comme flux d'entrée d'un pipeline de détection",
        ],
        "cons": [
            "Vues faibles : les vidéos n'ont pas eu le temps de se propager",
            "Difficile de distinguer viral de banal à si court terme",
            "Nécessite des collectes répétées pour suivre la courbe de croissance",
        ],
    },
    "D": {
        "name": "Shorts uniquement (≤ 60 secondes)",
        "description": "videoDuration=short (< 4min API) | filtrage post-API ≤ 60s | order=relevance",
        "pros": [
            "Cible le format le plus viral de YouTube en 2025-2026",
            "Les Shorts bénéficient d'un flux algorithmique dédié et massif",
            "Boucle de lecture automatique = métriques d'engagement très élevées",
            "Données propres : durée précisément bornée et vérifiable",
        ],
        "cons": [
            "Exclut tout le contenu long-format potentiellement viral",
            "Volume très réduit après filtrage strict à 60 secondes",
            "Le paramètre API 'short' couvre jusqu'à 4 min — filtrage additionnel indispensable",
            "Pertinence fragilisée si les mots-clés sont peu représentés en Shorts",
        ],
    },
    "E": {
        "name": "Hybride — Récent + Populaire + Court",
        "description": "publishedAfter=7j | order=viewCount | videoDuration=short | filtrage ≤ 60s",
        "pros": [
            "Signal combiné le plus fort pour détecter la viralité naissante",
            "Récent (≤ 7j) + déjà en traction = vidéo en phase de percée",
            "Format court = diffusion algorithmique maximisée",
            "Meilleur ratio signal/bruit pour un moteur de détection de tendances",
        ],
        "cons": [
            "Triple filtre = volume de résultats potentiellement très faible",
            "Risque de retours vides pour des mots-clés de niche peu représentés en Shorts",
            "Le tri par vues sur 7j peut encore favoriser les créateurs établis",
            "Nécessite un corpus de mots-clés élargi pour compenser le faible yield",
        ],
    },
}

# ── Utilitaires ────────────────────────────────────────────────────────────────

def _parse_duration(iso: str) -> int:
    m = _ISO_RE.match(iso or "")
    if not m:
        return 0
    h, mn, s = (int(g or 0) for g in m.groups())
    return h * 3600 + mn * 60 + s


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _fmt_dur(sec: int) -> str:
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_views(n: Optional[float]) -> str:
    if n is None:
        return "N/A"
    n = int(n)
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


# ── Appels API ─────────────────────────────────────────────────────────────────

def _search_and_fetch(
    youtube,
    keyword: str,
    max_results: int,
    *,
    order: str = "relevance",
    published_after: Optional[str] = None,
    video_duration: Optional[str] = None,
) -> list[VideoSnapshot]:
    """search.list (100 unités) + videos.list (1 unité / 50 IDs)."""
    params: dict = dict(
        q=keyword,
        type="video",
        part="id",
        maxResults=min(max_results, 50),
        regionCode=REGION_CODE,
        relevanceLanguage=LANGUAGE,
        order=order,
    )
    if published_after:
        params["publishedAfter"] = published_after
    if video_duration:
        params["videoDuration"] = video_duration

    try:
        search_resp = youtube.search().list(**params).execute()
    except Exception as exc:
        logger.error("search.list [%s] : %s", keyword, exc)
        return []

    video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
    if not video_ids:
        return []

    try:
        details_resp = (
            youtube.videos()
            .list(id=",".join(video_ids), part="snippet,statistics,contentDetails")
            .execute()
        )
    except Exception as exc:
        logger.error("videos.list [%s] : %s", keyword, exc)
        return []

    snapshots = []
    for item in details_resp.get("items", []):
        vid_id = item["id"]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        details = item.get("contentDetails", {})
        duration_iso = details.get("duration", "PT0S")
        snapshots.append(
            VideoSnapshot(
                video_id=vid_id,
                title=snippet.get("title", ""),
                channel_id=snippet.get("channelId", ""),
                channel_title=snippet.get("channelTitle", ""),
                published_at=snippet.get("publishedAt", ""),
                description=snippet.get("description", "")[:500],
                duration_iso=duration_iso,
                duration_seconds=_parse_duration(duration_iso),
                view_count=_safe_int(stats.get("viewCount")),
                like_count=_safe_int(stats.get("likeCount")),
                comment_count=_safe_int(stats.get("commentCount")),
                keyword=keyword,
            )
        )
    return snapshots


# ── Exécution d'une stratégie ──────────────────────────────────────────────────

def run_strategy(youtube, sid: str) -> list[VideoSnapshot]:
    """
    Paramétrage selon l'identifiant de stratégie :
      A → relevance, pas de filtre date
      B → viewCount, pas de filtre date
      C → date, publishedAfter 7j
      D → relevance, videoDuration=short, filtre post ≤60s
      E → viewCount, publishedAfter 7j, videoDuration=short, filtre post ≤60s
    """
    published_after: Optional[str] = None
    if sid in ("C", "E"):
        published_after = (
            datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    if sid in ("B", "E"):
        order = "viewCount"
    elif sid == "C":
        order = "date"
    else:
        order = "relevance"

    video_duration: Optional[str] = "short" if sid in ("D", "E") else None
    filter_60s = sid in ("D", "E")

    all_videos: list[VideoSnapshot] = []
    for kw in KEYWORDS:
        logger.info("  [%s] %-15s …", sid, kw)
        videos = _search_and_fetch(
            youtube, kw, MAX_RESULTS,
            order=order,
            published_after=published_after,
            video_duration=video_duration,
        )
        if filter_60s:
            before = len(videos)
            videos = [v for v in videos if v.duration_seconds <= 60]
            if before != len(videos):
                logger.info("    filtrage ≤60s : %d → %d", before, len(videos))
        all_videos.extend(videos)

    # Déduplique sur video_id (priorité au premier mot-clé trouvant la vidéo)
    seen: set[str] = set()
    unique: list[VideoSnapshot] = []
    for v in all_videos:
        if v.video_id not in seen:
            seen.add(v.video_id)
            unique.append(v)
    return unique


# ── Métriques ──────────────────────────────────────────────────────────────────

def compute_metrics(videos: list[VideoSnapshot]) -> dict:
    now = datetime.now(timezone.utc)
    shorts = [v for v in videos if v.duration_seconds <= 60]
    durations = [v.duration_seconds for v in videos]
    views = [v.view_count for v in videos if v.view_count is not None]

    ages: list[float] = []
    for v in videos:
        try:
            ages.append((now - _parse_dt(v.published_at)).total_seconds() / 86400)
        except Exception:
            pass

    top10 = sorted(
        [v for v in videos if v.view_count is not None],
        key=lambda v: v.view_count,  # type: ignore[arg-type]
        reverse=True,
    )[:10]

    return {
        "total": len(videos),
        "shorts_count": len(shorts),
        "avg_duration_s": statistics.mean(durations) if durations else 0.0,
        "avg_views": float(statistics.mean(views)) if views else 0.0,
        "median_views": float(statistics.median(views)) if views else 0.0,
        "avg_age_days": statistics.mean(ages) if ages else 0.0,
        "top10": top10,
    }


# ── Génération du rapport ──────────────────────────────────────────────────────

def generate_report(results: dict[str, dict]) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    SEP = "=" * 72
    THIN = "-" * 72

    def pct(n: int, total: int) -> str:
        return f"{n / total * 100:.0f}%" if total else "—"

    lines = [
        SEP,
        "  RAPPORT D'EXPERIMENTATION — STRATEGIES API YOUTUBE DATA v3",
        f"  Genere le   : {now_str}",
        f"  Mots-cles   : {', '.join(KEYWORDS)}",
        f"  Fenetre     : {DAYS_BACK} jours (strategies C & E)",
        f"  Region      : {REGION_CODE}  |  Langue : {LANGUAGE}",
        f"  Max resultats par mot-cle : {MAX_RESULTS}",
        SEP,
        "",
    ]

    # ── Section par stratégie ──
    for sid, data in results.items():
        meta = STRATEGIES_META[sid]
        m = data["metrics"]
        total = m["total"]

        lines += [
            THIN,
            f"  STRATEGIE {sid}  —  {meta['name']}",
            f"  {meta['description']}",
            THIN,
            "",
            "  METRIQUES",
            f"  {'Volume total':<38} {total} videos",
            f"  {'Format Shorts (≤ 60s)':<38} {m['shorts_count']} videos ({pct(m['shorts_count'], total)})",
            f"  {'Duree moyenne':<38} {_fmt_dur(int(m['avg_duration_s']))}",
            f"  {'Vues moyennes':<38} {_fmt_views(m['avg_views'])}",
            f"  {'Vues medianes':<38} {_fmt_views(m['median_views'])}",
            f"  {'Age moyen des videos':<38} {m['avg_age_days']:.1f} jours",
            "",
            "  TOP 10 VIDEOS PAR VUES",
        ]

        if m["top10"]:
            for i, v in enumerate(m["top10"], 1):
                try:
                    age_j = (datetime.now(timezone.utc) - _parse_dt(v.published_at)).days
                    age_s = f"{age_j}j"
                except Exception:
                    age_s = "?"
                title = v.title[:46].rstrip() + ("…" if len(v.title) > 46 else "")
                lines.append(
                    f"  {i:2}. [{_fmt_views(v.view_count):>8}]"
                    f" [{_fmt_dur(v.duration_seconds):>7}]"
                    f" [{age_s:>4}]  {title}"
                )
        else:
            lines.append("  Aucune video avec donnees de vues.")

        lines += ["", "  AVANTAGES"]
        for pro in meta["pros"]:
            lines.append(f"    + {pro}")
        lines += ["", "  INCONVENIENTS"]
        for con in meta["cons"]:
            lines.append(f"    - {con}")
        lines.append("")

    # ── Tableau comparatif ──
    hdr = f"  {'Strat.':<6}  {'Videos':>7}  {'Shorts':>7}  {'%Shorts':>7}  {'DureeMoy':>9}  {'VuesMoy':>9}  {'VuesMed':>9}  {'AgeMoy':>8}"
    lines += [
        SEP,
        "  TABLEAU COMPARATIF — SYNTHESE",
        SEP,
        "",
        hdr,
        "  " + THIN,
    ]
    for sid, data in results.items():
        m = data["metrics"]
        lines.append(
            f"  {sid:<6}  {m['total']:>7}  {m['shorts_count']:>7}  "
            f"{pct(m['shorts_count'], m['total']):>7}  "
            f"{_fmt_dur(int(m['avg_duration_s'])):>9}  "
            f"{_fmt_views(m['avg_views']):>9}  "
            f"{_fmt_views(m['median_views']):>9}  "
            f"{m['avg_age_days']:>7.1f}j"
        )
    lines.append("")

    # ── Recommandation finale ──
    lines += [
        SEP,
        "  RECOMMANDATION FINALE",
        SEP,
        "",
        "  Question : Quelle strategie constitue la meilleure base technique",
        "  pour construire notre futur moteur de detection de videos virales ?",
        "",
        "  REPONSE : STRATEGIE E (Hybride) — couplée a la Strategie C",
        THIN,
        "",
        "  Justification",
        "  ─────────────",
        "  La Strategie E génère le signal le plus precisement calibre pour",
        "  detecter une video en phase de viralisation :",
        "    • Recente (≤ 7j)  → on capture la fenêtre d'acceleration",
        "    • En traction (viewCount)  → deja validee par l'audience",
        "    • Format Short (≤ 60s)  → diffusion algorithmique maximale",
        "",
        "  Cependant, son triple filtre produit peu de resultats par mot-cle.",
        "  Elle doit donc être couplée à la Strategie C (flux large) pour",
        "  constituer un pipeline complet :",
        "",
        "    PHASE 1 — Collecte large (Strategie C)",
        "      Capture le flux hebdomadaire complet (tous formats, ≤ 7j)",
        "      → Alimente un historique de reference pour le scoring",
        "",
        "    PHASE 2 — Detection de percees (Strategie E)",
        "      Isole les videos deja en traction, recentes, au format viral",
        "      → Produit les alertes du moteur de detection",
        "",
        "    PHASE 3 — Score de viralite composite (evolution future)",
        "      Croissance des vues / age × facteur format",
        "      Necessite des collectes repetees — Strategie C comme base",
        "",
        "  Recommandations operationnelles",
        "  ───────────────────────────────",
        "    1. Elargir le corpus a 15-20 mots-cles pour compenser le faible",
        "       yield de la Strategie E.",
        "    2. Executer la Strategie C en collecte quotidienne (surveillance).",
        "    3. Executer la Strategie E en collecte bi-quotidienne (alertes).",
        "    4. Conserver la Strategie B pour des audits mensuels de niche.",
        "",
        "  Tableau de correspondance cas d'usage / strategie",
        "  ───────────────────────────────────────────────────",
        f"  {'Cas d\'usage':<48} {'Strategie recommandee'}",
        f"  {'─'*48} {'─'*22}",
        f"  {'Detection precoce (viralite naissante)':<48} E (Hybride)",
        f"  {'Monitoring regulier du flux hebdomadaire':<48} C (Recence 7j)",
        f"  {'Cartographie des acteurs dominants d\'une niche':<48} B (Popularite)",
        f"  {'Exploration / audit initial d\'un sujet':<48} A (Pertinence)",
        f"  {'Recherche Shorts viraux exclusivement':<48} D (Shorts)",
        "",
        SEP,
    ]

    return "\n".join(lines)


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY absent du fichier .env — arret.")
        sys.exit(1)

    youtube = build("youtube", "v3", developerKey=api_key)

    quota_est = len(STRATEGIES_META) * (len(KEYWORDS) * 100 + len(KEYWORDS))
    logger.info("=== Laboratoire d'experimentation — 5 strategies ===")
    logger.info("Mots-cles : %s", ", ".join(KEYWORDS))
    logger.info("Quota estime : ~%d unites / 10 000 disponibles aujourd'hui", quota_est)
    logger.info("")

    all_results: dict[str, dict] = {}
    for sid, meta in STRATEGIES_META.items():
        logger.info("── Strategie %s : %s", sid, meta["name"])
        videos = run_strategy(youtube, sid)
        metrics = compute_metrics(videos)
        all_results[sid] = {"videos": videos, "metrics": metrics}
        logger.info(
            "   → %d videos | %d Shorts | moy. %s | %s vues moy. | %.1fj age moy.",
            metrics["total"],
            metrics["shorts_count"],
            _fmt_dur(int(metrics["avg_duration_s"])),
            _fmt_views(metrics["avg_views"]),
            metrics["avg_age_days"],
        )
        logger.info("")

    report = generate_report(all_results)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"experiment_{timestamp}.txt"
    report_path.write_text(report, encoding="utf-8")

    logger.info("Rapport sauvegarde → %s", report_path.relative_to(ROOT))
    print("\n" + report)


if __name__ == "__main__":
    main()
