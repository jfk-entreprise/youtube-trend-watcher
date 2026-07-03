"""
Sprint 1 — Exploration de l'API YouTube Data v3
Objectif : évaluer les capacités techniques de l'API (pas de logique métier).
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
REGION_CODE = os.getenv("YOUTUBE_REGION_CODE", "FR")
LANGUAGE = os.getenv("YOUTUBE_LANGUAGE", "fr")

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "sprint1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

QUOTA_LOG: list[dict] = []


def log_quota(operation: str, cost: int, params: dict) -> None:
    QUOTA_LOG.append({"operation": operation, "quota_cost": cost, "params": params})
    print(f"  [quota -{cost}] {operation}")


def save_json(filename: str, data: object) -> Path:
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  -> Sauvegardé : {path.name}")
    return path


def build_client():
    return build("youtube", "v3", developerKey=API_KEY)


# ---------------------------------------------------------------------------
# TEST 1 — Recherche par mot-clé (search.list)
# Quota : 100 unités par appel
# ---------------------------------------------------------------------------

def test_search_by_keyword(youtube, keyword: str = "intelligence artificielle") -> list[str]:
    print("\n=== TEST 1 : Recherche par mot-clé ===")
    params = dict(
        q=keyword,
        type="video",
        part="snippet",
        maxResults=5,
        regionCode=REGION_CODE,
        relevanceLanguage=LANGUAGE,
        order="relevance",
    )
    log_quota("search.list", 100, params)
    response = youtube.search().list(**params).execute()
    save_json("test1_search_keyword.json", response)

    video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
    print(f"  Résultats : {len(video_ids)} vidéos")
    for item in response.get("items", []):
        print(f"    • {item['snippet']['title'][:70]}")
    return video_ids


# ---------------------------------------------------------------------------
# TEST 2 — Filtrage par date de publication
# Quota : 100 unités
# ---------------------------------------------------------------------------

def test_search_with_date_filter(youtube, keyword: str = "intelligence artificielle") -> None:
    print("\n=== TEST 2 : Filtrage par date de publication ===")
    published_after = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = dict(
        q=keyword,
        type="video",
        part="snippet",
        maxResults=5,
        publishedAfter=published_after,
        regionCode=REGION_CODE,
        relevanceLanguage=LANGUAGE,
        order="date",
    )
    log_quota("search.list (publishedAfter)", 100, params)
    response = youtube.search().list(**params).execute()
    save_json("test2_search_date_filter.json", response)

    items = response.get("items", [])
    print(f"  Vidéos publiées dans les 7 derniers jours : {len(items)}")
    for item in items:
        pub = item["snippet"]["publishedAt"]
        print(f"    • [{pub}] {item['snippet']['title'][:60]}")


# ---------------------------------------------------------------------------
# TEST 3 — Récupération complète des métadonnées (videos.list)
# Quota : 1 unité par appel (peu importe le nombre d'IDs, max 50)
# ---------------------------------------------------------------------------

def test_video_metadata(youtube, video_ids: list[str]) -> None:
    print("\n=== TEST 3 : Métadonnées complètes (videos.list) ===")
    params = dict(
        id=",".join(video_ids),
        part="snippet,statistics,contentDetails,status",
    )
    log_quota("videos.list", 1, params)
    response = youtube.videos().list(**params).execute()
    save_json("test3_video_metadata.json", response)

    print(f"  {len(response.get('items', []))} vidéos détaillées :")
    for v in response.get("items", []):
        snippet = v["snippet"]
        stats = v.get("statistics", {})
        details = v.get("contentDetails", {})
        print(f"""
    Titre         : {snippet['title'][:70]}
    Chaîne        : {snippet['channelTitle']}
    Publiée le    : {snippet['publishedAt']}
    Durée (ISO)   : {details.get('duration', 'N/A')}
    Vues          : {stats.get('viewCount', 'N/A')}
    Likes         : {stats.get('likeCount', 'N/A')}
    Commentaires  : {stats.get('commentCount', 'N/A')}
    Description   : {snippet.get('description', '')[:100]}…""")


# ---------------------------------------------------------------------------
# TEST 4 — Tri par différents critères
# Quota : 100 unités × nombre d'ordres testés
# ---------------------------------------------------------------------------

def test_sort_orders(youtube, keyword: str = "machine learning") -> None:
    print("\n=== TEST 4 : Tri par différents critères ===")
    orders = ["relevance", "date", "viewCount", "rating"]
    results = {}

    for order in orders:
        params = dict(q=keyword, type="video", part="snippet", maxResults=3,
                      order=order, regionCode=REGION_CODE)
        log_quota(f"search.list (order={order})", 100, params)
        resp = youtube.search().list(**params).execute()
        titles = [item["snippet"]["title"] for item in resp.get("items", [])]
        results[order] = titles
        print(f"  order={order:<12} -> {titles[0][:55] if titles else '—'}")

    save_json("test4_sort_orders.json", results)


# ---------------------------------------------------------------------------
# TEST 5 — Vidéos tendances (videos.list category mostPopular)
# Quota : 1 unité
# ---------------------------------------------------------------------------

def test_trending_videos(youtube) -> None:
    print("\n=== TEST 5 : Vidéos tendances (chart=mostPopular) ===")
    params = dict(
        part="snippet,statistics,contentDetails",
        chart="mostPopular",
        regionCode=REGION_CODE,
        maxResults=5,
    )
    log_quota("videos.list (chart=mostPopular)", 1, params)
    response = youtube.videos().list(**params).execute()
    save_json("test5_trending.json", response)

    print(f"  Top {len(response.get('items', []))} tendances en {REGION_CODE} :")
    for v in response.get("items", []):
        stats = v.get("statistics", {})
        print(f"    • {v['snippet']['title'][:60]:<60} | vues: {stats.get('viewCount','?')}")


# ---------------------------------------------------------------------------
# TEST 6 — Tendances par catégorie
# Quota : 1 unité
# ---------------------------------------------------------------------------

def test_trending_by_category(youtube, category_id: str = "28") -> None:
    """category_id 28 = Science & Technology."""
    print("\n=== TEST 6 : Tendances par catégorie (Science & Tech) ===")
    params = dict(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=REGION_CODE,
        videoCategoryId=category_id,
        maxResults=5,
    )
    log_quota("videos.list (chart + category)", 1, params)
    response = youtube.videos().list(**params).execute()
    save_json("test6_trending_category.json", response)

    for v in response.get("items", []):
        stats = v.get("statistics", {})
        print(f"    • {v['snippet']['title'][:60]:<60} | vues: {stats.get('viewCount','?')}")


# ---------------------------------------------------------------------------
# Rapport final
# ---------------------------------------------------------------------------

def print_quota_summary() -> None:
    total = sum(q["quota_cost"] for q in QUOTA_LOG)
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES QUOTAS CONSOMMÉS")
    print("=" * 60)
    print(f"  {'Opération':<40} {'Coût':>6}")
    print(f"  {'-'*40} {'-'*6}")
    for q in QUOTA_LOG:
        print(f"  {q['operation']:<40} {q['quota_cost']:>6}")
    print(f"  {'TOTAL':.<40} {total:>6}")
    print(f"\n  Quota journalier standard : 10 000 unités")
    print(f"  Quota restant estimé      : {10_000 - total} unités")
    save_json("quota_summary.json", {"calls": QUOTA_LOG, "total_consumed": total, "daily_limit": 10_000})


def main():
    print("=" * 60)
    print("SPRINT 1 — Exploration API YouTube Data v3")
    print(f"Région : {REGION_CODE} | Langue : {LANGUAGE}")
    print("=" * 60)

    youtube = build_client()

    video_ids = test_search_by_keyword(youtube)
    test_search_with_date_filter(youtube)

    if video_ids:
        test_video_metadata(youtube, video_ids)

    test_sort_orders(youtube)
    test_trending_videos(youtube)
    test_trending_by_category(youtube)

    print_quota_summary()
    print(f"\nFichiers JSON sauvegardés dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
