"""
Script de démonstration du Learning Engine v1 (Sprint 15).

Pipeline complet :
    Opportunity + CreativeBrief + BrandProfile + Script
    → PerformanceMetrics (simulées à partir de données réelles)
    → LearningEngine.record() → signaux
    → LearningEngine.build() → LearningProfile
    → Affichage des meilleures dimensions

Usage :
    python scripts/run_learning.py                          # Pipeline complet (marque par défaut)
    python scripts/run_learning.py --brand ia_fr             # Marque spécifique
    python scripts/run_learning.py --brand histoire_fr       # Marque spécifique
    python scripts/run_learning.py --brand business_fr       # Marque spécifique
    python scripts/run_learning.py --top 2                   # Top 2 opportunités
    python scripts/run_learning.py --output                  # Sauvegarde en JSON

Ce script simule des PerformanceMetrics en générant des données
réalistes à partir des données réelles collectées (views, likes, etc.).
Dans un environnement de production, ces métriques viendraient de YouTube Studio.
"""

import argparse
import json
import math
import random
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brand_engine import BrandEngine
from src.creative_engine import CreativeEngine
from src.learning_engine import (
    LearningEngine, LearningProfile, PerformanceMetrics, LearningSignal,
    JsonLearningStore,
)
from src.opportunity_engine import OpportunityEngine
from src.script_engine import ScriptEngine
from src.virality_engine import ViralityEngine
from src.content_understanding import ContentUnderstandingEngine
from src.knowledge_engine import KnowledgeEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Learning Engine — Démonstration Sprint 15")
    parser.add_argument("--brand", type=str, default="ia_fr",
                        help="Identifiant de la marque (défaut: ia_fr)")
    parser.add_argument("--top", type=int, default=1,
                        help="Nombre d'opportunités à traiter (défaut: 1)")
    parser.add_argument("--output", action="store_true",
                        help="Sauvegarde le profil en JSON dans data/learning/")
    args = parser.parse_args()

    csv_path = Path("data/videos.csv")
    if not csv_path.exists():
        logger.error("Fichier CSV introuvable : %s", csv_path)
        sys.exit(1)

    logger.info("=== Learning Engine — Démonstration Sprint 15 ===")
    logger.info("Marque : %s | Top N : %d", args.brand, args.top)

    # ── 1-2. Chargement des données ───────────────────────────────────────────
    logger.info("[1/6] Chargement des données depuis %s ...", csv_path.name)
    ve = ViralityEngine(csv_path)
    timelines = ve._load_timelines()
    logger.info("  → %d timelines chargées", len(timelines))

    # ── 3. ContentUnderstanding ───────────────────────────────────────────────
    logger.info("[2/6] Analyse sémantique...")
    cue = ContentUnderstandingEngine()
    profiles = cue.analyze_all(timelines)
    logger.info("  → %d ContentProfile", len(profiles))

    # ── 4. KnowledgeBase ─────────────────────────────────────────────────────
    logger.info("[3/6] Construction de la KnowledgeBase...")
    ke = KnowledgeEngine()
    kb = ke.build(profiles)
    logger.info("  → %d sujets, %d combinaisons", len(kb.topics), len(kb.combinations))

    # ── 5. Opportunités ─────────────────────────────────────────────────────
    logger.info("[4/6] Détection des opportunités...")
    oe = OpportunityEngine()
    opportunities = oe.build(profiles, timelines, kb, top_n=args.top)
    logger.info("  → %d opportunités", len(opportunities))

    # ── 6. Briefs créatifs ──────────────────────────────────────────────────
    logger.info("[5/6] Génération des briefs créatifs...")
    ce = CreativeEngine()
    briefs_map = ce.generate_all(opportunities)
    total_briefs = sum(len(b) for b in briefs_map.values())
    logger.info("  → %d briefs", total_briefs)

    # ── 7. Marque ───────────────────────────────────────────────────────────
    logger.info("[6/6] Chargement de la marque '%s'...", args.brand)
    be = BrandEngine()
    brand = be.load(args.brand)
    if brand is None:
        disponibles = [p.id for p in be.list()]
        logger.error(
            "Marque '%s' introuvable. Disponibles : %s",
            args.brand, ", ".join(disponibles),
        )
        sys.exit(1)

    # ── 8. Scripts ──────────────────────────────────────────────────────────
    logger.info("Génération des scripts...")
    se = ScriptEngine()
    scripts_map = se.generate_all(opportunities, briefs_map, brand)
    total_scripts = sum(len(s) for s in scripts_map.values())
    logger.info("  → %d scripts", total_scripts)

    # ── 9. Learning Engine ────────────────────────────────────────────────────
    logger.info("=== Learning Engine ===")

    # Simuler des PerformanceMetrics pour chaque script
    all_signals: list[LearningSignal] = []
    total_videos = 0

    for video_id, scripts in scripts_map.items():
        opp = next((o for o in opportunities if o.source_video_id == video_id), None)
        if opp is None:
            continue

        # Récupérer la timeline pour générer des métriques réalistes
        tl = next((t for t in timelines if t.video_id == video_id), None)

        for script in scripts:
            # Trouver le brief correspondant (même titre)
            briefs = briefs_map.get(video_id, [])
            brief = next((b for b in briefs if b.title == script.title), None)
            if brief is None:
                continue

            # Générer des PerformanceMetrics simulées mais réalistes
            metrics = _simulate_metrics(video_id, tl, opp, script, brief)

            # Enregistrer
            engine = LearningEngine()
            signals = engine.record(opp, brief, brand, script, metrics)
            all_signals.extend(signals)
            total_videos += 1

            logger.info(
                "  ✓ Vidéo '%s' (angle: %s) → score: %.4f | views: %d | retention: %.1f%%",
                script.title[:40],
                brief.angle,
                metrics.performance_score,
                metrics.views,
                metrics.retention * 100,
            )

    # ── 10. Construction du profil ──────────────────────────────────────────
    logger.info("Construction du LearningProfile...")
    profile = engine.build(all_signals, brand_id=brand.id)

    # ── 11. Affichage ──────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  LEARNING ENGINE v1 — RAPPORT D'APPRENTISSAGE")
    print(f"  Marque          : {brand.name} ({brand.id})")
    print(f"  Vidéos analysées : {total_videos}")
    print(f"  Signaux collectés: {len(all_signals)}")
    print(f"  Dimensions       : {len(profile.dimensions)} ({', '.join(profile.dimensions)})")
    print("=" * 72)

    # Meilleur hook
    hook, h_score = profile.best_hook()
    print(f"\n  ▶ MEILLEUR HOOK   : {hook[:70]}")
    print(f"    Score           : {h_score:.4f}")

    # Meilleur angle
    angle, a_score = profile.best_angle()
    print(f"\n  ▶ MEILLEUR ANGLE  : {angle}")
    print(f"    Score           : {a_score:.4f}")

    # Meilleure durée
    dur, d_score = profile.best_duration()
    print(f"\n  ▶ MEILLEURE DURÉE : {dur} secondes ({_fmt_duration(int(dur))})")
    print(f"    Score           : {d_score:.4f}")

    # Meilleur CTA
    cta, c_score = profile.best_cta()
    print(f"\n  ▶ MEILLEUR CTA    : {cta[:70]}")
    print(f"    Score           : {c_score:.4f}")

    # Meilleur style
    style, s_score = profile.best_style()
    print(f"\n  ▶ MEILLEUR STYLE  : {style}")
    print(f"    Score           : {s_score:.4f}")

    # Meilleure émotion
    emotion, e_score = profile.best_emotion()
    print(f"\n  ▶ MEILLEURE ÉMOTION : {emotion}")
    print(f"    Score           : {e_score:.4f}")

    # Meilleur format
    fmt, f_score = profile.best_format()
    print(f"\n  ▶ MEILLEUR FORMAT : {fmt}")
    print(f"    Score           : {f_score:.4f}")

    # Top 3 hooks
    print(f"\n  ▶ TOP 3 HOOKS :")
    for i, (h, s) in enumerate(profile.top_hooks(3), 1):
        print(f"    {i}. [{s:.4f}] {h[:70]}")

    # Top 3 angles
    print(f"\n  ▶ TOP 3 ANGLES :")
    for i, (a, s) in enumerate(profile.top_angles(3), 1):
        print(f"    {i}. [{s:.4f}] {a}")

    # Top 3 CTA
    print(f"\n  ▶ TOP 3 CTA :")
    for i, (c, s) in enumerate(profile.top_ctas(3), 1):
        print(f"    {i}. [{s:.4f}] {c[:70]}")

    # Résumé JSON
    print(f"\n  ▶ RÉSUMÉ :")
    summary = profile.summary()
    print(f"     {json.dumps(summary, ensure_ascii=False, indent=2)}")

    # ── 12. Sauvegarde ─────────────────────────────────────────────────────
    if args.output:
        store = JsonLearningStore("data/learning")
        engine = LearningEngine(store=store)
        engine.save(brand.id, profile)

        # Recharge pour vérifier
        reloaded = engine.load(brand.id)
        if reloaded:
            logger.info("Vérification rechargement : %d signaux — OK", reloaded.total_signals)

    print()
    print("=" * 72)
    print("  APPRENTISSAGE TERMINÉ — Learning Engine v1")
    print("=" * 72)


# ── Simulation réaliste de métriques ─────────────────────────────────────────

def _simulate_metrics(
    video_id: str,
    tl: any,
    opp: any,
    script: any,
    brief: any,
) -> PerformanceMetrics:
    """
    Génère des PerformanceMetrics simulées mais réalistes.

    Utilise les données réelles de la timeline pour la magnitude,
    et applique des variations aléatoires pour différencier les angles.

    Dans un environnement de production, ces métriques viendraient
    de YouTube Studio.
    """
    # Base : vues réelles depuis la timeline si disponibles
    if tl and tl.latest and tl.latest.view_count:
        base_views = tl.latest.view_count
    else:
        base_views = random.randint(500, 15000)

    # Facteur d'angle : chaque angle performe différemment
    angle_factor = {
        "Liste": random.uniform(0.8, 1.2),
        "Histoire": random.uniform(0.7, 1.4),
        "Erreurs fréquentes": random.uniform(0.9, 1.3),
        "Comparaison": random.uniform(0.7, 1.1),
        "Challenge": random.uniform(0.6, 1.5),
    }.get(brief.angle, random.uniform(0.7, 1.3))

    views = int(base_views * angle_factor * random.uniform(0.9, 1.1))
    views = max(100, views)

    likes = int(views * random.uniform(0.02, 0.08))
    comments = int(views * random.uniform(0.003, 0.02))

    retention = random.uniform(0.25, 0.65)

    watch_time = views * script.estimated_duration * retention / 3600
    watch_time = max(1.0, watch_time)

    impressions_ctr = random.uniform(0.04, 0.18)

    shares = int(views * random.uniform(0.002, 0.015))

    subscribers_gained = int(views * random.uniform(0.005, 0.03))

    return PerformanceMetrics(
        video_id=video_id,
        views=views,
        likes=likes,
        comments=comments,
        retention=round(retention, 4),
        watch_time=round(watch_time, 1),
        impressions_ctr=round(impressions_ctr, 4),
        shares=shares,
        subscribers_gained=subscribers_gained,
    )


def _fmt_duration(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


if __name__ == "__main__":
    main()
