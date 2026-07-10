"""
Script de démonstration du Script Engine v1 (Sprint 14).

Pipeline complet :
    Opportunity (via données réelles)
    → CreativeEngine (via CreativeBrief)
    → BrandEngine (via BrandProfile)
    → ScriptEngine (via Script)

Usage :
    python scripts/run_script.py                      # Top 1 opportunité, marque par défaut
    python scripts/run_script.py --brand ia_fr         # Marque spécifique
    python scripts/run_script.py --brand histoire_fr   # Marque spécifique
    python scripts/run_script.py --top 3               # Top 3 opportunités
    python scripts/run_script.py --output             # Sauvegarde les scripts en JSON

Le script affiche pour chaque script généré :
    - Titre
    - Hook
    - Durée estimée
    - Nombre de scènes
    - Détail de chaque scène (ordre, titre, narration, durée)
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# Chemin du projet pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brand_engine import BrandEngine
from src.creative_engine import CreativeEngine
from src.opportunity_engine import OpportunityEngine
from src.script_engine import ScriptEngine, HeuristicScriptGenerator
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
    parser = argparse.ArgumentParser(description="Script Engine — Démonstration Sprint 14")
    parser.add_argument("--brand", type=str, default="ia_fr",
                        help="Identifiant de la marque (défaut: ia_fr)")
    parser.add_argument("--top", type=int, default=1,
                        help="Nombre d'opportunités à traiter (défaut: 1)")
    parser.add_argument("--output", action="store_true",
                        help="Sauvegarde les scripts en JSON dans reports/")
    args = parser.parse_args()

    # ── 1. Chargement des données ─────────────────────────────────────────────
    csv_path = Path("data/videos.csv")
    if not csv_path.exists():
        logger.error("Fichier CSV introuvable : %s. Lancez d'abord test_agents.py", csv_path)
        sys.exit(1)

    logger.info("=== Script Engine — Démonstration Sprint 14 ===")
    logger.info("Marque : %s | Top N : %d", args.brand, args.top)

    # ── 2. ViralityEngine → timelines ──────────────────────────────────────────
    logger.info("[1/6] Chargement des données depuis %s ...", csv_path.name)
    ve = ViralityEngine(csv_path)
    timelines = ve._load_timelines()
    logger.info("  → %d timelines chargées", len(timelines))

    # ── 3. ContentUnderstandingEngine → profiles ───────────────────────────────
    logger.info("[2/6] Analyse sémantique des vidéos...")
    cue = ContentUnderstandingEngine()
    profiles = cue.analyze_all(timelines)
    logger.info("  → %d ContentProfile générés", len(profiles))

    # ── 4. KnowledgeEngine → KnowledgeBase ─────────────────────────────────────
    logger.info("[3/6] Construction de la KnowledgeBase...")
    ke = KnowledgeEngine()
    kb = ke.build(profiles)
    logger.info("  → %d sujets, %d combinaisons", len(kb.topics), len(kb.combinations))

    # ── 5. OpportunityEngine → Opportunities ──────────────────────────────────
    logger.info("[4/6] Détection des opportunités...")
    oe = OpportunityEngine()
    opportunities = oe.build(profiles, timelines, kb, top_n=args.top)
    logger.info("  → %d opportunités détectées", len(opportunities))

    # ── 6. CreativeEngine → CreativeBriefs ────────────────────────────────────
    logger.info("[5/6] Génération des briefs créatifs...")
    ce = CreativeEngine()
    briefs_map = ce.generate_all(opportunities)
    total_briefs = sum(len(b) for b in briefs_map.values())
    logger.info("  → %d briefs créatifs générés", total_briefs)

    # ── 7. BrandEngine → BrandProfile ─────────────────────────────────────────
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

    # ── 8. ScriptEngine → Scripts ─────────────────────────────────────────────
    logger.info("Génération des scripts...")
    se = ScriptEngine()
    scripts_map = se.generate_all(opportunities, briefs_map, brand)
    total_scripts = sum(len(s) for s in scripts_map.values())
    logger.info("  → %d scripts générés", total_scripts)

    # ── 9. Affichage ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("  SCRIPT ENGINE v1 — RAPPORT DE DÉMONSTRATION")
    print(f"  Marque          : {brand.name} ({brand.id})")
    print(f"  Top N opportunités : {args.top}")
    print(f"  Briefs générés  : {total_briefs}")
    print(f"  Scripts générés : {total_scripts}")
    print(f"  Générateur      : {se.generator_name}")
    print("=" * 72)

    for rank, (video_id, scripts) in enumerate(scripts_map.items(), 1):
        opp = next((o for o in opportunities if o.source_video_id == video_id), None)
        opp_title = opp.title[:60] if opp else video_id[:60]
        print()
        print(f"  ▸ Opportunité #{rank} : {opp_title}")
        print(f"    Score : {opp.overall_score:.4f} | Niche : {opp.niche}" if opp else "")

        for script_idx, script in enumerate(scripts, 1):
            print()
            print(f"    ── Script #{script_idx} : {script.title[:60]}")
            print(f"       Hook       : {script.hook[:80]}")
            print(f"       Durée      : {_fmt_duration(script.estimated_duration)}")
            print(f"       Scènes     : {len(script.scenes)}")
            print(f"       Style      : {script.style}")
            print(f"       Langue     : {script.language}")
            print(f"       Audience   : {script.target_audience}")

            for scene in script.scenes:
                print()
                print(f"       [{scene.order:02d}] {scene.title}")
                print(f"            Narration  : {scene.narration[:100]}")
                print(f"            Durée      : {scene.duration_seconds}s")
                print(f"            Visuel     : {scene.visual_description[:80]}")
                print(f"            Animation  : {scene.animation_notes[:80]}")
                print(f"            Son        : {scene.sound_effects[:60]}")
                print(f"            Image      : {scene.image_prompt[:80]}")

            # Ligne de séparation entre scripts
            if script_idx < len(scripts):
                print(f"       ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")

    # ── 10. Sauvegarde (optionnelle) ─────────────────────────────────────────
    if args.output:
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for video_id, scripts in scripts_map.items():
            data = {
                "generated_at": timestamp,
                "brand_id": brand.id,
                "brand_name": brand.name,
                "generator": se.generator_name,
                "scripts": [_script_to_dict(s) for s in scripts],
            }
            path = output_dir / f"script_{video_id[:12]}_{timestamp}.json"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Scripts sauvegardés → %s", path)

    print()
    print("=" * 72)
    print("  DÉMONSTRATION TERMINÉE — Script Engine v1")
    print("=" * 72)


# ── Formateurs ────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: int) -> str:
    """Formate une durée en secondes en format lisible."""
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _script_to_dict(script: "Script") -> dict:
    """Convertit un Script en dictionnaire pour sérialisation JSON."""
    return {
        "title": script.title,
        "hook": script.hook,
        "introduction": script.introduction,
        "conclusion": script.conclusion,
        "call_to_action": script.call_to_action,
        "estimated_duration": script.estimated_duration,
        "language": script.language,
        "target_audience": script.target_audience,
        "style": script.style,
        "scenes": [
            {
                "order": s.order,
                "title": s.title,
                "narration": s.narration,
                "visual_description": s.visual_description,
                "image_prompt": s.image_prompt,
                "animation_notes": s.animation_notes,
                "sound_effects": s.sound_effects,
                "duration_seconds": s.duration_seconds,
            }
            for s in script.scenes
        ],
        "metadata": script.metadata,
    }


if __name__ == "__main__":
    main()
