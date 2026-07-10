"""
LLM Script Generator - Script de demonstration et comparaison.

Compare un Script genere par :
  1. HeuristicScriptGenerator (approche actuelle - templates)
  2. LLMScriptGenerator     (approche IA - via build_llm())

Affiche pour chaque generateur :
  - nombre de scenes
  - duree totale estimee
  - temps de generation
  - tokens (LLM seulement)
  - cout estime (LLM seulement)
  - extrait du script

Usage :
    python scripts/run_llm_script.py                          # Auto (provider prioritaire)
    python scripts/run_llm_script.py --provider claude         # Forcer un provider
    python scripts/run_llm_script.py --model gpt-4o-mini       # Forcer un modele
    python scripts/run_llm_script.py --no-llm                  # Skips LLM (test heuristique seul)
    python scripts/run_llm_script.py --verbose                 # Affiche les scripts complets

Aucun moteur existant n'est modifie.
Le LLM Provider est utilise exclusivement via build_llm().
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.brand_engine import BrandProfile
from src.creative_engine import CreativeBrief
from src.llm_script_generator import LLMScriptGenerator
from src.opportunity_engine import Opportunity
from src.script_engine import HeuristicScriptGenerator, ScriptEngine

SEP = "=" * 72
SEP2 = "-" * 72


def create_demo_opportunity() -> Opportunity:
    """Cree une Opportunity de demonstration (niche IA)."""
    return Opportunity(
        title="L'IA va-t-elle remplacer les developpeurs ?",
        niche="Intelligence Artificielle",
        source_video_id="demo_ia_001",
        overall_score=0.85,
        virality_score=0.72,
        growth_score=0.65,
        evergreen_score=0.80,
        trend_score=0.78,
        competition_score=0.45,
        production_difficulty=0.50,
        urgency=0.70,
        recommendation=(
            "Produire rapidement - tendance active sur le sujet. "
            "Niche en croissance, angle Liste adapte."
        ),
        rationale=[
            "Potentiel viral eleve (score composite 0.72)",
            "Sujet perenne - evergreen 0.80",
            "En tendance active (trend 0.78)",
        ],
        metadata={
            "demo": True,
            "niche": "Intelligence Artificielle",
            "source": "Script de demonstration Sprint 17",
        },
    )


def create_demo_brief() -> CreativeBrief:
    """Cree un CreativeBrief de demonstration (angle Liste)."""
    return CreativeBrief(
        opportunity_id="demo_ia_001",
        title="5 metiers de developpeur que l'IA va transformer en 2027",
        angle="Liste",
        hook="Voici pourquoi 80% des developpeurs sous-estiment l'impact de l'IA.",
        promise="Dans cette video, je vais vous montrer les 5 metiers les plus impactes par l'IA.",
        audience="Developpeurs et ingenieurs curieux de l'impact de l'IA sur leur carriere.",
        emotion="Informatif",
        format="Analyse",
        duration_seconds=480,
        structure=[
            "Hook accrocheur",
            "Introduction",
            "Point #1",
            "Point #2",
            "Point #3",
            "Point #4",
            "Point #5",
            "Conclusion",
            "CTA",
        ],
        visual_style="Graphiques et donnees visuelles, slides epures, voix off posee",
        cta="Abonne-toi pour ne pas rater notre prochaine analyse sur les tendances tech.",
        originality_score=0.85,
        production_notes=[
            "Citer les sources a l'ecran",
            "Preparer les donnees avant le tournage",
        ],
        rationale=[
            "Angle liste : forte retention, SEO efficace",
            "Sujet porteur dans la niche IA",
        ],
        metadata={
            "niche": "Intelligence Artificielle",
            "language": "fr",
            "opportunity_score": 0.85,
            "urgency": 0.70,
        },
    )


def create_demo_brand() -> BrandProfile:
    """Charge le BrandProfile reel ia_fr depuis le disque."""
    from src.brand_engine import JsonBrandStore
    store = JsonBrandStore(Path(__file__).resolve().parent.parent / "brands")
    profile = store.load("ia_fr")
    if profile is None:
        raise RuntimeError("Impossible de charger le profil brand 'ia_fr'. Fichier introuvable.")
    return profile


def format_scenes_summary(scenes) -> str:
    """Formate un resume des scenes."""
    lines = []
    for s in scenes:
        duration = s.duration_seconds
        title = s.title
        narration_preview = s.narration[:80].replace("\n", " ").strip()
        lines.append(f"      {s.order:2d}. [{duration:3d}s] {title}")
        lines.append(f"           {narration_preview}...")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Comparaison Script Heuristique vs Script LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--provider", type=str, default=None,
                        help="Provider LLM (openai, gemini, claude)")
    parser.add_argument("--model", type=str, default=None,
                        help="Modele LLM specifique")
    parser.add_argument("--no-llm", action="store_true",
                        help="Ignorer la generation LLM (test heuristique seul)")
    parser.add_argument("--verbose", action="store_true",
                        help="Afficher les scripts complets")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature LLM (defaut: 0.7)")
    args = parser.parse_args()

    # Preparer les donnees de demontration
    opportunity = create_demo_opportunity()
    brief = create_demo_brief()
    brand = create_demo_brand()

    print(SEP)
    print("  LLM SCRIPT GENERATOR - SPRINT 17")
    print(SEP)
    print()
    print("  Donnees de demonstration :")
    print(f"    Opportunite             : {opportunity.title}")
    print(f"    Niche                   : {opportunity.niche}")
    print(f"    Score                   : {opportunity.overall_score}/100")
    print(f"    Angle                   : {brief.angle}")
    print(f"    Hook                    : {brief.hook[:60]}...")
    print(f"    Marque                  : {brand.name} ({brand.tone})")
    print(f"    Duree cible             : {brief.duration_seconds}s")
    print()

    # Phase 1 : HeuristicScriptGenerator
    print(SEP2)
    print("  [1] HEURISTIC SCRIPT GENERATOR (templates)")
    print(SEP2)

    start = time.time()
    heuristic_gen = HeuristicScriptGenerator()
    heuristic_script = heuristic_gen.generate(opportunity, brief, brand)
    heuristic_time = int((time.time() - start) * 1000)

    print(f"\n  Resultat :")
    print(f"    Generateur        : {heuristic_gen.name}")
    print(f"    Titre             : {heuristic_script.title}")
    print(f"    Scenes            : {len(heuristic_script.scenes)}")
    print(f"    Duree totale      : {heuristic_script.estimated_duration}s "
          f"({heuristic_script.estimated_duration // 60}m{heuristic_script.estimated_duration % 60:02d}s)")
    print(f"    Style             : {heuristic_script.style}")
    print(f"    Temps de gene     : {heuristic_time} ms")
    print(f"    Cout              : $0.000000 (heuristique - aucun appel API)")
    print()
    print(f"  Scenes :")
    print(format_scenes_summary(heuristic_script.scenes))

    if args.verbose:
        print()
        print(f"  Script complet (title + hook + introduction) :")
        print(f"    Titre         : {heuristic_script.title}")
        print(f"    Hook          : {heuristic_script.hook}")
        print(f"    Introduction  : {heuristic_script.introduction[:100]}...")
        print(f"    Conclusion    : {heuristic_script.conclusion[:100]}...")
        print(f"    CTA           : {heuristic_script.call_to_action}")
        print(f"    Metadonnees   : {json.dumps(heuristic_script.metadata, ensure_ascii=False, indent=4)[:300]}...")

    print()

    # Variables pour la phase 2 (doivent etre visibles apres)
    llm_gen = None
    llm_script = None
    llm_time = 0
    stats = {}

    # Phase 2 : LLMScriptGenerator
    if not args.no_llm:
        print(SEP2)
        print("  [2] LLM SCRIPT GENERATOR (IA)")
        print(SEP2)

        start = time.time()
        llm_gen = LLMScriptGenerator(
            provider_name=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=4096,
        )
        try:
            llm_script = llm_gen.generate(opportunity, brief, brand)
            llm_time = int((time.time() - start) * 1000)
            stats = llm_gen.stats

            fallback_tag = ""
            if stats["fallbacks"] > 0:
                fallback_tag = " [FALLBACK VERS HEURISTIQUE]"

            print(f"\n  Resultat{fallback_tag} :")
            print(f"    Generateur        : {llm_gen.name}")
            print(f"    Titre             : {llm_script.title}")
            print(f"    Scenes            : {len(llm_script.scenes)}")
            print(f"    Duree totale      : {llm_script.estimated_duration}s "
                  f"({llm_script.estimated_duration // 60}m{llm_script.estimated_duration % 60:02d}s)")
            print(f"    Style             : {llm_script.style}")
            print(f"    Temps de gene     : {llm_time} ms (dont LLM: {stats['total_time_ms']} ms)")
            print(f"    Appels LLM        : {stats['llm_calls']} (succes: {stats['llm_success']}, echecs: {stats['llm_failures']})")
            print(f"    Tokens            : {stats['total_prompt_tokens']} -> {stats['total_completion_tokens']} "
                  f"(total: {stats['total_prompt_tokens'] + stats['total_completion_tokens']})")
            print(f"    Cout estime       : ${stats['total_cost_usd']:.6f} USD")
            print()
            print(f"  Scenes :")

            if len(llm_script.scenes) > 0:
                print(format_scenes_summary(llm_script.scenes))
            else:
                print("    (aucune scene - fallback actif)")

            if args.verbose:
                print()
                print(f"  Detail LLM :")
                print(f"    Provider utilise  : {llm_script.metadata.get('llm_provider', '?')}")
                print(f"    Modele utilise    : {llm_script.metadata.get('llm_model', '?')}")
                print(f"    Temps LLM         : {llm_script.metadata.get('llm_time_ms', '?')} ms")
                print(f"    Tokens LLM        : {llm_script.metadata.get('llm_tokens', '?')}")
                print(f"    Cout LLM          : ${llm_script.metadata.get('llm_cost_usd', 0):.6f}")
                print()
                print(f"  Script complet :")
                print(f"    Titre         : {llm_script.title}")
                print(f"    Hook          : {llm_script.hook}")
                print(f"    Introduction  : {llm_script.introduction[:100]}...")
                print(f"    Conclusion    : {llm_script.conclusion[:100]}...")
                print(f"    CTA           : {llm_script.call_to_action}")
                print(f"    Metadonnees   : {json.dumps(llm_script.metadata, ensure_ascii=False, indent=4)[:400]}...")

        except Exception as e:
            llm_time = int((time.time() - start) * 1000)
            print(f"\n  [ERR] ERREUR : {e}")

        print()

    # Comparaison
    print(SEP)
    print("  COMPARAISON")
    print(SEP)

    llm_name = llm_gen.name if llm_gen and not args.no_llm else "(non teste)"
    llm_scenes = len(llm_script.scenes) if llm_script else 0
    llm_duration = llm_script.estimated_duration if llm_script else 0
    llm_display_time = f"{llm_time:4d} ms" if llm_time else "?"
    llm_cost = stats.get("total_cost_usd", 0) if stats else 0

    print(f"""
  Critere               Heuristique          LLM
  ---------------------------------------------------
  Generateur             {heuristic_gen.name:20s} {llm_name:20s}
  Scenes                 {len(heuristic_script.scenes):3d}                  {llm_scenes:3d}
  Duree                  {heuristic_script.estimated_duration:4d}s               {llm_duration:4d}s
  Temps                  {heuristic_time:4d} ms              {llm_display_time}
  Cout                   $0.000000           ${llm_cost:.6f}
""")

    # Resume
    print()
    print("  Resume :")
    print(f"    - HeuristicGenerator  : {len(heuristic_script.scenes)} scenes, {heuristic_script.estimated_duration}s, {heuristic_time}ms")
    if not args.no_llm:
        if stats and stats["fallbacks"] > 0:
            print(f"    - LLMGenerator        : Fallback actif ({stats['fallbacks']} fois)")
        elif llm_script:
            print(f"    - LLMGenerator        : {len(llm_script.scenes)} scenes, {llm_script.estimated_duration}s, "
                  f"{llm_time}ms, ${llm_cost:.6f}")
    print(f"    - Provider LLM        : {(args.provider or 'auto-detection')} / "
          f"{(args.model or 'defaut du provider')}")
    print()
    print(SEP)
    print("  TEST TERMINE")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
