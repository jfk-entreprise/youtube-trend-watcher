"""
Visual Engine Demo - Script de demonstration du Visual Engine (Sprint 18).

Affiche le plan visuel genere par HeuristicVisualGenerator a partir d'un Script,
avec comparaison ScriptScene vs VisualScene pour chaque scene.

Usage :
    python scripts/run_visual.py                          # Demo complete
    python scripts/run_visual.py --verbose                 # Affiche les prompts visuels complets
    python scripts/run_visual.py --script mini             # Utilise un mini-script (3 scenes)
    python scripts/run_visual.py --script full             # Utilise un script complet (8 scenes)

Aucun moteur existant n'est modifie.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.script_engine import Script, ScriptScene
from src.visual_engine import (
    HeuristicVisualGenerator,
    VisualEngine,
    VisualPlan,
    VisualScene,
)

SEP = "=" * 72
SEP2 = "-" * 72


def create_mini_script() -> Script:
    """Cree un mini-script avec 3 scenes."""
    scenes = [
        ScriptScene(
            order=1,
            title="Hook",
            narration="Voici pourquoi 80% des developpeurs sous-estiment l'IA.",
            visual_description="Plan d'accroche dynamique — visuel choc a l'ecran.",
            image_prompt="Dynamic abstract composition with bold typography, high contrast lighting, cinematic depth of field",
            animation_notes="Fade-in from black. Bold text animation (scale up + stabilize). 0.5s buildup.",
            sound_effects="Whoosh + impact sound. Music starts strong then drops to background.",
            duration_seconds=8,
        ),
        ScriptScene(
            order=2,
            title="Introduction",
            narration="Aujourd'hui, on va decouvrir les 5 metiers les plus impactes par l'IA.",
            visual_description="Tete parlante face camera avec musique douce en fond.",
            image_prompt="Clean professional workspace with warm ambient lighting, shallow depth of field",
            animation_notes="Crossfade transition. Gentle parallax on background. Soft text reveal.",
            sound_effects="Background music at speaking volume. Subtle room tone.",
            duration_seconds=12,
        ),
        ScriptScene(
            order=3,
            title="Point #1",
            narration="Premier metier : le developpeur full-stack va etre transforme par l'IA.",
            visual_description="Infographie ou liste animee. Numero a l'ecran.",
            image_prompt="Numbered infographic #1, clean design, accent color highlighting",
            animation_notes="Number flies in from left. Content fades below. Staggered bullet reveal.",
            sound_effects="Soft chime on number reveal. Pop sound for text line.",
            duration_seconds=16,
        ),
    ]
    return Script(
        title="5 metiers developpeur transformes par l'IA en 2027",
        hook="Voici pourquoi 80% des developpeurs sous-estiment l'IA.",
        introduction="Aujourd'hui, on va decouvrir les 5 metiers les plus impactes.",
        scenes=scenes,
        conclusion="Pour conclure, l'IA transforme mais ne remplace pas les developpeurs.",
        call_to_action="Abonne-toi pour plus d'analyses tech chaque semaine.",
        estimated_duration=36,
        language="fr",
        target_audience="Developpeurs curieux de l'IA",
        style="Innovant",
        metadata={"generator": "heuristic_v1", "angle": "Liste", "niche": "Intelligence Artificielle"},
    )


def create_full_script() -> Script:
    """Cree un script complet avec 8 scenes (structure Liste)."""
    scenes_data = [
        (1, "Hook", 8, "Hook accrocheur - texte impactant"),
        (2, "Introduction", 12, "Presentation du sujet et promesse"),
        (3, "Point #1", 16, "Le developpeur full-stack"),
        (4, "Point #2", 14, "Le data scientist"),
        (5, "Point #3", 18, "Le devOps"),
        (6, "Point bonus", 10, "Le cloud architect"),
        (7, "Conclusion", 12, "Synthese et takeaways"),
        (8, "CTA", 10, "Appel a l'action final"),
    ]
    scenes = [
        ScriptScene(
            order=o, title=t,
            narration=f"Narration pour {t}.",
            visual_description=f"Visuel pour {t}.",
            image_prompt=f"Image prompt for {t}, cinematic, professional",
            animation_notes=f"Animation notes for {t}.",
            sound_effects=f"Sound for {t}.",
            duration_seconds=d,
        )
        for o, t, d, _ in scenes_data
    ]
    return Script(
        title="Les 5 metiers IT transformes par l'IA",
        hook="90% des developpeurs vont voir leur metier transforme d'ici 2027.",
        introduction="Decouvrons ensemble les 5 metiers les plus impactes par l'IA.",
        scenes=scenes,
        conclusion="L'IA ne remplace pas, elle transforme. A vous de vous adapter.",
        call_to_action="Abonne-toi et active la cloche pour ne rien rater.",
        estimated_duration=100,
        language="fr",
        target_audience="Professionnels IT",
        style="Innovant",
        metadata={"generator": "heuristic_v1", "angle": "Liste", "niche": "Tech"},
    )


def format_visual_scene(vs: VisualScene) -> str:
    """Formate une VisualScene en texte lisible."""
    lines = [
        f"      Scene #{vs.scene_order:2d}  [{vs.duration_seconds:3d}s]",
        f"      Type plan    : {vs.shot_type}",
        f"      Mouvement    : {vs.camera_motion}",
        f"      Transition   : {vs.transition}",
        f"      Overlay      : \"{vs.overlay_text}\"",
        f"      Composition  : {vs.composition[:70]}...",
        f"      Lumiere      : {vs.lighting[:70]}...",
        f"      Palette      : {', '.join(vs.color_palette)}",
    ]
    return "\n".join(lines)


def format_visual_scene_verbose(vs: VisualScene) -> str:
    """Formate une VisualScene de maniere detaillee (mode verbose)."""
    lines = [
        f"      Scene #{vs.scene_order:2d}  [{vs.duration_seconds:3d}s]",
        f"      Type plan    : {vs.shot_type}",
        f"      Mouvement    : {vs.camera_motion}",
        f"      Transition   : {vs.transition}",
        f"      Overlay      : \"{vs.overlay_text}\"",
        f"      Composition  : {vs.composition}",
        f"      Lumiere      : {vs.lighting}",
        f"      Palette      : {', '.join(vs.color_palette)}",
        f"      Prompt visuel: {vs.visual_prompt[:100]}...",
        f"      Animation    : {vs.animation_notes[:80]}...",
        f"      Metadata     : {json.dumps(vs.metadata, ensure_ascii=False)}",
    ]
    return "\n".join(lines)


def compare_script_vs_visual(script: Script, plan: VisualPlan):
    """Compare les scenes du Script avec les scenes du VisualPlan."""
    print("  Correspondance ScriptScene -> VisualScene :")
    print()
    for i, script_scene in enumerate(script.scenes):
        visual_scene = plan.scenes[i] if i < len(plan.scenes) else None
        if visual_scene:
            print(f"    [{script_scene.order}] \"{script_scene.title}\"")
            print(f"         image_prompt original : {script_scene.image_prompt[:60]}...")
            print(f"         visual prompt genere  : {visual_scene.visual_prompt[:80]}...")
            print(f"         shot_type      : {visual_scene.shot_type} (depuis titre: {script_scene.title})")
            print(f"         camera_motion  : {visual_scene.camera_motion} (depuis titre: {script_scene.title})")
            print(f"         duree          : {script_scene.duration_seconds}s -> {visual_scene.duration_seconds}s")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Visual Engine Demo - Plan visuel a partir d'un Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Afficher les prompts visuels complets")
    parser.add_argument("--script", type=str, choices=["mini", "full"], default="mini",
                        help="Type de script de demonstration (defaut: mini)")
    args = parser.parse_args()

    # Creer le script de demo
    if args.script == "mini":
        script = create_mini_script()
    else:
        script = create_full_script()

    print(SEP)
    print("  VISUAL ENGINE v1 - Plan visuel automatique")
    print(SEP)
    print()
    print("  Script source :")
    print(f"    Titre              : {script.title}")
    print(f"    Style              : {script.style}")
    print(f"    Scenes textuelles  : {len(script.scenes)}")
    print(f"    Duree totale       : {script.estimated_duration}s "
          f"({script.estimated_duration // 60}m{script.estimated_duration % 60:02d}s)")
    print(f"    Langue             : {script.language}")
    print(f"    Audience           : {script.target_audience}")
    print()

    # Generer le VisualPlan
    print(SEP2)
    print("  GENERATION DU PLAN VISUEL")
    print(SEP2)

    start = time.time()
    engine = VisualEngine()
    plan = engine.generate(script)
    elapsed_ms = int((time.time() - start) * 1000)

    print(f"\n  Plan visuel genere en {elapsed_ms} ms :")
    print(f"    Generateur         : {engine.generator_name}")
    print(f"    Titre              : {plan.title}")
    print(f"    Style              : {plan.style}")
    print(f"    Aspect ratio       : {plan.aspect_ratio}")
    print(f"    Scenes visuelles   : {len(plan.scenes)}")
    print(f"    Palette globale    : {', '.join(plan.color_palette)}")
    print()

    # Afficher chaque scene visuelle
    print(SEP2)
    print("  DETAIL DES SCENES VISUELLES")
    print(SEP2)
    print()

    for vs in plan.scenes:
        if args.verbose:
            print(format_visual_scene_verbose(vs))
        else:
            print(format_visual_scene(vs))
        print()

    # Comparaison ScriptScene → VisualScene
    if args.verbose:
        print(SEP2)
        compare_script_vs_visual(script, plan)

    # Resume
    total_duration = sum(vs.duration_seconds for vs in plan.scenes)
    print(SEP2)
    print("  RESUME")
    print(SEP2)
    print(f"""
    Script source        : {script.title}
    Scenes (Script)      : {len(script.scenes)}
    Scenes (Visual)      : {len(plan.scenes)}
    Duree (Script)       : {script.estimated_duration}s
    Duree (Visual)       : {total_duration}s
    Aspect ratio         : {plan.aspect_ratio}
    Palette couleurs     : {', '.join(plan.color_palette)}
    Generateur           : {engine.generator_name}
    Temps de generation  : {elapsed_ms} ms
""")

    # Apercu rapide
    print(SEP2)
    print("  APERCU RAPIDE : Scene par scene")
    print(SEP2)
    print()
    print(f"  {'#':3s} {'Titre':20s} {'Plan':20s} {'Camera':15s} {'Transition':15s} {'Duree':5s}")
    print(f"  {'---':3s} {'--------------------':20s} {'--------------------':20s} {'---------------':15s} {'---------------':15s} {'-----':5s}")
    for vs in plan.scenes:
        title_short = vs.metadata.get("script_scene_title", "")[:18]
        print(f"  {vs.scene_order:3d} {title_short:20s} {vs.shot_type:20s} {vs.camera_motion:15s} {vs.transition:15s} {vs.duration_seconds:5d}s")
    print()

    print(SEP)
    print("  TEST TERMINE")
    print(SEP)
    print()


if __name__ == "__main__":
    main()
