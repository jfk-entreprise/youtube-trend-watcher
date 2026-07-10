#!/usr/bin/env python3
"""
Script de demonstration du Image Engine (Sprint 19).

Genere les images a partir d'un plan visuel complet.

Usage :
    python scripts/run_image.py                    # Script court (3 scenes)
    python scripts/run_image.py --script full      # Script complet (8 scenes)
    python scripts/run_image.py --script full --verbose  # Mode detaille
"""

import argparse
import sys
import time
from pathlib import Path

# Ajout du repertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visual_engine import VisualEngine
from src.image_engine import ImageEngine, HeuristicImageGenerator


# ── Scripts de demonstration ─────────────────────────────────────────────────

def make_mini_script():
    """Cree un petit script de 3 scenes pour demonstration rapide."""
    from src.script_engine import Script, ScriptScene

    hook = ScriptScene(
        order=1,
        title="Hook",
        narration="90% des developpeurs vont voir leur metier transforme d'ici 2027.",
        visual_description="Hook visuel accrocheur",
        image_prompt="Image prompt for Hook, cinematic, professional",
        animation_notes="Animation notes for Hook. Bold text animation.",
        sound_effects="Whoosh + impact sound.",
        duration_seconds=8,
    )

    intro = ScriptScene(
        order=2,
        title="Introduction",
        narration="Aujourd'hui, on va parler des metiers transforms par l'IA.",
        visual_description="Introduction du sujet",
        image_prompt="Image prompt for Introduction, cinematic, professional",
        animation_notes="Animation notes for Introduction. Gentle parallax.",
        sound_effects="Background music at speaking volume.",
        duration_seconds=12,
    )

    point1 = ScriptScene(
        order=3,
        title="Point #1",
        narration="Premier point cle : l'IA transforme deja ces metiers.",
        visual_description="Premier point cle",
        image_prompt="Image prompt for Point #1, cinematic, professional",
        animation_notes="Animation notes for Point #1. Number flies in.",
        sound_effects="Soft chime on number reveal.",
        duration_seconds=16,
    )

    return Script(
        title="5 metiers developpeur transformes par l'IA en 2027",
        hook="90% des developpeurs vont voir leur metier transforme d'ici 2027.",
        introduction="Aujourd'hui, on va parler des metiers transforms par l'IA.",
        scenes=[hook, intro, point1],
        conclusion="Pour conclure, retenez ceci : l'IA transforme tout.",
        call_to_action="Abonne-toi pour plus de contenu !",
        estimated_duration=36,
        language="fr",
        target_audience="Developpeurs curieux de l'IA",
        style="Innovant",
        metadata={"niche": "IA et developpement"},
    )


def make_full_script():
    """Cree un script complet de 8 scenes (du test, correspond a la fixture)."""
    from src.script_engine import Script, ScriptScene

    scenes_data = [
        ("Hook",           8,  "90% des developpeurs vont voir leur metier transforme d'ici "),
        ("Introduction",  12,  "Les 5 metiers IT transformes par l'IA"),
        ("Point #1",      16,  "1"),
        ("Point #2",      14,  "2"),
        ("Point #3",      18,  "3"),
        ("Point bonus",   10,  "BONUS"),
        ("Conclusion",    12,  "Conclusion"),
        ("CTA",           10,  "Abonne-toi !"),
    ]

    scenes = [
        ScriptScene(
            order=i+1,
            title=title,
            narration=f"Narration pour {title}.",
            visual_description=f"Scene {title}",
            image_prompt=f"Image prompt for {title}, cinematic, professional",
            animation_notes=f"Animation notes for {title}.",
            sound_effects=f"Effets sonores pour {title}.",
            duration_seconds=dur,
        )
        for i, (title, dur, _) in enumerate(scenes_data)
    ]

    return Script(
        title="Les 5 metiers IT transformes par l'IA",
        hook="90% des developpeurs vont voir leur metier transforme d'ici 2027.",
        introduction="Aujourd'hui, on explore les metiers IT impacts par l'IA.",
        scenes=scenes,
        conclusion="En resume, ces 5 metiers sont en pleine transformation.",
        call_to_action="Abonne-toi pour rester a jour !",
        estimated_duration=sum(d for _, d, _ in scenes_data),
        language="fr",
        target_audience="Professionnels IT",
        style="Innovant",
        metadata={"niche": "IA et IT"},
    )


# ── Affichage ────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 72
SUB_SEP = "-" * 72


def header(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(f"{SEPARATOR}\n")


def scene_detail(img, vs, plan):
    """Affiche les details d'une GeneratedImage."""
    print(f"      Scene #{img.scene_order:2d}  [{vs.duration_seconds:3d}s]")
    print(f"      Provider      : {img.provider}")
    print(f"      Dimensions    : {img.width}x{img.height}")
    print(f"      Aspect ratio  : {img.aspect_ratio}")
    print(f"      Seed          : {img.seed}")
    print(f"      Qualite       : {img.quality} ({img.steps} steps)")
    print(f"      Style         : {img.style}")
    print(f"      Prompt        : {img.prompt[:90]}...")
    print(f"      Negative      : {img.negative_prompt[:80]}...")
    print()


def scene_verbose(img, vs, plan):
    """Affiche les details complets d'une GeneratedImage."""
    print(f"      Scene #{img.scene_order:2d}  [{vs.duration_seconds:3d}s]")
    print(f"      Provider      : {img.provider}")
    print(f"      Dimensions    : {img.width}x{img.height}")
    print(f"      Aspect ratio  : {img.aspect_ratio}")
    print(f"      Seed          : {img.seed}")
    print(f"      Qualite       : {img.quality} ({img.steps} steps)")
    print(f"      Style         : {img.style}")
    print(f"      Palette couleurs : {', '.join(img.color_palette)}")
    print(f"      Prompt final  :")
    print(f"        {img.prompt}")
    print(f"      Negative prompt :")
    print(f"        {img.negative_prompt}")
    print(f"      Metadata      : {img.metadata}")
    print()


def apercu_rapide(images, plan):
    """Affiche un tableau recapitulatif rapide."""
    print(f"\n{SUB_SEP}")
    print("  APERCU RAPIDE : Image par image")
    print(f"{SUB_SEP}")
    print(f"  {'#':>3} {'Titre':20s} {'Provider':20s} {'Ratio':8s} {'Dimensions':12s} {'Seed':>11s} {'Qualite'}")
    print(f"  {'---':>3} {'--------------------':20s} {'--------------------':20s} {'--------':8s} {'------------':12s} {'-----------':>11s} {'-------'}")
    for img in plan.scenes:
        title = img.metadata.get("script_scene_title", "")[:18]
        dim = f"{img.metadata.get('width', '?')}x{img.metadata.get('height', '?')}"
        ratio = img.metadata.get("aspect_ratio", "?")
        seed_val = img.metadata.get("seed", 0)
        qualite = img.metadata.get("quality", "standard")
        provider = img.metadata.get("generator", "?")
        print(f"  {img.scene_order:3d} {title:20s} {provider:20s} {ratio:8s} {dim:12s} {seed_val:>11d} {qualite}")


def compare_script_vs_visual(script, plan, images):
    """Compare ScriptScene -> VisualScene -> GeneratedImage."""
    print(f"\n{SUB_SEP}")
    print("  Correspondance ScriptScene -> VisualScene -> GeneratedImage :")
    print(f"{SUB_SEP}")

    for i, (ss, vs) in enumerate(zip(script.scenes, plan.scenes)):
        img = images[i] if i < len(images) else None
        print(f"\n    [{ss.order}] \"{ss.title}\"")
        print(f"         image_prompt original : {ss.image_prompt[:60]}...")
        if img:
            print(f"         prompt final     : {img.prompt[:80]}...")
            print(f"         negative prompt   : {img.negative_prompt[:80]}...")
            print(f"         dimensions        : {img.width}x{img.height} ({img.aspect_ratio})")
            print(f"         seed              : {img.seed}")
            print(f"         qualite           : {img.quality} ({img.steps} steps)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Image Engine - Demonstration")
    parser.add_argument(
        "--script", choices=["mini", "full"], default="mini",
        help="Type de script (mini=3 scenes, full=8 scenes)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Affiche les prompts complets et metadonnees",
    )
    args = parser.parse_args()

    # 1. Creer le script
    script = make_full_script() if args.script == "full" else make_mini_script()

    header("IMAGE ENGINE v1 - Generation d'images")

    # 2. Afficher les infos du script source
    print("  Script source :")
    print(f"    Titre              : {script.title}")
    print(f"    Style              : {script.style}")
    print(f"    Scenes textuelles  : {len(script.scenes)}")
    print(f"    Duree totale       : {sum(s.duration_seconds for s in script.scenes)}s")
    print(f"    Langue             : {script.language}")
    print(f"    Audience           : {script.target_audience}")
    print()

    # 3. Generer le plan visuel
    print(f"{SUB_SEP}")
    print("  GENERATION DU PLAN VISUEL")
    print(f"{SUB_SEP}")
    start = time.time()
    visual_engine = VisualEngine()
    plan = visual_engine.generate(script)
    elapsed = int((time.time() - start) * 1000)

    print(f"\n  Plan visuel genere en {elapsed} ms :")
    print(f"    Generateur         : {visual_engine.generator_name}")
    print(f"    Titre              : {plan.title}")
    print(f"    Style              : {plan.style}")
    print(f"    Aspect ratio       : {plan.aspect_ratio}")
    print(f"    Scenes visuelles   : {len(plan.scenes)}")
    print(f"    Palette globale    : {', '.join(plan.color_palette)}")
    print()

    # 4. Generer les images
    print(f"{SUB_SEP}")
    print("  GENERATION DES IMAGES")
    print(f"{SUB_SEP}")
    start = time.time()
    image_engine = ImageEngine()
    images = image_engine.generate(plan)
    elapsed = int((time.time() - start) * 1000)

    print(f"\n  {len(images)} image(s) generee(s) en {elapsed} ms :")
    print(f"    Generateur         : {image_engine.generator_name}")
    print(f"    Dimensions         : {images[0].width}x{images[0].height} (defaut)" if images else "")
    print(f"    Aspect ratio       : {images[0].aspect_ratio} (defaut)" if images else "")
    print()

    # 5. Detail des scenes
    print(f"{SUB_SEP}")
    print("  DETAIL DES IMAGES GENEREES")
    print(f"{SUB_SEP}")
    print()

    for img in plan.scenes:
        # On recupere la GeneratedImage correspondante
        gen_img = next((g for g in images if g.scene_order == img.scene_order), None)
        if gen_img:
            if args.verbose:
                scene_verbose(gen_img, img, plan)
            else:
                scene_detail(gen_img, img, plan)

    # 6. Comparaison Script -> Visual -> Image (si verbose)
    if args.verbose:
        compare_script_vs_visual(script, plan, images)

    # 7. Resume
    print(f"{SUB_SEP}")
    print("  RESUME")
    print(f"{SUB_SEP}")
    print()
    total_images = len(images)
    total_steps = sum(i.steps for i in images)
    hd_count = sum(1 for i in images if i.quality == "hd")
    print(f"    Script source        : {script.title}")
    print(f"    Scenes (Script)      : {len(script.scenes)}")
    print(f"    Scenes (Visuel)      : {len(plan.scenes)}")
    print(f"    Images generees      : {total_images}")
    print(f"    Dimensions           : {images[0].width}x{images[0].height}" if images else "")
    print(f"    Aspect ratio         : {images[0].aspect_ratio}" if images else "")
    print(f"    Palette couleurs     : {', '.join(plan.color_palette)}")
    print(f"    Generator (Visuel)   : {visual_engine.generator_name}")
    print(f"    Generator (Image)    : {image_engine.generator_name}")
    print(f"    HD images            : {hd_count}/{total_images}")
    print(f"    Total inference steps: {total_steps}")
    print(f"    Temps de generation  : {elapsed} ms")
    print()

    # 8. Apercu rapide
    # On injecte les donnees dans les metadata des scenes du plan pour l'affichage
    for img in images:
        # Les scenes du plan sont immuables, on utilise un mapping par scene_order
        pass
    print(f"{SUB_SEP}")
    print("  APERCU RAPIDE : Scene par scene")
    print(f"{SUB_SEP}")
    print(f"  {'#':>3} {'Titre':20s} {'Provider':20s} {'Ratio':8s} {'Dimensions':12s} {'Seed':>11s} {'Qualite'}")
    print(f"  {'---':>3} {'--------------------':20s} {'--------------------':20s} {'--------':8s} {'------------':12s} {'-----------':>11s} {'-------'}")
    for img in sorted(images, key=lambda i: i.scene_order):
        title = plan.scenes[img.scene_order - 1].metadata.get("script_scene_title", "")[:18]
        dim = f"{img.width}x{img.height}"
        print(f"  {img.scene_order:3d} {title:20s} {img.provider:20s} {img.aspect_ratio:8s} {dim:12s} {img.seed:>11d} {img.quality}")

    print(f"\n{SEPARATOR}")
    print("  TEST TERMINE")
    print(f"{SEPARATOR}")
    print()


if __name__ == "__main__":
    main()
