#!/usr/bin/env python3
"""
Benchmark — Prompt Heuristique vs Prompt LLM pour la génération d'image
(Sprint 23).

Compare, scène par scène, les prompts produits par HeuristicImageGenerator
et par LLMImageGenerator sur les 8 dimensions attendues d'un prompt
professionnel : composition, caméra, lumière, ambiance, couleurs,
profondeur, style cinématographique, niveau de détail.

Méthodologie (transparente, pas une note de qualité d'image réelle — aucun
rendu n'est effectué) :
  Pour chaque prompt, on détecte par mots-clés la présence de chacune des
  8 dimensions et on calcule un score de couverture /8. C'est un indicateur
  de RICHESSE DESCRIPTIVE du prompt, pas de qualité visuelle finale.

Usage :
    python scripts/run_image_prompt_benchmark.py                     # brand ia_fr, provider deepseek
    python scripts/run_image_prompt_benchmark.py --brand histoire_fr
    python scripts/run_image_prompt_benchmark.py --provider groq
"""

import argparse
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.brand_engine import BrandEngine
from src.image_engine import GeneratedImage, HeuristicImageGenerator
from src.llm_image_generator import LLMImageGenerator
from src.script_engine import Script, ScriptScene
from src.visual_engine import VisualEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports")


# ── Script de démonstration ──────────────────────────────────────────────────

def make_demo_script() -> Script:
    """Script de démonstration (5 scènes) pour comparer les deux générateurs."""
    scenes_data = [
        ("Hook", 8, "Et si l'IA remplacait deja votre monteur video sans que vous le sachiez ?"),
        ("Contexte", 10, "Voici ce qui vient de changer dans les studios de production."),
        ("Demonstration", 14, "Regardez cette interface generer un montage complet en quelques secondes."),
        ("Impact", 12, "Les monteurs professionnels doivent desormais s'adapter a cet outil."),
        ("Conclusion", 10, "Le montage video ne sera plus jamais comme avant."),
    ]
    scenes = [
        ScriptScene(
            order=i + 1, title=title,
            narration=narration,
            visual_description=f"Scene illustrant : {narration}",
            image_prompt=f"Image prompt heuristique pour {title}",
            animation_notes="Transition douce", sound_effects="Ambiance sonore",
            duration_seconds=dur,
        )
        for i, (title, dur, narration) in enumerate(scenes_data)
    ]
    return Script(
        title="L'IA qui remplace les monteurs video",
        hook=scenes_data[0][2],
        introduction=scenes_data[1][2],
        scenes=scenes,
        conclusion=scenes_data[-1][2],
        call_to_action="Quel outil de montage IA as-tu deja teste ? Dis-le en commentaire.",
        estimated_duration=sum(d for _, d, _ in scenes_data),
        language="fr",
        target_audience="Createurs de contenu et monteurs video",
        style="Innovant",
        metadata={"niche": "IA et production video"},
    )


# ── Scoring de richesse descriptive ─────────────────────────────────────────

_DIMENSION_KEYWORDS: Dict[str, List[str]] = {
    "composition": ["composition", "rule of thirds", "framing", "centered", "foreground", "background", "frame"],
    "camera": ["camera", "angle", "close-up", "close up", "wide shot", "low angle", "high angle", "lens", "shot"],
    "lighting": ["light", "lighting", "shadow", "glow", "backlit", "rim light", "illuminat"],
    "ambiance": ["mood", "atmosphere", "ambiance", "tension", "energetic", "calm", "dramatic", "cozy"],
    "color": ["color", "colour", "palette", "tone", "hue", "warm", "cool", "vibrant"],
    "depth": ["depth of field", "bokeh", "blur", "focus", "sharp focus", "shallow"],
    "cinematic_style": ["cinematic", "photorealistic", "realistic", "illustration", "style", "film"],
    "detail": ["detail", "8k", "4k", "ultra", "hd", "high resolution", "intricate"],
}


def score_prompt_richness(prompt: str) -> Dict[str, bool]:
    """Détecte par mots-clés la présence de chacune des 8 dimensions attendues."""
    text = prompt.lower()
    return {
        dim: any(re.search(re.escape(kw), text) for kw in keywords)
        for dim, keywords in _DIMENSION_KEYWORDS.items()
    }


def coverage_score(hits: Dict[str, bool]) -> int:
    return sum(1 for v in hits.values() if v)


# ── Rapport ──────────────────────────────────────────────────────────────────

def build_markdown_report(rows: List[Dict], provider: str) -> str:
    lines = [
        "# Benchmark — Prompt Heuristique vs Prompt LLM (Sprint 23)",
        "",
        "Comparaison de la richesse descriptive des prompts d'image sur 8 dimensions "
        "(composition, caméra, lumière, ambiance, couleurs, profondeur, style "
        "cinématographique, niveau de détail). Détection par mots-clés — indicateur "
        "de richesse descriptive, pas une note de qualité d'image réelle (aucun rendu).",
        "",
        "## Détail par scène",
        "",
        "| Scène | Heuristique /8 | LLM /8 | Longueur (H) | Longueur (LLM) | Negative prompt (LLM) |",
        "|-------|---------------:|-------:|-------------:|----------------:|:----------------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['scene_title']} | {r['heuristic_coverage']} | {r['llm_coverage']} "
            f"| {r['heuristic_len']} | {r['llm_len']} | {'Oui' if r['llm_has_negative'] else 'Non'} |"
        )

    avg_h = sum(r["heuristic_coverage"] for r in rows) / len(rows)
    avg_llm = sum(r["llm_coverage"] for r in rows) / len(rows)
    lines += [
        "",
        "## Moyennes",
        "",
        f"- **Heuristique** : {avg_h:.1f}/8 dimensions couvertes en moyenne",
        f"- **LLM ({provider})** : {avg_llm:.1f}/8 dimensions couvertes en moyenne",
        "",
        "## Détail des prompts",
        "",
    ]
    for r in rows:
        lines += [
            f"### {r['scene_title']}",
            "",
            f"**Heuristique** ({r['heuristic_coverage']}/8) :",
            f"> {r['heuristic_prompt']}",
            "",
            f"**LLM** ({r['llm_coverage']}/8) :",
            f"> {r['llm_prompt']}",
            "",
        ]
        if r["llm_has_negative"]:
            lines += [f"**Negative prompt (LLM)** : {r['llm_negative_prompt']}", ""]

    if avg_llm > avg_h:
        verdict = f"✅ Le LLM produit des prompts plus riches en moyenne (+{avg_llm - avg_h:.1f} dimension(s))."
    elif avg_llm < avg_h:
        verdict = f"⚠️ L'heuristique reste compétitive (-{avg_h - avg_llm:.1f} dimension(s) pour le LLM)."
    else:
        verdict = "📊 Égalité de richesse descriptive entre les deux approches."

    lines += ["## Conclusion", "", verdict, ""]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark Prompt Heuristique vs Prompt LLM (image)")
    parser.add_argument("--brand", type=str, default="ia_fr", help="Identifiant de la marque (défaut: ia_fr)")
    parser.add_argument("--provider", type=str, default="deepseek", help="Provider LLM (défaut: deepseek)")
    parser.add_argument("--llm-model", type=str, default=None, help="Modèle LLM spécifique")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print("  BENCHMARK IMAGE — Prompt Heuristique vs Prompt LLM (Sprint 23)")
    print("=" * 72)
    print(f"  Marque   : {args.brand}")
    print(f"  Provider : {args.provider}")
    print("=" * 72)
    print()

    be = BrandEngine()
    brand = be.load(args.brand)
    if brand is None:
        disponibles = [p.id for p in be.list()]
        logger.error("Marque '%s' introuvable. Disponibles : %s", args.brand, ", ".join(disponibles))
        sys.exit(1)

    script = make_demo_script()
    visual_plan = VisualEngine().generate(script)

    heuristic_gen = HeuristicImageGenerator()
    llm_gen = LLMImageGenerator(
        script=script, brand_profile=brand,
        provider_name=args.provider, model=args.llm_model, max_retries=1,
    )

    rows: List[Dict] = []
    for i, visual_scene in enumerate(visual_plan.scenes):
        script_scene = script.scenes[i]

        h_image: GeneratedImage = heuristic_gen.generate(visual_scene, visual_plan)
        if i > 0:
            time.sleep(1.0)
        llm_image = llm_gen.generate_from_scenes(script_scene, visual_scene, brand, script=script)

        # Contrat ImagePrompt (Sprint 24.1) : la richesse descriptive vit dans
        # subject + scene_description + style — "prompt" ne décrit plus que
        # l'action. On synthétise le texte complet pour le scoring comparatif.
        llm_full_text = f"{llm_image.subject}. {llm_image.scene_description} {llm_image.style}. {llm_image.prompt}"

        h_hits = score_prompt_richness(h_image.prompt)
        llm_hits = score_prompt_richness(llm_full_text)

        rows.append({
            "scene_title": script_scene.title,
            "heuristic_prompt": h_image.prompt,
            "llm_prompt": llm_full_text,
            "heuristic_coverage": coverage_score(h_hits),
            "llm_coverage": coverage_score(llm_hits),
            "heuristic_len": len(h_image.prompt),
            "llm_len": len(llm_full_text),
            "llm_has_negative": bool(llm_image.negative_prompt.strip()),
            "llm_negative_prompt": llm_image.negative_prompt,
        })

        logger.info(
            "[%s] Heuristique=%d/8 (%dc) | LLM=%d/8 (%dc)",
            script_scene.title, coverage_score(h_hits), len(h_image.prompt),
            coverage_score(llm_hits), len(llm_full_text),
        )

    markdown = build_markdown_report(rows, provider=args.provider)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"benchmark_image_prompts_{timestamp}.md"
    path.write_text(markdown, encoding="utf-8")
    logger.info("Rapport sauvegardé → %s", path)

    avg_h = sum(r["heuristic_coverage"] for r in rows) / len(rows)
    avg_llm = sum(r["llm_coverage"] for r in rows) / len(rows)
    print()
    print("=" * 72)
    print(f"  Heuristique : {avg_h:.1f}/8 dimensions en moyenne")
    print(f"  LLM ({args.provider}) : {avg_llm:.1f}/8 dimensions en moyenne")
    print(f"  Rapport     : {path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
