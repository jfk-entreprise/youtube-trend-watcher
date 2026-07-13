"""
Production Package Builder — Sprint 28 (Studio de production autonome),
mis à jour Sprint 31.1 (Storyboard JSON + nettoyage des métadonnées techniques)
puis Sprint 34.6 (prompts image/vidéo au format riche "mega-prompt").

Construit, pour une niche/chaîne donnée, le package de production "propre"
attendu en sortie quotidienne du pipeline :

    outputs/YYYY-MM-DD/niche_01/
        final_script.json
        image_prompts/
        animation_prompts/
        report.md

Les dossiers techniques internes (shot_plans, .cache, benchmark.json) restent
écrits ailleurs par le pipeline (scripts/run_daily_pipeline.py) — ce module ne
les duplique jamais : seul ce qui est nécessaire à la production réelle de la
vidéo se retrouve dans niche_XX/.

Sprint 31.1 :
  - final_script.json adopte le format Storyboard Studio unifié
    (title + scenes[{order, scene, dialogues, transition, duration_seconds}])
    — aucun champ interne (metadata, language, style...) n'y est écrit.

Sprint 34.6 :
  - image_prompts/scene_XX.json et animation_prompts/scene_XX.json adoptent un
    format "mega-prompt" à 3 clés {prompt, negative_prompt, instruction_format}
    — le champ "prompt" concatène des libellés riches ("Subject: ... Clothing:
    ... Camera Angle: ...") construits à partir du contenu déjà généré
    (ImagePrompt/AnimationPrompt, ShotPlan, SceneDescription, BrandProfile) —
    aucune nouvelle génération LLM ici, uniquement une reformulation.

Ne dépend d'aucun autre moteur créatif : il consomme uniquement les objets
déjà produits (Script, ImagePrompt, AnimationPrompt, ShotPlan) via
NicheProductionResult.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.brand_engine import BrandProfile
from src.niche_intelligence import Niche
from src.script_engine import Script

logger = logging.getLogger(__name__)

_INSTRUCTION_FORMAT = "Respond STRICTLY in valid JSON. Do not include any explanation or markdown."


# ── Contrat d'entrée ──────────────────────────────────────────────────────────

@dataclass
class NicheProductionResult:
    """
    Résultat complet de production pour UNE niche/chaîne (un des deux
    vidéos/jour). Regroupe ce qui est nécessaire au package final.
    """
    niche: Niche
    brand: BrandProfile
    final_script: Script
    images: List[Dict[str, Any]]        # [{"scene_order": int, "image_prompt": ImagePrompt, ...}]
    animations: List[Dict[str, Any]]    # [{"scene_order": int, "animation_prompt": AnimationPrompt}]
    rewrite_result: Optional[Dict[str, Any]] = None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _serialize_script(script: Script) -> Dict[str, Any]:
    """
    Projette un Script sur le contrat storyboard cinématographique (Sprint 32.1) :
    UNIQUEMENT {title, scenes[{scene: {number, type, description{9 champs}},
    dialogues, transition, duration_seconds}]}. Les champs internes du
    pipeline (metadata, language, style, target_audience, estimated_duration)
    ne font pas partie du contrat de production — ils restent sur l'objet
    Python pour les besoins internes (évaluateur, rapport) mais ne sont
    jamais écrits sur disque ici.
    """
    return {
        "title": script.title,
        "scenes": [
            {
                "scene": {
                    "number": scene.scene.number,
                    "type": scene.scene.type,
                    "description": {
                        "setting": scene.scene.description.setting,
                        "composition": scene.scene.description.composition,
                        "characters": scene.scene.description.characters,
                        "lighting": scene.scene.description.lighting,
                        "camera": scene.scene.description.camera,
                        "mood": scene.scene.description.mood,
                        "symbolism": scene.scene.description.symbolism,
                        "director_notes": scene.scene.description.director_notes,
                        "viewer_emotion": scene.scene.description.viewer_emotion,
                    },
                },
                "dialogues": [
                    {"personnage": d.personnage, "replique": d.replique}
                    for d in scene.dialogues
                ],
                "transition": scene.transition,
                "duration_seconds": scene.duration_seconds,
            }
            for scene in script.scenes
        ],
    }


def _s(value: Any, default: str = "unspecified") -> str:
    """Normalise une valeur texte potentiellement absente/vide."""
    text = str(value).strip() if value is not None else ""
    return text or default


def _build_image_prompt_file(
    image_prompt: Any, shot_plan: Optional[Any], description: Any, brand: BrandProfile,
) -> Dict[str, Any]:
    """
    Construit le fichier image_prompts/scene_XX.json (Sprint 34.6) : un
    "mega-prompt" texte unique regroupant des libellés riches, à partir du
    contenu déjà généré par LLMImageGenerator (ImagePrompt), le VisualDirector
    (ShotPlan, si disponible) et l'identité de marque (BrandProfile) — aucune
    nouvelle génération ici, uniquement une reformulation.
    """
    meta = image_prompt.metadata or {}
    camera_angle = shot_plan.camera_angle if shot_plan else description.camera
    lens = shot_plan.lens if shot_plan else "unspecified"
    composition = shot_plan.composition if shot_plan else description.composition
    color_palette = shot_plan.color_palette if shot_plan else ", ".join(brand.color_palette)
    details = " ".join(
        part for part in (description.symbolism, description.director_notes) if part
    )

    fields = [
        ("Subject", image_prompt.subject),
        ("Appearance", meta.get("appearance")),
        ("Clothing", meta.get("clothing")),
        ("Accessories", meta.get("accessories")),
        ("Pose", meta.get("pose")),
        ("Action", image_prompt.prompt),
        ("Facial Expression", meta.get("facial_expression")),
        ("Emotion", meta.get("emotion")),
        ("Environment", description.setting),
        ("Background", meta.get("background")),
        ("Weather", meta.get("weather")),
        ("Time of Day", meta.get("time_of_day")),
        ("Lighting", description.lighting),
        ("Camera Angle", camera_angle),
        ("Lens", lens),
        ("Composition", composition),
        ("Style", image_prompt.style),
        ("Color Palette", color_palette),
        ("Details", details),
        ("Text (optional)", "None"),
        ("Language", "None (no on-screen text)"),
    ]
    prompt = " ".join(f"{label}: {_s(value)}." for label, value in fields)

    return {
        "prompt": prompt,
        "negative_prompt": image_prompt.negative_prompt,
        "instruction_format": _INSTRUCTION_FORMAT,
    }


def _dialogue_fields(dialogues: List[Any]) -> Dict[str, str]:
    """Dérive Dialogue/Speaker/Narration (verbatim) depuis les répliques de la scène."""
    if not dialogues:
        return {"dialogue": "None", "speaker": "None", "narration": "None"}
    speakers = [
        "NARRATOR" if not d.personnage.strip() or d.personnage.strip().upper() in ("NARRATEUR", "NARRATOR")
        else d.personnage.strip()
        for d in dialogues
    ]
    lines = [d.replique for d in dialogues]
    narration = " ".join(
        d.replique for d in dialogues if (d.personnage or "").strip().upper() in ("", "NARRATEUR", "NARRATOR")
    )
    return {
        "dialogue": " / ".join(lines),
        "speaker": " / ".join(speakers),
        "narration": narration or "None",
    }


def _build_animation_prompt_file(
    animation_prompt: Any, image_prompt: Any, shot_plan: Optional[Any], description: Any, language: str,
) -> Dict[str, Any]:
    """
    Construit le fichier animation_prompts/scene_XX.json (Sprint 34.6) — même
    principe que _build_image_prompt_file, en réutilisant en plus l'ImagePrompt
    de la même scène (apparence/vêtements déjà établis) et l'AnimationPrompt
    (mouvement/son/transition déjà générés) : aucune nouvelle génération ici.
    """
    meta = animation_prompt.metadata or {}
    img_meta = image_prompt.metadata or {}
    camera_angle = shot_plan.camera_angle if shot_plan else description.camera
    lens = shot_plan.lens if shot_plan else "unspecified"
    composition = shot_plan.composition if shot_plan else description.composition
    shot_type = shot_plan.shot_type if shot_plan else "unspecified"
    dialogue_fields = _dialogue_fields(animation_prompt.dialogues)

    fields = [
        ("Subject", image_prompt.subject),
        ("Appearance", img_meta.get("appearance")),
        ("Clothing", img_meta.get("clothing")),
        ("Accessories", img_meta.get("accessories")),
        ("Initial Pose", img_meta.get("pose")),
        ("Character Action", animation_prompt.subject_motion),
        ("Secondary Actions", animation_prompt.environment_motion),
        ("Facial Expression", img_meta.get("facial_expression")),
        ("Emotion", meta.get("emotion")),
        ("Environment", description.setting),
        ("Background", img_meta.get("background")),
        ("Weather", img_meta.get("weather")),
        ("Time of Day", img_meta.get("time_of_day")),
        ("Lighting", animation_prompt.lighting_changes),
        ("Camera Shot", shot_type),
        ("Camera Angle", camera_angle),
        ("Camera Movement", animation_prompt.camera_motion),
        ("Lens", lens),
        ("Composition", composition),
        ("Visual Style", image_prompt.style),
        ("Animation Style", meta.get("animation_style")),
        ("Scene Duration", f"{animation_prompt.duration}s"),
        ("Frame Rate", "24 fps"),
        ("Dialogue", dialogue_fields["dialogue"]),
        ("Speaker", dialogue_fields["speaker"]),
        ("Narration", dialogue_fields["narration"]),
        ("Language", language),
        ("Voice", meta.get("voice")),
        ("Lip Sync", "Synced to spoken dialogue audio"),
        ("Sound Effects", meta.get("sound_effects")),
        ("Ambient Sounds", animation_prompt.sound_design),
        ("Background Music", meta.get("background_music")),
        ("Atmosphere", description.mood),
        ("Ending Scene", animation_prompt.transition),
    ]
    prompt = " ".join(f"{label}: {_s(value)}." for label, value in fields)

    return {
        "prompt": prompt,
        "negative_prompt": image_prompt.negative_prompt,
        "instruction_format": _INSTRUCTION_FORMAT,
    }


# ── ProductionPackageBuilder ─────────────────────────────────────────────────

class ProductionPackageBuilder:
    """
    Exemple minimal :
        builder = ProductionPackageBuilder()
        package_dir = builder.build(output_dir, niche_index=1, result=niche_result)
    """

    def build(self, output_dir: Path, niche_index: int, result: NicheProductionResult) -> Path:
        package_dir = Path(output_dir) / f"niche_{niche_index:02d}"
        image_dir = package_dir / "image_prompts"
        animation_dir = package_dir / "animation_prompts"
        package_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        animation_dir.mkdir(parents=True, exist_ok=True)

        _write_json(package_dir / "final_script.json", _serialize_script(result.final_script))

        scenes_by_number = {s.scene.number: s for s in result.final_script.scenes}
        images_by_order = {e["scene_order"]: e for e in result.images}
        language = result.final_script.language

        for entry in sorted(result.images, key=lambda e: e["scene_order"]):
            script_scene = scenes_by_number.get(entry["scene_order"])
            description = script_scene.scene.description if script_scene else None
            _write_json(
                image_dir / f"scene_{entry['scene_order']:02d}.json",
                _build_image_prompt_file(
                    entry["image_prompt"], entry.get("shot_plan"), description, result.brand,
                ),
            )

        for entry in sorted(result.animations, key=lambda e: e["scene_order"]):
            script_scene = scenes_by_number.get(entry["scene_order"])
            description = script_scene.scene.description if script_scene else None
            image_entry = images_by_order.get(entry["scene_order"], {})
            _write_json(
                animation_dir / f"scene_{entry['scene_order']:02d}.json",
                _build_animation_prompt_file(
                    entry["animation_prompt"], image_entry.get("image_prompt"),
                    image_entry.get("shot_plan"), description, language,
                ),
            )

        (package_dir / "report.md").write_text(self._build_report(result), encoding="utf-8")

        logger.info("Package de production créé : %s", package_dir)
        return package_dir

    @staticmethod
    def _scene_metrics_cells(metadata: Optional[Dict[str, Any]]) -> tuple:
        """Extrait (provider, statut, temps, coût) depuis un metadata NON strippé."""
        if not metadata:
            return ("—", "—", "—", "—")
        provider = str(metadata.get("provider", "—"))
        fallback_reason = metadata.get("fallback_reason")
        status = f"fallback ({fallback_reason})" if fallback_reason else "LLM"
        time_ms = metadata.get("time_ms", 0)
        cost = metadata.get("cost_usd", 0.0)
        return (provider, status, f"{time_ms} ms", f"${cost:.6f}")

    @staticmethod
    def _build_report(result: NicheProductionResult) -> str:
        script = result.final_script
        lines = [
            f"# Package de production — {result.niche.name}",
            "",
            f"**Chaîne :** {result.brand.name} ({result.brand.id})  ",
            f"**Niche :** {result.niche.name} (score={result.niche.niche_score:.3f})  ",
            f"**Titre :** {script.title}  ",
            f"**Hook :** {script.hook}  ",
            f"**Durée estimée :** {script.estimated_duration}s  ",
            f"**Scènes :** {len(script.scenes)}  ",
        ]
        if result.rewrite_result is not None:
            applied = result.rewrite_result.get("rewrite_applied")
            lines.append(f"**Réécriture :** {'appliquée' if applied else 'non appliquée'}  ")
        lines += [
            "",
            f"- Prompts image générés : {len(result.images)}",
            f"- Prompts animation générés : {len(result.animations)}",
            "",
            "## Métriques techniques par scène",
            "",
            "Source unique des informations techniques (provider, modèle, temps, "
            "coût, statut, fallback) — ces champs n'apparaissent plus dans "
            "`image_prompts/*.json` ni `animation_prompts/*.json` (Sprint 31.1).",
            "",
            "| Scène | Image — provider | Image — statut | Image — temps | Image — coût "
            "| Animation — provider | Animation — statut | Animation — temps | Animation — coût |",
            "|---|---|---|---|---|---|---|---|---|",
        ]

        images_by_order = {e["scene_order"]: e["image_prompt"] for e in result.images}
        animations_by_order = {e["scene_order"]: e["animation_prompt"] for e in result.animations}
        all_orders = sorted(set(images_by_order) | set(animations_by_order))

        for order in all_orders:
            img = images_by_order.get(order)
            anim = animations_by_order.get(order)
            img_cells = ProductionPackageBuilder._scene_metrics_cells(
                img.metadata if img is not None else None
            )
            anim_cells = ProductionPackageBuilder._scene_metrics_cells(
                anim.metadata if anim is not None else None
            )
            lines.append(
                f"| {order} | {img_cells[0]} | {img_cells[1]} | {img_cells[2]} | {img_cells[3]} "
                f"| {anim_cells[0]} | {anim_cells[1]} | {anim_cells[2]} | {anim_cells[3]} |"
            )

        lines.append("")
        return "\n".join(lines)
