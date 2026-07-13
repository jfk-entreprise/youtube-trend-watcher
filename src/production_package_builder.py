"""
Production Package Builder — Sprint 28 (Studio de production autonome),
mis à jour Sprint 31.1 (Storyboard JSON + nettoyage des métadonnées techniques).

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
  - image_prompt.json / animation_prompt.json ne contiennent plus jamais
    provider/model/time_ms/cost_usd dans leur metadata — ces informations
    techniques ne vivent plus que dans report.md (tableau par scène).

Ne dépend d'aucun autre moteur créatif : il consomme uniquement les objets
déjà produits (Script, ImagePrompt, AnimationPrompt) via NicheProductionResult.
"""

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.brand_engine import BrandProfile
from src.niche_intelligence import Niche
from src.script_engine import Script

logger = logging.getLogger(__name__)

# Clés techniques qui ne doivent plus jamais apparaître dans un fichier de
# production (image_prompt.json / animation_prompt.json) — Sprint 31.1.
# Elles restent disponibles sur les objets Python en mémoire (pour le
# rapport) ; seule la SÉRIALISATION disque les retire.
_TECHNICAL_METADATA_KEYS = ("provider", "model", "time_ms", "cost_usd")


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


def _scene_description_paragraph(description) -> str:
    """
    Condense une scène à l'essentiel pour rester léger dans Storyboard
    Studio (Sprint 34.4) : uniquement le décor (setting) et les personnages
    à l'écran (characters) — composition/lighting/camera/mood/symbolism/
    director_notes/viewer_emotion restent dans final_script.json (contrat
    complet, Sprint 32.1) mais ne sont plus dupliqués ici, le script global
    étant jugé trop long pour un import direct.
    """
    return f"{description.setting} {description.characters}"


def _dialogue_label(personnage: str) -> str:
    """
    Sprint 34.1 — labels toujours en anglais (NARRATOR/CHARACTER), quel que
    soit le marché : seules les répliques elles-mêmes suivent la langue du
    script (français pour le marché FR, anglais pour le marché US).
    """
    name = (personnage or "").strip()
    if not name or name.upper() in ("NARRATEUR", "NARRATOR"):
        return "NARRATOR"
    return f"CHARACTER ({name})"


def _serialize_script_txt(script: Script) -> str:
    """
    Projette un Script sur un format texte brut compact pensé pour l'import
    dans Google Flow Storyboard Studio (Sprint 34.1, allégé aux Sprints
    34.4/34.5) : chaque scène tient sur 3 lignes consécutives (SCENE /
    NARRATOR ou CHARACTER / TRANSITION, sans saut de ligne interne, sans
    numérotation), séparées des autres scènes par une seule ligne vide (pas
    de séparateur "---") :

        TITLE : ...

        SCENE: <décor + personnages>
        NARRATOR: <replique>
        TRANSITION: <transition>

        SCENE: <décor + personnages>
        CHARACTER (Nom): <replique>
        TRANSITION: <transition>

    Seules les répliques (dialogues.replique) suivent la langue du script
    (script.language) — tout le reste (titre, description de scène,
    transition) est déjà généré en anglais par le LLM (voir
    llm_script_generator.py), quel que soit le marché ciblé.
    """
    scene_blocks: List[str] = []
    for scene in script.scenes:
        lines = [f"SCENE: {_scene_description_paragraph(scene.scene.description)}"]
        for dialogue in scene.dialogues:
            lines.append(f"{_dialogue_label(dialogue.personnage)}: {dialogue.replique}")
        lines.append(f"TRANSITION: {scene.transition}")
        scene_blocks.append("\n".join(lines))
    return f"TITLE : {script.title}\n\n" + "\n\n".join(scene_blocks)


def _strip_technical_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retire provider/model/time_ms/cost_usd de `data["metadata"]` avant
    écriture disque (Sprint 31.1) — ces informations techniques ne vivent
    plus que dans report.md.
    """
    data = dict(data)
    metadata = dict(data.get("metadata") or {})
    for key in _TECHNICAL_METADATA_KEYS:
        metadata.pop(key, None)
    data["metadata"] = metadata
    return data


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
        (package_dir / "script_final.txt").write_text(
            _serialize_script_txt(result.final_script), encoding="utf-8"
        )

        for entry in sorted(result.images, key=lambda e: e["scene_order"]):
            _write_json(
                image_dir / f"scene_{entry['scene_order']:02d}.json",
                _strip_technical_metadata(dataclasses.asdict(entry["image_prompt"])),
            )

        for entry in sorted(result.animations, key=lambda e: e["scene_order"]):
            _write_json(
                animation_dir / f"scene_{entry['scene_order']:02d}.json",
                _strip_technical_metadata(dataclasses.asdict(entry["animation_prompt"])),
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
