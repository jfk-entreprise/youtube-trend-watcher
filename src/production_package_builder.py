"""
Production Package Builder — Sprint 28 (Studio de production autonome).

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

        _write_json(package_dir / "final_script.json", dataclasses.asdict(result.final_script))

        for entry in sorted(result.images, key=lambda e: e["scene_order"]):
            _write_json(
                image_dir / f"scene_{entry['scene_order']:02d}.json",
                dataclasses.asdict(entry["image_prompt"]),
            )

        for entry in sorted(result.animations, key=lambda e: e["scene_order"]):
            _write_json(
                animation_dir / f"scene_{entry['scene_order']:02d}.json",
                dataclasses.asdict(entry["animation_prompt"]),
            )

        (package_dir / "report.md").write_text(self._build_report(result), encoding="utf-8")

        logger.info("Package de production créé : %s", package_dir)
        return package_dir

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
        ]
        return "\n".join(lines)
