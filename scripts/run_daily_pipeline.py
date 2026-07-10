#!/usr/bin/env python3
"""
Daily Pipeline — Studio de production autonome (Sprint 28).

Orchestre la chaîne complète Collecte → Analyse → Décision → Création en
8 étapes, en composant les moteurs existants SANS EN MODIFIER AUCUN, pour
produire exactement 2 vidéos/jour (une par niche/chaîne active) :

    1.  Chargement des données (Supabase en source principale, CsvStorage
        uniquement en fallback — via build_storage()).
    2.  Construction de la KnowledgeBase (ContentUnderstandingEngine + KnowledgeEngine).
    3.  Détection des niches candidates (NicheAnalyzer) puis sélection des
        niches ACTIVES du jour (NicheSelector, persistance `active_niches`) —
        une niche n'est remplacée que par une candidate significativement
        meilleure (voir src/niche_selector.py).
    4.  Sélection des meilleures opportunités par niche active (max 3/niche).
    5.  Production complète par niche (répétée une fois par niche active,
        2 fois/jour au maximum) : CreativeBrief → scripts LLM DeepSeek →
        évaluation → réécriture optionnelle → VisualPlan → ShotPlan
        (VisualDirector) → ImagePrompt → AnimationPrompt. La chaîne (BrandProfile)
        de chaque niche est déterminée via BrandProfile.niche_keywords
        (select_brand_for_niche), avec repli sur --brand si aucun mapping.
    6.  Construction du package de production propre par niche
        (outputs/YYYY-MM-DD/niche_XX/{final_script.json, image_prompts/,
        animation_prompts/, report.md} — ProductionPackageBuilder) + sauvegarde
        des sorties techniques internes (shot_plans, scripts intermédiaires,
        benchmark.json, rapport.md) séparément.
    7.  Envoi vers Supabase Storage (StorageUploader, Sprint 30 — remplace
        Google Drive, qui n'a aucun quota de stockage utilisable par un
        compte de service) — upload réel si SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY
        sont configurés (SupabaseStorageUploader), sinon NoOpStorageUploader
        (aucune régression). Échec d'upload capturé : n'interrompt jamais le
        pipeline.
    8.  Notification du résumé quotidien (NotificationService) — envoi réel
        via Telegram si TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID sont configurés
        (TelegramNotificationService, avec lien direct vers le package
        Supabase Storage de l'étape 7 quand l'upload a réussi), sinon
        LoggingNotificationService journalise le résumé formaté. Échec
        réseau/API capturé : n'interrompt jamais le pipeline.

Le ShotPlan (VisualDirector) est la source de vérité du cadrage : ce script
d'orchestration reconstruit, PAR COMPOSITION, une VisualScene « dirigée »
(shot_type/composition/lighting/color_palette remplacés par les décisions du
ShotPlan, détails additionnels — angle, focale, profondeur de champ, focal
point, priorité visuelle, moment miniature — glissés dans VisualScene.metadata)
avant de l'envoyer à LLMImageGenerator puis LLMAnimationGenerator. Aucun des
deux moteurs n'est modifié : ils continuent de lire les mêmes champs de
VisualScene qu'avant, seul leur CONTENU change.

Ce script est UNIQUEMENT un orchestrateur : aucune logique métier n'est
dupliquée ou modifiée ici — chaque étape délègue à un moteur existant.

Arrêt propre : toute étape en échec interrompt le pipeline avec un message
clair (pas de traceback brut) et un code de sortie non nul ; les résultats
déjà produits par les étapes précédentes restent sur le disque.

Usage :
    python scripts/run_daily_pipeline.py
    python scripts/run_daily_pipeline.py --brand histoire_fr --provider deepseek --top 40
    python scripts/run_daily_pipeline.py --llm-judge --rewrite
    python scripts/run_daily_pipeline.py --output-dir /mnt/data/outputs
"""

import argparse
import dataclasses
import json
import logging
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.brand_engine import BrandEngine, BrandProfile
from src.content_understanding import ContentUnderstandingEngine
from src.creative_engine import CreativeBrief, CreativeEngine
from src.knowledge_engine import KnowledgeEngine
from src.llm_animation_generator import LLMAnimationGenerator
from src.llm_image_generator import LLMImageGenerator
from src.llm_script_evaluator import LLMScriptEvaluator, LLMScriptScore
from src.llm_script_generator import LLMScriptGenerator
from src.niche_intelligence import Niche, NicheAnalyzer
from src.niche_selector import NicheSelector
from src.opportunity_engine import Opportunity, OpportunityEngine
from src.production_package_builder import NicheProductionResult, ProductionPackageBuilder
from src.supabase_storage_uploader import UploadResult, build_storage_uploader
from src.notification_service import (
    ChannelSummary,
    DailyProductionSummary,
    NotificationResult,
    _format_duration,
    build_notification_service,
)
from src.rewrite_engine import RewriteEngine
from src.script_engine import Script
from src.script_evaluator import ScriptEvaluator
from src.storage import CsvStorage, build_storage
from src.utils import csv_snapshots_to_timelines
from src.virality_engine import VideoTimeline
from src.visual_director import ShotPlan, VisualDirector
from src.visual_engine import VisualEngine, VisualScene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("daily_pipeline")


# ── Règles fixes du pipeline (Sprint 25 — pas configurables en CLI) ─────────

MAX_NICHES = 2
MAX_OPPORTUNITIES_PER_NICHE = 3
MAX_SCRIPTS_PER_OPPORTUNITY = 2  # nombre de CreativeBrief transformés en script LLM

TOTAL_STEPS = 8
DEFAULT_CSV_FALLBACK = Path("data/videos.csv")

# Identité des générateurs de secours (Sprint 26/29.1) — sert à distinguer
# LLM vs fallback heuristique dans le résumé de fin de run (Issue 5).
_IMAGE_FALLBACK_PROVIDER = "heuristic_image_v1"
_ANIMATION_FALLBACK_PROVIDER = "fallback_heuristic"


# ── Arrêt propre ─────────────────────────────────────────────────────────────

class PipelineStepError(Exception):
    """Levée quand une étape du pipeline échoue — provoque un arrêt propre."""


def run_step(num: int, name: str, fn, *args, **kwargs):
    """Exécute une étape avec logs clairs et conversion des erreurs en arrêt propre."""
    logger.info("=" * 76)
    logger.info("[%d/%d] %s", num, TOTAL_STEPS, name)
    logger.info("=" * 76)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("ÉCHEC étape [%d/%d] %s après %.1fs : %s", num, TOTAL_STEPS, name, elapsed, exc)
        raise PipelineStepError(f"Étape {num} ({name}) en échec : {exc}") from exc
    elapsed = time.time() - t0
    logger.info("OK [%d/%d] %s terminée en %.1fs", num, TOTAL_STEPS, name, elapsed)
    return result


# ── Sérialisation JSON ───────────────────────────────────────────────────────

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _asdict(obj: Any) -> Dict[str, Any]:
    return dataclasses.asdict(obj)


def _niche_summary(niche: Niche) -> Dict[str, Any]:
    """Résumé JSON-safe d'une Niche (exclut `timelines`, non sérialisable)."""
    return {
        "name": niche.name,
        "volume": niche.volume,
        "avg_views": niche.avg_views,
        "avg_engagement": niche.avg_engagement,
        "avg_growth_speed": niche.avg_growth_speed,
        "niche_score": niche.niche_score,
    }


# ── Étape 1 : Chargement des données ────────────────────────────────────────

def step_load_data(csv_fallback: Path, cache_dir: Path) -> tuple[list[VideoTimeline], Path]:
    """
    Charge les VideoSnapshot via build_storage() (Supabase en source
    principale, CsvStorage uniquement en fallback), puis les matérialise
    dans un CSV de travail pour les moteurs qui exigent un `csv_path`
    (ViralityEngine, NicheAnalyzer) — sans jamais toucher au CSV de
    collecte partagé (data/videos.csv) ni modifier ces moteurs.
    """
    storage = build_storage(csv_fallback)
    snapshots = storage.load()
    if not snapshots:
        raise RuntimeError(
            "Aucun snapshot chargé (Supabase et CSV fallback vides ou inaccessibles)."
        )
    logger.info("%d snapshots chargés (Supabase prioritaire, CSV en repli).", len(snapshots))

    mirror_path = cache_dir / "snapshots_mirror.csv"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if mirror_path.exists():
        mirror_path.unlink()
    CsvStorage(mirror_path).save(snapshots)

    buckets = csv_snapshots_to_timelines(mirror_path)
    timelines = [VideoTimeline(vid_id, snaps) for vid_id, snaps in buckets.items()]
    logger.info("%d timelines vidéo reconstituées.", len(timelines))
    return timelines, mirror_path


# ── Étape 2 : KnowledgeBase ──────────────────────────────────────────────────

def step_build_knowledge(timelines):
    cue = ContentUnderstandingEngine()
    profiles = cue.analyze_all(timelines)
    kb = KnowledgeEngine().build(profiles)
    logger.info("%d ContentProfile → %d sujets, %d combinaisons connues.",
                len(profiles), len(kb.topics), len(kb.combinations))
    return profiles, kb


# ── Étape 3 : Détection des niches candidates + sélection des niches actives ─

def step_detect_niches(mirror_csv: Path) -> List[Niche]:
    """
    Retourne TOUTES les niches candidates (triées par score décroissant) —
    aucune troncature ici. C'est NicheSelector (step_select_active_niches)
    qui décide, à partir de ce classement, des niches réellement actives
    aujourd'hui (règles business : conservation vs remplacement significatif).
    """
    analyzer = NicheAnalyzer(csv_path=mirror_csv)
    all_niches = analyzer.analyze()
    if not all_niches:
        raise RuntimeError("Aucune niche détectée à partir des données chargées.")
    for n in all_niches[:10]:
        logger.info("  Niche candidate : %-30s score=%.3f volume=%d", n.name, n.niche_score, n.volume)
    return all_niches


def step_select_active_niches(
    mirror_csv: Path, selector: NicheSelector
) -> tuple[List[Niche], List[Niche]]:
    """
    Détecte les candidates puis applique NicheSelector pour obtenir les
    niches actives du jour (persistées dans `active_niches`, Sprint 28).
    """
    candidates = step_detect_niches(mirror_csv)
    selected = selector.select_daily_niches(candidates)
    return candidates, selected


# ── Étape 4 : Sélection des meilleures opportunités ─────────────────────────

def step_select_opportunities(
    profiles, timelines, kb, niches: List[Niche], top_n: int
) -> Dict[str, List[Opportunity]]:
    """
    Calcule les opportunités sur l'ensemble du dataset (contexte de scoring
    complet et cohérent), puis les répartit par niche via le video_id
    (NicheAnalyzer regroupe par mot-clé, OpportunityEngine étiquette par
    sujet sémantique — les deux taxonomies diffèrent ; le video_id est la
    seule clé fiable pour les faire correspondre).

    Retourne au maximum MAX_OPPORTUNITIES_PER_NICHE opportunités par niche,
    pour au maximum MAX_NICHES niches.
    """
    oe = OpportunityEngine()
    all_opportunities = oe.build(profiles, timelines, kb, top_n=max(top_n, 1))
    logger.info("%d opportunités calculées sur l'ensemble du dataset.", len(all_opportunities))

    by_video_id = {opp.source_video_id: opp for opp in all_opportunities}

    selected: Dict[str, List[Opportunity]] = {}
    for niche in niches:
        niche_video_ids = {tl.video_id for tl in niche.timelines}
        niche_opps = [
            opp for opp in all_opportunities
            if opp.source_video_id in niche_video_ids
        ][:MAX_OPPORTUNITIES_PER_NICHE]
        selected[niche.name] = niche_opps
        logger.info("  Niche '%s' → %d opportunité(s) retenue(s).", niche.name, len(niche_opps))

    total = sum(len(v) for v in selected.values())
    if total == 0:
        raise RuntimeError(
            "Aucune opportunité ne correspond aux niches sélectionnées "
            f"(sur {len(by_video_id)} opportunités calculées)."
        )
    return selected


# ── Étape 5 : CreativeBrief ──────────────────────────────────────────────────

def step_generate_briefs(opportunities: List[Opportunity]) -> Dict[str, List[CreativeBrief]]:
    briefs_map = CreativeEngine().generate_all(opportunities)
    total = sum(len(b) for b in briefs_map.values())
    logger.info("%d CreativeBrief générés pour %d opportunité(s).", total, len(opportunities))
    return briefs_map


# ── Étape 6 : Scripts LLM (DeepSeek) ─────────────────────────────────────────

def step_generate_scripts(
    opportunities: List[Opportunity],
    briefs_map: Dict[str, List[CreativeBrief]],
    brand: BrandProfile,
    provider: str,
) -> List[Dict[str, Any]]:
    """
    Génère plusieurs scripts LLM par opportunité (un par CreativeBrief,
    plafonné à MAX_SCRIPTS_PER_OPPORTUNITY). LLMScriptGenerator gère déjà
    lui-même son fallback heuristique en cas d'échec LLM — réutilisé tel quel.
    """
    generator = LLMScriptGenerator(provider_name=provider, max_retries=1)
    entries: List[Dict[str, Any]] = []

    for opp in opportunities:
        briefs = briefs_map.get(opp.source_video_id, [])[:MAX_SCRIPTS_PER_OPPORTUNITY]
        if not briefs:
            logger.warning("  Aucun brief pour '%s' — opportunité ignorée.", opp.title[:50])
            continue
        for brief in briefs:
            script = generator.generate(opp, brief, brand)
            is_fallback = script.metadata.get("generator") != "llm_v1"
            logger.info(
                "  [%s / %s] %s — %d scènes, %ds%s",
                opp.niche, brief.angle, script.title[:45], len(script.scenes),
                script.estimated_duration, " (fallback heuristique)" if is_fallback else "",
            )
            entries.append({
                "opportunity": opp,
                "brief": brief,
                "script": script,
            })

    if not entries:
        raise RuntimeError("Aucun script généré.")
    logger.info("%d script(s) générés au total (stats LLM : %s).", len(entries), generator.stats)
    return entries


# ── Étape 7 : Évaluation ─────────────────────────────────────────────────────

def step_evaluate_scripts(
    entries: List[Dict[str, Any]], provider: str, use_llm_judge: bool
) -> None:
    """Ajoute 'heuristic_score' (toujours) et 'llm_judge_score' (si activé) à chaque entrée."""
    heuristic_eval = ScriptEvaluator()
    llm_judge = LLMScriptEvaluator(provider_name=provider, max_retries=1) if use_llm_judge else None

    for entry in entries:
        score = heuristic_eval.evaluate(entry["script"])
        entry["heuristic_score"] = score
        entry["llm_judge_score"] = None
        logger.info("  Heuristique : %.1f/80 — %s", score.composite_score, entry["script"].title[:45])

        if llm_judge is not None:
            try:
                judge_score = llm_judge.evaluate(entry["script"])
                entry["llm_judge_score"] = judge_score
                logger.info("  LLM-judge   : %.1f/80 — %s", judge_score.global_score, entry["script"].title[:45])
            except Exception as exc:
                logger.warning("  LLM-judge indisponible pour '%s' : %s", entry["script"].title[:45], exc)


def pick_best_entry(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Le meilleur script du run = le plus haut score heuristique composite."""
    return max(entries, key=lambda e: e["heuristic_score"].composite_score)


# ── Étape 8+9 : Réécriture + réévaluation (RewriteEngine) ───────────────────

def step_rewrite_best_script(
    best_entry: Dict[str, Any], provider: str
) -> Dict[str, Any]:
    """
    RewriteEngine.rewrite() effectue en une seule passe la réécriture (étape 8)
    ET la réévaluation immédiate qui décide de garder ou non la nouvelle
    version (étape 9) — c'est le contrat public du moteur (Sprint 22),
    volontairement non modifié ici.
    """
    judge_score: Optional[LLMScriptScore] = best_entry.get("llm_judge_score")
    if judge_score is None:
        logger.info("  Pas de score LLM-judge disponible pour le meilleur script — évaluation dédiée.")
        judge_score = LLMScriptEvaluator(provider_name=provider, max_retries=1).evaluate(best_entry["script"])
        best_entry["llm_judge_score"] = judge_score

    engine = RewriteEngine(
        evaluator=LLMScriptEvaluator(provider_name=provider, max_retries=1),
        provider_name=provider,
        max_retries=1,
    )
    rewritten = engine.rewrite(best_entry["script"], judge_score)
    applied = bool(rewritten.metadata.get("rewritten"))

    if applied:
        logger.info(
            "  Réécriture retenue : score LLM-judge %.1f → %.1f/80",
            rewritten.metadata["rewrite_score_before"], rewritten.metadata["rewrite_score_after"],
        )
    else:
        logger.info("  Réécriture non retenue (score non amélioré) — script original conservé.")

    return {
        "final_script": rewritten,
        "rewrite_applied": applied,
        "rewrite_stats": engine.stats,
        "score_before": judge_score.global_score,
        "score_after": rewritten.metadata.get("rewrite_score_after"),
    }


# ── Étape 10 : VisualPlan ────────────────────────────────────────────────────

def step_generate_visual_plan(script: Script):
    plan = VisualEngine().generate(script)
    logger.info("VisualPlan '%s' — %d scènes, style=%s, ratio=%s",
                plan.title[:45], len(plan.scenes), plan.style, plan.aspect_ratio)
    return plan


# ── Étape 11 : ShotPlan — décisions de réalisation (VisualDirector, Sprint 26) ─

def step_generate_shot_plans(
    script: Script, brand: BrandProfile, provider: str
) -> Dict[int, ShotPlan]:
    """
    Génère un ShotPlan (contrat Sprint 26) par scène, dans l'ordre du script,
    avec une seule instance de VisualDirector pour que les décisions de
    cadrage évoluent naturellement d'une scène à l'autre (historique interne
    des ShotPlan déjà établis).
    """
    director = VisualDirector(script=script, brand_profile=brand, provider_name=provider, max_retries=1)
    shot_plans: Dict[int, ShotPlan] = {}

    for scene in sorted(script.scenes, key=lambda s: s.order):
        shot_plan = director.generate_shot_plan(scene, brand, script=script)
        shot_plans[scene.order] = shot_plan
        provider_used = shot_plan.metadata.get("provider", "")
        logger.info(
            "  Scène #%d : %s / %s / %s / %s — focal_point=%s (%s)",
            scene.order, shot_plan.shot_type, shot_plan.camera_angle, shot_plan.lens,
            shot_plan.lighting_style, shot_plan.focal_point,
            "LLM" if provider_used != "fallback_heuristic" else "fallback heuristique",
        )

    logger.info("%d ShotPlan généré(s) (stats : %s).", len(shot_plans), director.stats)
    return shot_plans


def _apply_shot_plan_to_visual_scene(visual_scene: VisualScene, shot_plan: Optional[ShotPlan]) -> VisualScene:
    """
    Applique le ShotPlan à une VisualScene — le ShotPlan devient la source de
    vérité du cadrage (Sprint 26) : shot_type/composition/lighting/color_palette
    sont remplacés par ses décisions. Les informations additionnelles (angle,
    focale, profondeur de champ, focal point, priorité visuelle, moment
    miniature) sont glissées dans VisualScene.metadata, un champ déjà prévu à
    cet effet — LLMImageGenerator et LLMAnimationGenerator ne sont PAS
    modifiés : ils continuent de lire les mêmes champs de VisualScene qu'avant,
    seul leur contenu change.
    """
    if shot_plan is None:
        return visual_scene
    return dataclasses.replace(
        visual_scene,
        shot_type=shot_plan.shot_type,
        composition=shot_plan.composition,
        lighting=shot_plan.lighting_style,
        color_palette=[shot_plan.color_palette],
        metadata={
            **visual_scene.metadata,
            "camera_angle": shot_plan.camera_angle,
            "lens": shot_plan.lens,
            "depth_of_field": shot_plan.depth_of_field,
            "focal_point": shot_plan.focal_point,
            "visual_priority": shot_plan.visual_priority,
            "thumbnail_moment": shot_plan.thumbnail_moment,
            "cinematic_goal": shot_plan.cinematic_goal,
        },
    )


# ── Étape 12 : Prompts image Nano Banana (dirigés par le ShotPlan) ──────────

def step_generate_image_prompts(
    script: Script, visual_plan, shot_plans: Dict[int, ShotPlan], brand: BrandProfile, provider: str
) -> List[Dict[str, Any]]:
    """
    Génère un ImagePrompt (contrat Sprint 24.1) par scène, dans l'ordre du
    script, avec une seule instance de LLMImageGenerator pour que la bible
    de personnages (Sprint 24.3) assure la cohérence visuelle d'une scène à
    l'autre. Chaque VisualScene est d'abord « dirigée » par le ShotPlan
    correspondant (Sprint 26) avant d'être transmise au générateur.
    """
    generator = LLMImageGenerator(script=script, brand_profile=brand, provider_name=provider, max_retries=1)
    entries: List[Dict[str, Any]] = []
    ordered_scenes = sorted(visual_plan.scenes, key=lambda s: s.scene_order)

    for visual_scene in ordered_scenes:
        script_scene = next((s for s in script.scenes if s.order == visual_scene.scene_order), None)
        if script_scene is None:
            logger.warning("  Aucune ScriptScene pour la scène visuelle #%d — ignorée.", visual_scene.scene_order)
            continue
        shot_plan = shot_plans.get(visual_scene.scene_order)
        directed_scene = _apply_shot_plan_to_visual_scene(visual_scene, shot_plan)
        image_prompt = generator.generate_from_scenes(script_scene, directed_scene, brand, script=script)
        entries.append({"scene_order": visual_scene.scene_order, "image_prompt": image_prompt, "shot_plan": shot_plan})
        provider_used = image_prompt.metadata.get("provider", "")
        logger.info("  Scène #%d : provider=%s (%s)", visual_scene.scene_order, provider_used,
                    "LLM" if provider_used != "heuristic_image_v1" else "fallback heuristique")

    if not entries:
        raise RuntimeError("Aucun prompt d'image généré.")
    logger.info("%d prompt(s) d'image généré(s) (stats : %s, personnages : %s).",
                len(entries), generator.stats, list(generator.characters_bible.keys()))
    return entries


# ── Étape 13 : Prompts d'animation (Veo/Kling/Runway, dirigés par le ShotPlan) ─

def step_generate_animation_prompts(
    script: Script, visual_plan, shot_plans: Dict[int, ShotPlan],
    images: List[Dict[str, Any]], provider: str,
) -> List[Dict[str, Any]]:
    """
    Génère un AnimationPrompt (contrat Sprint 25) par scène, à partir de
    l'ImagePrompt déjà produit pour cette scène et de la même VisualScene
    « dirigée » par le ShotPlan (Sprint 26) que celle utilisée pour l'image —
    LLMAnimationGenerator reçoit exactement les mêmes décisions de cadrage.
    """
    generator = LLMAnimationGenerator(script=script, provider_name=provider, max_retries=1)
    entries: List[Dict[str, Any]] = []
    ordered_scenes = sorted(visual_plan.scenes, key=lambda s: s.scene_order)
    images_by_order = {e["scene_order"]: e["image_prompt"] for e in images}

    for visual_scene in ordered_scenes:
        script_scene = next((s for s in script.scenes if s.order == visual_scene.scene_order), None)
        image_prompt = images_by_order.get(visual_scene.scene_order)
        if script_scene is None or image_prompt is None:
            logger.warning("  Aucun ImagePrompt/ScriptScene pour la scène #%d — animation ignorée.", visual_scene.scene_order)
            continue
        shot_plan = shot_plans.get(visual_scene.scene_order)
        directed_scene = _apply_shot_plan_to_visual_scene(visual_scene, shot_plan)
        animation_prompt = generator.generate_from_scenes(script_scene, directed_scene, image_prompt, script=script)
        entries.append({"scene_order": visual_scene.scene_order, "animation_prompt": animation_prompt})
        provider_used = animation_prompt.metadata.get("provider", "")
        logger.info("  Scène #%d : provider=%s (%s)", visual_scene.scene_order, provider_used,
                    "LLM" if provider_used != "fallback_heuristic" else "fallback heuristique")

    if not entries:
        raise RuntimeError("Aucun prompt d'animation généré.")
    logger.info("%d prompt(s) d'animation généré(s) (stats : %s).", len(entries), generator.stats)
    return entries


# ── Mapping niche → chaîne (BrandProfile.niche_keywords, Sprint 28) ─────────

def _keyword_matches_niche(niche_name_lower: str, keyword_lower: str) -> bool:
    """
    Correspondance mot entier (word-boundary) dans les deux sens — une simple
    sous-chaîne ferait matcher, par exemple, la niche "US" avec le mot-clé
    "business" (qui contient littéralement "us"). \b garantit que seuls des
    mots complets se correspondent.
    """
    return bool(
        re.search(rf"\b{re.escape(keyword_lower)}\b", niche_name_lower)
        or re.search(rf"\b{re.escape(niche_name_lower)}\b", keyword_lower)
    )


def select_brand_for_niche(niche: Niche, brand_engine: BrandEngine, default_brand_id: str) -> BrandProfile:
    """
    Fait correspondre une Niche détectée à une chaîne via BrandProfile.niche_keywords
    (correspondance mot entier, insensible à la casse, dans les deux sens).
    Retombe sur `default_brand_id` (--brand) si aucun mapping ne matche.
    """
    niche_name_lower = niche.name.lower()
    for profile in brand_engine.list():
        if any(_keyword_matches_niche(niche_name_lower, kw.lower()) for kw in profile.niche_keywords):
            logger.info("  Niche '%s' → chaîne '%s' (mapping niche_keywords).", niche.name, profile.id)
            return profile

    logger.warning(
        "  Aucun mapping niche_keywords pour '%s' — repli sur la marque par défaut '%s'.",
        niche.name, default_brand_id,
    )
    fallback = brand_engine.load(default_brand_id)
    if fallback is None:
        raise RuntimeError(f"Marque par défaut '{default_brand_id}' introuvable pour le repli niche→chaîne.")
    return fallback


# ── Production d'une vidéo complète pour UNE niche (Sprint 28) ─────────────

def produce_niche(
    niche: Niche,
    opportunities: List[Opportunity],
    brand_engine: BrandEngine,
    default_brand_id: str,
    provider: str,
    use_llm_judge: bool,
    do_rewrite: bool,
) -> Dict[str, Any]:
    """
    Produit une vidéo complète (brief → script → réécriture → visuel →
    shot plans → prompts image/animation) pour une seule niche, en composant
    les étapes existantes (aucune n'est modifiée) — appelée une fois par
    niche active du jour (2 fois/jour au maximum).
    """
    brand = select_brand_for_niche(niche, brand_engine, default_brand_id)

    briefs_map = step_generate_briefs(opportunities)
    entries = step_generate_scripts(opportunities, briefs_map, brand, provider)
    step_evaluate_scripts(entries, provider, use_llm_judge)
    best_entry = pick_best_entry(entries)
    logger.info(
        "  Meilleur script (%s) : '%s' (%.1f/80)",
        niche.name, best_entry["script"].title[:50], best_entry["heuristic_score"].composite_score,
    )

    if do_rewrite:
        rewrite_result = step_rewrite_best_script(best_entry, provider)
    else:
        logger.info("  Réécriture ignorée (--rewrite non activé).")
        rewrite_result = None

    final_script = rewrite_result["final_script"] if rewrite_result else best_entry["script"]

    visual_plan = step_generate_visual_plan(final_script)
    shot_plans = step_generate_shot_plans(final_script, brand, provider)
    images = step_generate_image_prompts(final_script, visual_plan, shot_plans, brand, provider)
    animations = step_generate_animation_prompts(final_script, visual_plan, shot_plans, images, provider)

    return {
        "niche": niche,
        "brand": brand,
        "entries": entries,
        "best_entry": best_entry,
        "rewrite_result": rewrite_result,
        "final_script": final_script,
        "visual_plan": visual_plan,
        "shot_plans": shot_plans,
        "images": images,
        "animations": animations,
    }


# ── Packages de production + sauvegarde technique ───────────────────────────

def step_build_packages(
    output_dir: Path,
    niche_productions: List[Dict[str, Any]],
    builder: ProductionPackageBuilder,
) -> List[Path]:
    """
    Construit le package de production propre (outputs/YYYY-MM-DD/niche_XX/)
    de chaque niche produite — voir ProductionPackageBuilder.
    """
    package_dirs: List[Path] = []
    for idx, prod in enumerate(niche_productions, 1):
        package_result = NicheProductionResult(
            niche=prod["niche"],
            brand=prod["brand"],
            final_script=prod["final_script"],
            images=prod["images"],
            animations=prod["animations"],
            rewrite_result=prod["rewrite_result"],
        )
        package_dirs.append(builder.build(output_dir, idx, package_result))
    return package_dirs


def step_save_technical_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    kb,
    candidate_niches: List[Niche],
    selected_niches: List[Niche],
    opportunities_by_niche: Dict[str, List[Opportunity]],
    niche_productions: List[Dict[str, Any]],
) -> None:
    """
    Sauvegarde tout ce qui est technique/interne (pas dans les packages de
    production) : knowledge.json, opportunities.json, scripts intermédiaires,
    shot_plans, benchmark.json, rapport.md — namespacé par niche.
    """
    (output_dir / "knowledge.json").write_text(kb.to_json(), encoding="utf-8")

    all_opportunities = [opp for opps in opportunities_by_niche.values() for opp in opps]
    _write_json(output_dir / "opportunities.json", {
        "niches_candidates": [_niche_summary(n) for n in candidate_niches[:10]],
        "niches_selected": [_niche_summary(n) for n in selected_niches],
        "opportunities_by_niche": {
            niche_name: [_asdict(o) for o in opps]
            for niche_name, opps in opportunities_by_niche.items()
        },
        "total_opportunities": len(all_opportunities),
    })

    benchmark: Dict[str, Any] = {
        "generated_at": date.today().isoformat(),
        "provider": args.provider,
        "llm_judge_enabled": args.llm_judge,
        "rewrite_enabled": args.rewrite,
        "niches_selected": [_niche_summary(n) for n in selected_niches],
        "productions": [],
    }

    for idx, prod in enumerate(niche_productions, 1):
        niche_slug = f"niche_{idx:02d}"
        scripts_dir = output_dir / "scripts" / niche_slug
        shot_plan_dir = output_dir / "shot_plans" / niche_slug
        scripts_dir.mkdir(parents=True, exist_ok=True)
        shot_plan_dir.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(prod["entries"], 1):
            opp = entry["opportunity"]
            brief = entry["brief"]
            payload = {
                "opportunity_id": opp.source_video_id,
                "niche": opp.niche,
                "brief_angle": brief.angle,
                "script": _asdict(entry["script"]),
                "heuristic_score": _asdict(entry["heuristic_score"]),
                "llm_judge_score": _asdict(entry["llm_judge_score"]) if entry.get("llm_judge_score") else None,
            }
            _write_json(scripts_dir / f"script_{i:02d}_{opp.source_video_id}_{brief.angle}.json", payload)

        _write_json(scripts_dir / "final_script.json", _asdict(prod["final_script"]))

        for order, shot_plan in sorted(prod["shot_plans"].items()):
            _write_json(shot_plan_dir / f"scene_{order:02d}.json", _asdict(shot_plan))
        _write_json(
            shot_plan_dir / "all_shot_plans.json",
            {order: _asdict(sp) for order, sp in sorted(prod["shot_plans"].items())},
        )

        rewrite_result = prod["rewrite_result"]
        benchmark["productions"].append({
            "niche": prod["niche"].name,
            "brand": prod["brand"].id,
            "scripts_generated": len(prod["entries"]),
            "best_script_title": prod["best_entry"]["script"].title,
            "heuristic_composite": prod["best_entry"]["heuristic_score"].composite_score,
            "rewrite_applied": rewrite_result["rewrite_applied"] if rewrite_result else False,
            "shot_plans_generated": len(prod["shot_plans"]),
            "images_generated": len(prod["images"]),
            "animations_generated": len(prod["animations"]),
        })

    _write_json(output_dir / "benchmark.json", benchmark)

    report = build_report_markdown(args, selected_niches, opportunities_by_niche, niche_productions)
    (output_dir / "rapport.md").write_text(report, encoding="utf-8")

    logger.info("Sorties techniques sauvegardées → %s", output_dir)


def build_report_markdown(
    args: argparse.Namespace,
    niches: List[Niche],
    opportunities_by_niche: Dict[str, List[Opportunity]],
    niche_productions: List[Dict[str, Any]],
) -> str:
    lines: List[str] = [
        "# Rapport Pipeline Quotidien — Studio de production",
        "",
        f"**Date :** {date.today().isoformat()}  ",
        f"**Provider LLM :** {args.provider}  ",
        f"**LLM Judge :** {'activé' if args.llm_judge else 'désactivé'}  ",
        f"**Réécriture :** {'activée' if args.rewrite else 'désactivée'}",
        "",
        "## Niches actives du jour",
        "",
        "| Niche | Score | Volume | Vues moy. | Engagement moy. |",
        "|-------|------:|-------:|----------:|-----------------:|",
    ]
    for n in niches:
        lines.append(f"| {n.name} | {n.niche_score:.3f} | {n.volume} | {n.avg_views:.0f} | {n.avg_engagement:.2f}% |")

    lines += ["", "## Opportunités retenues", ""]
    for niche_name, opps in opportunities_by_niche.items():
        lines.append(f"### {niche_name}")
        lines.append("")
        for opp in opps:
            lines.append(f"- **{opp.title[:70]}** — score={opp.overall_score:.3f}, urgence={opp.urgency:.2f}")
        lines.append("")

    for idx, prod in enumerate(niche_productions, 1):
        niche = prod["niche"]
        brand = prod["brand"]
        script = prod["final_script"]

        lines += [
            f"## Production {idx} — {niche.name} ({brand.name})",
            "",
            "### Scripts générés",
            "",
            "| # | Angle | Titre | Heuristique /80 | LLM-Judge /80 |",
            "|---|-------|-------|-----------------:|---------------:|",
        ]
        for i, e in enumerate(prod["entries"], 1):
            judge = e["llm_judge_score"].global_score if e.get("llm_judge_score") else "—"
            lines.append(
                f"| {i} | {e['brief'].angle} | {e['script'].title[:40]} "
                f"| {e['heuristic_score'].composite_score:.1f} | {judge if judge == '—' else f'{judge:.1f}'} |"
            )

        lines += [
            "",
            "### Script final",
            "",
            f"**Titre :** {script.title}  ",
            f"**Hook :** {script.hook}  ",
            f"**Durée estimée :** {script.estimated_duration}s  ",
            f"**Scènes :** {len(script.scenes)}",
            "",
        ]

        rewrite_result = prod["rewrite_result"]
        lines += ["### Réécriture (RewriteEngine)", ""]
        if rewrite_result is None:
            lines.append("Étape non exécutée (`--rewrite` non activé).")
        elif rewrite_result["rewrite_applied"]:
            lines.append(
                f"✅ Réécriture retenue — score LLM-judge {rewrite_result['score_before']:.1f} "
                f"→ {rewrite_result['score_after']:.1f}/80."
            )
        else:
            lines.append(
                f"⚠️ Réécriture non retenue (score initial {rewrite_result['score_before']:.1f}/80 "
                "non amélioré) — script original conservé."
            )

        visual_plan = prod["visual_plan"]
        shot_plans = prod["shot_plans"]
        lines += [
            "",
            "### VisualPlan / ShotPlan (VisualDirector)",
            "",
            f"- Scènes : {len(visual_plan.scenes)}",
            f"- Style : {visual_plan.style}",
            f"- Aspect ratio : {visual_plan.aspect_ratio}",
            f"- {len(shot_plans)} ShotPlan généré(s)",
            "",
        ]

        images = prod["images"]
        animations = prod["animations"]
        lines += [
            "### Prompts image / animation",
            "",
            f"- {len(images)} prompt(s) image généré(s) (contrat ImagePrompt — Whisk / Nano Banana)",
            f"- {len(animations)} prompt(s) animation généré(s) (contrat AnimationPrompt — Veo / Kling / Runway)",
            "",
        ]

    return "\n".join(lines)


# ── Résumé de fin de run (Issue 5, Sprint 29.1) ─────────────────────────────

def _count_llm_vs_fallback(
    niche_productions: List[Dict[str, Any]], list_key: str, prompt_key: str, fallback_provider: str,
) -> tuple[int, int]:
    """Compte, sur l'ensemble des niches produites, combien de prompts
    viennent du LLM vs du générateur de secours (heuristique)."""
    llm = 0
    fallback = 0
    for prod in niche_productions:
        for entry in prod[list_key]:
            provider = entry[prompt_key].metadata.get("provider", "")
            if provider == fallback_provider:
                fallback += 1
            else:
                llm += 1
    return llm, fallback


def build_production_summary_text(
    niche_productions: List[Dict[str, Any]],
    storage_results: List[UploadResult],
    telegram_result: NotificationResult,
    elapsed_seconds: float,
) -> str:
    """
    Construit le résumé final affiché dans les logs GitHub Actions (Issue 5) —
    seul indicateur qui condense en un coup d'œil l'état complet du run :
    vidéos produites, part LLM vs fallback (image/animation), résultat
    Supabase Storage, résultat Telegram, durée, statut global.
    """
    images_llm, images_fallback = _count_llm_vs_fallback(
        niche_productions, "images", "image_prompt", _IMAGE_FALLBACK_PROVIDER,
    )
    animations_llm, animations_fallback = _count_llm_vs_fallback(
        niche_productions, "animations", "animation_prompt", _ANIMATION_FALLBACK_PROVIDER,
    )

    storage_uploaded = sum(r.uploaded_count for r in storage_results)
    storage_total = sum(r.total_count for r in storage_results)
    if not storage_results:
        storage_status = "SKIPPED"
    elif all(r.success for r in storage_results):
        storage_status = "SUCCESS"
    elif storage_uploaded > 0:
        storage_status = "PARTIAL"
    else:
        storage_status = "FAILED"

    telegram_status = "SENT" if telegram_result.success else telegram_result.status.upper()

    warnings = storage_status != "SUCCESS" or not telegram_result.success
    overall_status = "SUCCESS (with warnings)" if warnings else "SUCCESS"

    lines = [
        "=" * 26,
        "Production Summary",
        "=" * 26,
        "",
        f"Videos produced: {len(niche_productions)}",
        "",
        "Image Prompts",
        "-" * 13,
        f"LLM: {images_llm}",
        f"Fallback: {images_fallback}",
        "",
        "Animation Prompts",
        "-" * 17,
        f"LLM: {animations_llm}",
        f"Fallback: {animations_fallback}",
        "",
        "Supabase Storage",
        "-" * 16,
        f"Uploaded files: {storage_uploaded}/{storage_total}",
        f"Upload status: {storage_status}",
        "",
        "Telegram",
        "-" * 8,
        f"Status: {telegram_status}",
        "",
        "Pipeline duration:",
        _format_duration(elapsed_seconds),
        "",
        "Overall status:",
        overall_status,
    ]
    return "\n".join(lines)


# ── Orchestration principale ────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline quotidien — point d'entrée unique du projet")
    parser.add_argument("--brand", type=str, default="ia_fr",
                        help="Marque de repli si aucun mapping niche_keywords ne correspond (défaut: ia_fr)")
    parser.add_argument("--provider", type=str, default="deepseek", help="Provider LLM (défaut: deepseek)")
    parser.add_argument("--top", type=int, default=30,
                        help="Nombre d'opportunités candidates évaluées avant sélection finale "
                             "(2 niches max × 3 opportunités max) — défaut: 30")
    parser.add_argument("--llm-judge", action="store_true", help="Évaluer aussi tous les scripts avec LLMScriptEvaluator")
    parser.add_argument("--rewrite", action="store_true", help="Réécrire le meilleur script avec RewriteEngine")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Répertoire de base des sorties (le sous-dossier YYYY-MM-DD est créé automatiquement)")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV_FALLBACK),
                        help="Chemin du CSV de repli si Supabase est indisponible (défaut: data/videos.csv)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / date.today().isoformat()
    cache_dir = output_dir / ".cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 76)
    print("  DAILY PIPELINE — YouTube Trend Watcher")
    print("=" * 76)
    print(f"  Marque repli : {args.brand}")
    print(f"  Provider     : {args.provider}")
    print(f"  Top N        : {args.top}")
    print(f"  LLM Judge    : {args.llm_judge}")
    print(f"  Réécriture   : {args.rewrite}")
    print(f"  Sortie       : {output_dir}")
    print("=" * 76)
    print()

    t_start = time.time()

    try:
        timelines, mirror_csv = run_step(1, "Chargement des données (Supabase → CSV fallback)",
                                          step_load_data, Path(args.csv), cache_dir)

        profiles, kb = run_step(2, "Construction de la KnowledgeBase", step_build_knowledge, timelines)

        niche_selector = NicheSelector(max_niches=MAX_NICHES)
        candidate_niches, niches = run_step(
            3, "Sélection des niches actives (NicheAnalyzer + NicheSelector — persistance)",
            step_select_active_niches, mirror_csv, niche_selector,
        )

        opportunities_by_niche = run_step(
            4, "Sélection des meilleures opportunités",
            step_select_opportunities, profiles, timelines, kb, niches, args.top,
        )

        be = BrandEngine()
        niche_productions: List[Dict[str, Any]] = []
        for idx, niche in enumerate(niches, 1):
            opportunities = opportunities_by_niche.get(niche.name, [])
            if not opportunities:
                logger.warning("Niche '%s' sans opportunité — production ignorée pour cette niche.", niche.name)
                continue
            prod = run_step(
                5, f"Production niche {idx}/{len(niches)} — {niche.name}",
                produce_niche, niche, opportunities, be, args.brand, args.provider, args.llm_judge, args.rewrite,
            )
            niche_productions.append(prod)

        if not niche_productions:
            raise PipelineStepError("Aucune niche n'a produit de vidéo — pipeline interrompu.")

        builder = ProductionPackageBuilder()
        package_dirs = run_step(
            6, "Construction des packages de production + sauvegarde technique",
            step_build_packages, output_dir, niche_productions, builder,
        )
        step_save_technical_outputs(output_dir, args, kb, candidate_niches, niches, opportunities_by_niche, niche_productions)

        storage_uploader = build_storage_uploader()

        def _upload_packages() -> List[UploadResult]:
            results = []
            for package_dir, prod in zip(package_dirs, niche_productions):
                remote_folder_name = f"production/{date.today().isoformat()}/{package_dir.name}"
                result = storage_uploader.upload_package(package_dir, remote_folder_name)
                logger.info(
                    "  Supabase Storage [%s] : success=%s uploaded=%d/%d error=%s",
                    prod["niche"].name, result.success, result.uploaded_count, result.total_count, result.error,
                )
                results.append(result)
            return results

        storage_results = run_step(7, "Envoi Supabase Storage (upload des packages de production)", _upload_packages)

        notifier = build_notification_service()

        def _send_notification() -> NotificationResult:
            channels = [
                ChannelSummary(
                    niche_name=prod["niche"].name,
                    channel_name=prod["brand"].name,
                    subject=prod["final_script"].title,
                    duration_seconds=prod["final_script"].estimated_duration,
                    scene_count=len(prod["final_script"].scenes),
                    storage_link=(storage_results[i].remote_url if storage_results[i].success else None),
                )
                for i, prod in enumerate(niche_productions)
            ]
            summary = DailyProductionSummary(
                date=date.today().isoformat(),
                channels=channels,
                pipeline_duration_seconds=time.time() - t_start,
            )
            return notifier.send_daily_summary(summary)

        telegram_result = run_step(8, "Notification du résumé quotidien", _send_notification)

    except PipelineStepError as exc:
        logger.error("=" * 76)
        logger.error("PIPELINE ARRÊTÉ : %s", exc)
        logger.error("Les résultats des étapes précédentes restent disponibles dans %s", output_dir)
        logger.error("=" * 76)
        sys.exit(1)

    elapsed = time.time() - t_start
    summary_text = build_production_summary_text(niche_productions, storage_results, telegram_result, elapsed)
    print()
    print(summary_text)
    print()
    print("=" * 76)
    print(f"  PIPELINE TERMINÉ — {elapsed:.1f}s")
    print(f"  Sorties : {output_dir}")
    print("=" * 76)


if __name__ == "__main__":
    main()
