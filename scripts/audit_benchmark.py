"""
Script d'audit du benchmark (Sprint 20).

Exécute un mini-benchmark (top 2-3 opportunités seulement) avec
des logs ultra-détaillés pour identifier :
  - Le provider réellement utilisé
  - Les fallbacks silencieux
  - Les titres identiques
  - Les différences réelles entre scripts

Usage :
    python scripts/audit_benchmark.py --top 3
    python scripts/audit_benchmark.py --top 3 --brand ia_fr
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Forcer UTF-8 pour éviter les erreurs d'encodage sous Windows
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Import des vrais modules
from src.brand_engine import BrandEngine
from src.creative_engine import CreativeEngine
from src.llm_script_generator import LLMScriptGenerator
from src.opportunity_engine import Opportunity
from src.opportunity_engine import OpportunityEngine
from src.script_evaluator import ScriptEvaluator
from src.script_engine import (
    Script,
    HeuristicScriptGenerator,
)
from src.virality_engine import ViralityEngine
from src.content_understanding import ContentUnderstandingEngine
from src.knowledge_engine import KnowledgeEngine

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-7s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Moniteur LLM ──────────────────────────────────────────────────────────────

# Compteurs globaux pour l'audit
AUDIT = {
    "heuristic": {"count": 0, "titles": []},
    "groq_v1": {"count": 0, "titles": [], "fallback": 0, "provider": "", "model": "", "errors": []},
    "groq_v2": {"count": 0, "titles": [], "fallback": 0, "provider": "", "model": "", "errors": []},
    "groq_v3": {"count": 0, "titles": [], "fallback": 0, "provider": "", "model": "", "errors": []},
    "all_groq_titles": [],
    "all_heuristic_titles": [],
    "all_groq_scenes_count": [],
    "all_heuristic_scenes_count": [],
}


def instrumented_generate_llm(original_generate, self, opportunity, creative_brief, brand_profile):
    """Wrapper d'audit autour de LLMScriptGenerator.generate()."""
    logger.info("=" * 60)
    logger.info("[AUDIT] LLMScriptGenerator.generate() appelé")
    logger.info("[AUDIT] Provider config: provider_name=%s, model=%s",
                self._provider_name, self._model)
    logger.info("[AUDIT] Temperature=%.2f, max_retries=%d",
                self._temperature, self._max_retries)
    logger.info("[AUDIT] Brief: title='%s', angle='%s'",
                creative_brief.title[:60], creative_brief.angle)
    
    # Vérifier quel provider sera utilisé
    if self._provider is None and self._provider_name == "groq":
        from src.llm import build_llm, GroqProvider
        try:
            test_provider = build_llm(provider="groq", model=self._model)
            logger.info("[AUDIT] Provider réel: %s / %s",
                        test_provider.name, test_provider.model)
            if isinstance(test_provider, GroqProvider):
                logger.info("[AUDIT] ✓ GroqProvider sera utilisé")
                has_key = bool(test_provider._api_key)
                logger.info("[AUDIT] Clé API Groq présente: %s", has_key)
                if has_key:
                    logger.info("[AUDIT]   (préfixe: %s...)", test_provider._api_key[:10])
                else:
                    logger.warning("[AUDIT] ✗ PAS DE CLÉ GROQ! Fallback attendu!")
            else:
                logger.warning("[AUDIT] ✗ Provider n'est PAS GroqProvider! C'est: %s",
                              type(test_provider).__name__)
        except Exception as e:
            logger.error("[AUDIT] ✗ Erreur build_llm: %s", e)
    
    # Appel réel
    start = time.time()
    try:
        script = original_generate(self, opportunity, creative_brief, brand_profile)
        elapsed = time.time() - start
        
        # Analyser le résultat
        gen_name = script.metadata.get("generator", "unknown")
        gen_provider = script.metadata.get("llm_provider", "")
        gen_model = script.metadata.get("llm_model", "")
        
        logger.info("[AUDIT] ✓ Script reçu: '%s'", script.title[:60])
        logger.info("[AUDIT]   metadata.generator = '%s'", gen_name)
        logger.info("[AUDIT]   metadata.llm_provider = '%s'", gen_provider)
        logger.info("[AUDIT]   metadata.llm_model = '%s'", gen_model)
        logger.info("[AUDIT]   Scènes: %d", len(script.scenes))
        logger.info("[AUDIT]   Durée: %ds", script.estimated_duration)
        logger.info("[AUDIT]   Temps total: %.1fs", elapsed)
        
        # Détection de fallback
        if gen_name == "heuristic_v1":
            logger.warning("[AUDIT] ⚠ FALLBACK DÉTECTÉ! Le générateur LLM a retourné un script heuristique!")
        elif "llm" in gen_name:
            logger.info("[AUDIT] ✓ Génération LLM confirmée")
        else:
            logger.warning("[AUDIT] ? Générateur inattendu: %s", gen_name)
        
        return script
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[AUDIT] ✗ Exception dans generate(): %s", e)
        logger.error("[AUDIT]   Traceback: %s", traceback.format_exc())
        raise


# ── Pipeline d'audit ──────────────────────────────────────────────────────────

def load_data(csv_path: Path):
    """Charge les données (copié de run_script_benchmark)."""
    logger.info("=== CHARGEMENT DES DONNÉES ===")
    if not csv_path.exists():
        logger.error("CSV introuvable: %s", csv_path)
        sys.exit(1)
    
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    
    ve = ViralityEngine(csv_path)
    timelines = ve._load_timelines()
    logger.info("  → %d timelines", len(timelines))
    
    cue = ContentUnderstandingEngine()
    profiles = cue.analyze_all(timelines)
    logger.info("  → %d profiles", len(profiles))
    
    ke = KnowledgeEngine()
    kb = ke.build(profiles)
    logger.info("  → KnowledgeBase: %d sujets", len(kb.topics))
    
    oe = OpportunityEngine()
    opportunities = oe.build(profiles, timelines, kb, top_n=100)
    logger.info("  → %d opportunités", len(opportunities))
    
    return timelines, profiles, kb, opportunities


def main():
    parser = argparse.ArgumentParser(description="Audit du benchmark Sprint 20")
    parser.add_argument("--top", type=int, default=3, help="Nombre d'opportunités (défaut: 3)")
    parser.add_argument("--brand", type=str, default="ia_fr", help="Marque")
    parser.add_argument("--groq-model", type=str, default="llama-3.3-70b-versatile")
    parser.add_argument("--csv", type=str, default="data/videos.csv")
    args = parser.parse_args()
    
    print()
    print("=" * 72)
    print("  AUDIT DU BENCHMARK — Heuristique vs Groq")
    print("=" * 72)
    print(f"  Top N        : {args.top}")
    print(f"  Marque       : {args.brand}")
    print(f"  Groq modèle  : {args.groq_model}")
    print("=" * 72)
    print()
    
    start_total = time.time()
    
    # ── 1. Chargement ─────────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    timelines, profiles, kb, opportunities = load_data(csv_path)
    
    # ── 2. Brand ──────────────────────────────────────────────────────────────
    be = BrandEngine()
    brand = be.load(args.brand)
    if brand is None:
        logger.error("Marque '%s' introuvable", args.brand)
        sys.exit(1)
    logger.info("Marque: %s (%s)", brand.name, brand.id)
    
    # ── 3. Briefs ─────────────────────────────────────────────────────────────
    ce = CreativeEngine()
    briefs_map = ce.generate_all(opportunities[:args.top])
    
    # ── 4. Audit des générateurs ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 72)
    logger.info("  PHASE D'AUDIT — Génération des scripts")
    logger.info("=" * 72)
    
    # Patch de LLMScriptGenerator.generate() pour instrumentation
    original_generate = LLMScriptGenerator.generate
    
    def patched_generate(self, opportunity, creative_brief, brand_profile):
        return instrumented_generate_llm(original_generate, self, opportunity, creative_brief, brand_profile)
    
    LLMScriptGenerator.generate = patched_generate
    
    # Initialiser les générateurs
    heuristic_gen = HeuristicScriptGenerator()
    
    results: Dict[str, List[Script]] = {
        "heuristic": [],
        "groq_v1": [],
        "groq_v2": [],
        "groq_v3": [],
    }
    
    groq_temps = [0.7, 0.85, 0.6]
    groq_models = [args.groq_model, args.groq_model, args.groq_model]
    
    for idx, opp in enumerate(opportunities[:args.top], 1):
        print()
        print(f"── [{idx}/{args.top}] {opp.title[:70]} (score={opp.overall_score:.4f}) ──")
        
        briefs = briefs_map.get(opp.source_video_id, [])
        if not briefs:
            logger.warning("  Pas de brief, skip")
            continue
        
        brief = briefs[0]
        logger.info("  Brief utilisé: angle='%s', hook='%s'", brief.angle, brief.hook[:50])
        
        # ── Script heuristique ────────────────────────────────────────────────
        print()
        print(f"  [HEURISTIQUE] Génération...")
        t0 = time.time()
        script_h = heuristic_gen.generate(opp, brief, brand)
        t_h = time.time() - t0
        results["heuristic"].append(script_h)
        AUDIT["heuristic"]["count"] += 1
        AUDIT["heuristic"]["titles"].append(script_h.title)
        AUDIT["all_heuristic_titles"].append(script_h.title)
        AUDIT["all_heuristic_scenes_count"].append(len(script_h.scenes))
        print(f"  ✓ Heuristique: '{script_h.title}' ({len(script_h.scenes)} scènes, {t_h:.1f}s)")
        print(f"    Hook: {script_h.hook[:60]}...")
        print(f"    metadata.generator: {script_h.metadata.get('generator')}")
        
        # ── 3 variantes Groq ──────────────────────────────────────────────────
        for variant in range(3):
            label = f"groq_v{variant + 1}"
            print()
            print(f"  [{label.upper()}] Génération (T={groq_temps[variant]})...")
            
            try:
                var_gen = LLMScriptGenerator(
                    provider_name="groq",
                    model=groq_models[variant],
                    temperature=groq_temps[variant],
                    max_tokens=4096,
                    max_retries=1,
                )
                
                t0 = time.time()
                script_g = var_gen.generate(opp, brief, brand)
                t_g = time.time() - t0
                
                results[label].append(script_g)
                gen_name = script_g.metadata.get("generator", "unknown")
                llm_prov = script_g.metadata.get("llm_provider", "")
                llm_mod = script_g.metadata.get("llm_model", "")
                
                AUDIT[label]["count"] += 1
                AUDIT[label]["titles"].append(script_g.title)
                AUDIT["all_groq_titles"].append(script_g.title)
                AUDIT["all_groq_scenes_count"].append(len(script_g.scenes))
                
                # Détection de fallback
                is_fallback = gen_name == "heuristic_v1"
                if is_fallback:
                    AUDIT[label]["fallback"] += 1
                    print(f"  ⚠ FALLBACK: le LLM a retourné un script heuristique!")
                
                print(f"  ✓ '{script_g.title}'")
                print(f"    Scènes: {len(script_g.scenes)}, Durée: {script_g.estimated_duration}s")
                print(f"    Temps: {t_g:.1f}s")
                print(f"    metadata.generator: '{gen_name}'")
                print(f"    metadata.llm_provider: '{llm_prov}'")
                print(f"    metadata.llm_model: '{llm_mod}'")
                print(f"    Fallback: {'OUI ⚠' if is_fallback else 'NON ✓'}")
                
            except Exception as exc:
                AUDIT[label]["errors"].append(str(exc))
                logger.error("  ✗ %s ÉCHEC: %s", label, exc)
    
    # ── 5. Rapport d'audit ────────────────────────────────────────────────────
    print()
    print()
    print("=" * 72)
    print("  RAPPORT D'AUDIT")
    print("=" * 72)
    print()
    
    # 5a. Statistiques par label
    print("── Statistiques de génération ──")
    for label in ["heuristic", "groq_v1", "groq_v2", "groq_v3"]:
        data = AUDIT[label]
        print(f"  {label}:")
        print(f"    Scripts générés : {data['count']}")
        if data['titles']:
            print(f"    Titres          : {data['titles']}")
        if isinstance(data, dict) and 'fallback' in data:
            print(f"    Fallbacks       : {data['fallback']}")
        if isinstance(data, dict) and data.get('errors'):
            print(f"    Erreurs         : {data['errors']}")
    print()
    
    # 5b. Titres identiques
    print("── Analyse des titres ──")
    all_titles = {}
    for label in ["heuristic", "groq_v1", "groq_v2", "groq_v3"]:
        for t in AUDIT[label]["titles"]:
            all_titles.setdefault(t, []).append(label)
    
    duplicates = {t: labels for t, labels in all_titles.items() if len(labels) > 1}
    if duplicates:
        print(f"  ⚠ Titres identiques détectés ({len(duplicates)}):")
        for t, labels in duplicates.items():
            print(f"    '{t[:60]}' → {labels}")
    else:
        print("  ✓ Aucun titre dupliqué")
    
    # Comparaison heuristique vs groq
    heuristic_titles = set(AUDIT["heuristic"]["titles"])
    groq_titles = set(AUDIT["all_groq_titles"])
    common = heuristic_titles & groq_titles
    if common:
        print(f"  ⚠ Titres identiques entre heuristique et Groq: {common}")
    else:
        print("  ✓ Aucun titre commun entre heuristique et Groq")
    print()
    
    # 5c. Analyse des providers dans metadata
    print("── Analyse des metadata des scripts Groq ──")
    for label in ["groq_v1", "groq_v2", "groq_v3"]:
        for i, script in enumerate(results.get(label, [])):
            gen = script.metadata.get("generator", "N/A")
            prov = script.metadata.get("llm_provider", "N/A")
            mod = script.metadata.get("llm_model", "N/A")
            score_info = script.metadata.get("opportunity_score", "N/A")
            print(f"  {label}#{i+1}: gen='{gen}', prov='{prov}', model='{mod}', score={score_info}")
    print()
    
    # 5d. Évaluation
    print("── Évaluation avec ScriptEvaluator ──")
    evaluator = ScriptEvaluator()
    
    all_scripts: List[Script] = []
    all_labels: List[str] = []
    for label in ["heuristic", "groq_v1", "groq_v2", "groq_v3"]:
        for i, script in enumerate(results.get(label, [])):
            all_scripts.append(script)
            all_labels.append(f"{label} #{i+1}")
    
    if all_scripts:
        comparison = evaluator.compare(all_scripts, all_labels)
        print()
        print(f"  { 'Rang':<5} {'Label':<25} {'Générateur':<20} {'Score':<8} {'Titre':<40}")
        print(f"  {'-'*5} {'-'*25} {'-'*20} {'-'*8} {'-'*40}")
        for rank, r in enumerate(comparison["ranked"], 1):
            gen = r["score"].details.get("generator", "?")
            print(f"  {rank:<5} {r['label']:<25} {gen:<20} {r['score'].composite_score:<8.1f} {r['title'][:38]}")
        
        print()
        print("  Moyennes par générateur (metadata.generator):")
        for gen, avg in sorted(comparison["generator_averages"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {gen:<25}: {avg:.1f}/80")
        
        print()
        print("  Détail des scores par critère:")
        for rank, r in enumerate(comparison["ranked"], 1):
            s = r["score"]
            gen = r["score"].details.get("generator", "?")
            print(f"  #{rank} {r['label'][:20]:20} gen={gen:<15} H={s.hook_score:.1f} C={s.curiosity_score:.1f} Cl={s.clarity_score:.1f} R={s.rhythm_score:.1f} CTA={s.cta_score:.1f} Ret={s.retention_score:.1f} E={s.emotion_score:.1f} O={s.originality_score:.1f} | Total={s.composite_score:.1f}")
    
    # ── 6. Temps total ────────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    print()
    print("=" * 72)
    print(f"  AUDIT TERMINÉ — {elapsed:.1f}s")
    total_scripts = sum(len(s) for s in results.values())
    total_fallbacks = sum(AUDIT[l].get("fallback", 0) for l in ["groq_v1", "groq_v2", "groq_v3"])
    print(f"  Scripts générés : {total_scripts}")
    print(f"  Fallbacks       : {total_fallbacks}")
    print(f"  Titres uniques  : {len(all_titles)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
