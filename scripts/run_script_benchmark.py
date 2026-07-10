"""
Script de benchmark — Heuristique vs LLM (Sprint 21).

Objectif :
  Valider que les scripts générés par un LLM (DeepSeek, Groq) sont meilleurs
  que les scripts heuristiques, en utilisant le ScriptEvaluator pour noter
  chaque script sur 8 critères objectifs.

Pipeline :
  1. Charger les données (CSV → timelines → profiles → KnowledgeBase)
  2. Détecter les Top 20-30 opportunités avec OpportunityEngine
  3. Transformer chaque opportunité en CreativeBrief (CreativeEngine)
  4. Générer un script heuristique (HeuristicScriptGenerator)
  5. Générer 3 variantes LLM (LLMScriptGenerator avec provider choisi)
  6. Évaluer tous les scripts avec ScriptEvaluator
  7. Sauvegarder tous les scripts dans outputs/scripts/
  8. Produire un rapport Markdown classé

Usage :
    python scripts/run_script_benchmark.py                                    # Top 20, deepseek-chat
    python scripts/run_script_benchmark.py --top 30                          # Top 30 opportunités
    python scripts/run_script_benchmark.py --brand histoire_fr               # Marque spécifique
    python scripts/run_script_benchmark.py --provider groq                   # Utiliser Groq
    python scripts/run_script_benchmark.py --provider deepseek               # Utiliser DeepSeek (défaut)
    python scripts/run_script_benchmark.py --llm-model deepseek-chat         # Modèle spécifique
    python scripts/run_script_benchmark.py --output-only                     # Ne pas regénérer
    python scripts/run_script_benchmark.py --llm-judge                      # + évaluation LLM-judge (Sprint 21)
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Chemin du projet pour les imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.brand_engine import BrandEngine
from src.creative_engine import CreativeEngine
from src.llm_script_generator import LLMScriptGenerator
from src.llm_script_evaluator import LLMScriptEvaluator
from src.opportunity_engine import OpportunityEngine
from src.script_evaluator import ScriptEvaluator
from src.script_engine import (
    Script,
    ScriptEngine,
    HeuristicScriptGenerator,
)
from src.virality_engine import ViralityEngine
from src.content_understanding import ContentUnderstandingEngine
from src.knowledge_engine import KnowledgeEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Constantes ────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("outputs/scripts")
REPORT_DIR = Path("reports")


# ── Fonctions de pipeline ────────────────────────────────────────────────────

def load_data(csv_path: Path):
    """Charge et transforme les données du CSV jusqu'à KnowledgeBase."""
    logger.info("=== CHARGEMENT DES DONNÉES ===")

    if not csv_path.exists():
        logger.error("Fichier CSV introuvable : %s", csv_path)
        sys.exit(1)

    # ViralityEngine → timelines
    logger.info("[1/4] Chargement des timelines depuis %s ...", csv_path.name)
    ve = ViralityEngine(csv_path)
    timelines = ve._load_timelines()
    logger.info("  → %d timelines chargées", len(timelines))

    # ContentUnderstandingEngine → profiles
    logger.info("[2/4] Analyse sémantique des vidéos...")
    cue = ContentUnderstandingEngine()
    profiles = cue.analyze_all(timelines)
    logger.info("  → %d ContentProfile générés", len(profiles))

    # KnowledgeEngine → KnowledgeBase
    logger.info("[3/4] Construction de la KnowledgeBase...")
    ke = KnowledgeEngine()
    kb = ke.build(profiles)
    logger.info("  → %d sujets, %d combinaisons", len(kb.topics), len(kb.combinations))

    # OpportunityEngine → Opportunities (garder assez pour top_n)
    logger.info("[4/4] Détection des opportunités...")
    oe = OpportunityEngine()
    # On garde 3x top_n pour filtrer après
    opportunities = oe.build(profiles, timelines, kb, top_n=100)
    logger.info("  → %d opportunités détectées", len(opportunities))

    return timelines, profiles, kb, opportunities


def generate_scripts(
    opportunities,
    brand,
    top_n: int,
    provider: str = "deepseek",
    llm_model: Optional[str] = None,
):
    """
    Génère tous les scripts (heuristique + 3 variantes LLM) pour les top_n
    meilleures opportunités.

    Args:
        provider: "deepseek" (défaut) ou "groq"
        llm_model: Modèle LLM spécifique (None = défaut du provider)

    Returns:
        Dict[str, List[Script]] — mapping méthode → liste de scripts
    """
    logger.info("")
    logger.info("=== GÉNÉRATION DES SCRIPTS ===")
    logger.info("Top N : %d | Marque : %s | Provider : %s | Modèle : %s",
                top_n, brand.id, provider, llm_model or "(défaut)")

    # Créer les briefs pour les top opportunités
    ce = CreativeEngine()
    briefs_map = ce.generate_all(opportunities[:top_n])
    total_briefs = sum(len(b) for b in briefs_map.values())
    logger.info("Briefs créatifs : %d", total_briefs)

    # Initialiser les générateurs
    heuristic_gen = HeuristicScriptGenerator()
    llm_gen = LLMScriptGenerator(
        provider_name=provider,
        model=llm_model,
        temperature=0.7,
        max_tokens=4096,
        max_retries=1,
    )

    # Dictionnaire des résultats : label → liste de scripts
    results: Dict[str, List[Script]] = {
        "heuristic": [],
        f"{provider}_v1": [],
        f"{provider}_v2": [],
        f"{provider}_v3": [],
    }

    # Enregistrement des durées pour le rapport (target vs estimated)
    duration_records: Dict[str, List[Dict[str, Any]]] = {
        "heuristic": [], f"{provider}_v1": [], f"{provider}_v2": [], f"{provider}_v3": [],
    }

    # Statistiques de fallback
    fallback_count = {f"{provider}_v1": 0, f"{provider}_v2": 0, f"{provider}_v3": 0}
    llm_success_count = {f"{provider}_v1": 0, f"{provider}_v2": 0, f"{provider}_v3": 0}

    total = min(len(opportunities), top_n)
    llm_temps = [0.7, 0.85, 0.6]  # 3 températures différentes pour variété

    for idx, opp in enumerate(opportunities[:top_n], 1):
        logger.info("")
        logger.info("── Opportunité %d/%d : %s (score: %.4f) ──",
                     idx, total, opp.title[:60], opp.overall_score)

        briefs = briefs_map.get(opp.source_video_id, [])
        if not briefs:
            logger.warning("  Aucun brief pour cette opportunité, skip.")
            continue

        brief = briefs[0]  # Premier brief uniquement

        target_dur = brief.duration_seconds  # durée cible du brief

        # ── Script heuristique ─────────────────────────────────────────────────
        try:
            t0 = time.time()
            script_h = heuristic_gen.generate(opp, brief, brand)
            t_h = time.time() - t0
            results["heuristic"].append(script_h)
            duration_records["heuristic"].append({
                "target": target_dur,
                "estimated": script_h.estimated_duration,
                "title": script_h.title[:60],
            })
            logger.info("  [Heuristique] %s — %d scènes, %ds (%.1fs) [target=%ds]",
                        script_h.title[:40], len(script_h.scenes),
                        script_h.estimated_duration, t_h, target_dur)
        except Exception as exc:
            logger.error("  [Heuristique] ÉCHEC : %s", exc)

        # ── 3 variantes LLM ───────────────────────────────────────────────────
        for variant in range(3):
            label = f"{provider}_v{variant + 1}"

            # Délai inter-requêtes pour respecter les quotas
            if idx > 1 or variant > 0:
                api_delay = 3.0
                logger.info("  [%s] Respect quota : pause %.1fs avant %s...", provider, api_delay, label)
                time.sleep(api_delay)

            try:
                t0 = time.time()

                # Créer un générateur avec température différente pour chaque variante
                var_gen = LLMScriptGenerator(
                    provider_name=provider,
                    model=llm_model,
                    temperature=llm_temps[variant],
                    max_tokens=4096,
                    max_retries=1,
                )

                script_g = var_gen.generate(opp, brief, brand)
                t_g = time.time() - t0

                # Détection de fallback via metadata.generator
                gen_name = script_g.metadata.get("generator", "unknown")
                llm_prov = script_g.metadata.get("llm_provider", "")
                llm_mod = script_g.metadata.get("llm_model", "")
                is_fallback = (gen_name == "heuristic_v1")

                results[label].append(script_g)
                duration_records[label].append({
                    "target": target_dur,
                    "estimated": script_g.estimated_duration,
                    "title": script_g.title[:60],
                })

                if is_fallback:
                    fallback_count[label] += 1
                    logger.warning(
                        "  [%s] ⚠ FALLBACK (heuristic_v1) — title='%s'",
                        label, script_g.title[:40],
                    )
                else:
                    llm_success_count[label] += 1
                    logger.info(
                        "  [%s] %s — %d scènes, %ds (%.1fs, T=%.1f, prov=%s, model=%s)",
                        label, script_g.title[:40], len(script_g.scenes),
                        script_g.estimated_duration, t_g, llm_temps[variant],
                        llm_prov, llm_mod,
                    )

            except Exception as exc:
                logger.error("  [%s] ÉCHEC : %s", label, exc)

    # Statistiques
    logger.info("")
    logger.info("=== STATISTIQUES DE GÉNÉRATION ===")
    for label, scripts in results.items():
        logger.info("  %s : %d scripts générés", label, len(scripts))
    
    # Afficher les fallbacks
    total_fallbacks = sum(fallback_count.values())
    total_llm = sum(llm_success_count.values()) + total_fallbacks
    if total_llm > 0:
        logger.info("  Dont fallbacks %s→heuristique : %d/%d (%.0f%%)",
                    provider, total_fallbacks, total_llm,
                    total_fallbacks / total_llm * 100)
        for label in [f"{provider}_v1", f"{provider}_v2", f"{provider}_v3"]:
            fb = fallback_count[label]
            total = fb + llm_success_count[label]
            if total > 0:
                logger.info("    %s : %d fallbacks / %d total (%.0f%%)",
                           label, fb, total, fb / total * 100)

    return results, duration_records


def evaluate_all_scripts(results: Dict[str, List[Script]], provider: str = "deepseek"):
    """
    Évalue tous les scripts avec ScriptEvaluator.

    Returns:
        Résultat complet de ScriptEvaluator.compare()
    """
    logger.info("")
    logger.info("=== ÉVALUATION DES SCRIPTS ===")

    evaluator = ScriptEvaluator()

    # Aplatir tous les scripts avec leurs labels
    all_scripts: List[Script] = []
    all_labels: List[str] = []

    for label in ["heuristic", f"{provider}_v1", f"{provider}_v2", f"{provider}_v3"]:
        for i, script in enumerate(results.get(label, [])):
            all_scripts.append(script)
            short_title = script.title[:30] if script.title else "Sans titre"
            all_labels.append(f"{label} #{i+1} — {short_title}")

    logger.info("Total scripts à évaluer : %d", len(all_scripts))

    if not all_scripts:
        logger.warning("Aucun script à évaluer!")
        return None

    comparison = evaluator.compare(all_scripts, all_labels)

    # Afficher le classement
    logger.info("")
    logger.info("=== CLASSEMENT ===")
    logger.info("%-5s %-45s %-20s %s", "Rang", "Script", "Générateur", "Score")
    logger.info("-" * 80)
    for rank, result in enumerate(comparison["ranked"], 1):
        logger.info("%-5d %-45s %-20s %.1f/80",
                    rank,
                    result["title"][:43],
                    result["generator"],
                    result["score"].composite_score)

    # Moyennes par générateur
    logger.info("")
    logger.info("=== MOYENNES PAR GÉNÉRATEUR ===")
    for gen, avg in sorted(
        comparison["generator_averages"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        logger.info("  %-25s : %.1f/80", gen, avg)

    return comparison


def evaluate_with_llm_judge(
    results: Dict[str, List[Script]],
    provider: str = "deepseek",
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Évalue tous les scripts avec LLMScriptEvaluator (LLM-as-judge, Sprint 21).

    Contrairement à ScriptEvaluator (heuristique, gratuit), chaque évaluation
    ici est un appel LLM facturé — les échecs individuels sont capturés pour
    ne pas interrompre le benchmark.

    Returns:
        Dict avec "ranked", "generator_averages", "total_scripts", "stats"
        (ou None si aucune évaluation n'a abouti).
    """
    logger.info("")
    logger.info("=== ÉVALUATION LLM-JUDGE (Sprint 21) ===")

    judge = LLMScriptEvaluator(provider_name=judge_provider, model=judge_model, max_retries=1)

    all_scripts: List[Script] = []
    all_labels: List[str] = []
    for label in ["heuristic", f"{provider}_v1", f"{provider}_v2", f"{provider}_v3"]:
        for i, script in enumerate(results.get(label, [])):
            all_scripts.append(script)
            short_title = script.title[:30] if script.title else "Sans titre"
            all_labels.append(f"{label} #{i+1} — {short_title}")

    judge_results: List[Dict[str, Any]] = []
    for i, (script, label) in enumerate(zip(all_scripts, all_labels)):
        if i > 0:
            time.sleep(1.5)  # respect quota
        try:
            score = judge.evaluate(script)
            judge_results.append({
                "label": label,
                "generator": script.metadata.get("generator", "unknown"),
                "title": script.title,
                "score": score,
            })
            logger.info("  [LLM-judge] %s → %.1f/80", label, score.global_score)
        except Exception as exc:
            logger.warning("  [LLM-judge] échec pour '%s' : %s", label, exc)

    if not judge_results:
        logger.warning("Aucune évaluation LLM-judge disponible.")
        return None

    ranked = sorted(judge_results, key=lambda r: r["score"].global_score, reverse=True)

    from collections import defaultdict
    gen_scores: Dict[str, List[float]] = defaultdict(list)
    for r in judge_results:
        gen_scores[r["generator"]].append(r["score"].global_score)
    gen_avg = {gen: round(sum(v) / len(v), 1) for gen, v in gen_scores.items()}

    logger.info("")
    logger.info("=== CLASSEMENT LLM-JUDGE ===")
    for rank, r in enumerate(ranked, 1):
        logger.info("%-5d %-45s %-20s %.1f/80", rank, r["title"][:43], r["generator"], r["score"].global_score)

    logger.info("")
    logger.info("=== MOYENNES LLM-JUDGE PAR GÉNÉRATEUR ===")
    for gen, avg in sorted(gen_avg.items(), key=lambda x: x[1], reverse=True):
        logger.info("  %-25s : %.1f/80", gen, avg)

    return {
        "ranked": ranked,
        "generator_averages": gen_avg,
        "total_scripts": len(judge_results),
        "stats": judge.stats,
    }


def _add_llm_judge_section(
    markdown: str,
    heuristic_comparison: Dict[str, Any],
    judge_comparison: Optional[Dict[str, Any]],
) -> str:
    """Ajoute la comparaison Évaluateur Heuristique vs LLM-Judge au rapport."""
    if judge_comparison is None:
        return markdown

    lines = [
        "",
        "## Comparaison Évaluateur Heuristique vs LLM-Judge (Sprint 21)",
        "",
        "| Générateur | Heuristique (/80) | LLM-Judge (/80) | Écart |",
        "|-----------|-------------------:|-----------------:|------:|",
    ]

    h_avg = heuristic_comparison["generator_averages"]
    j_avg = judge_comparison["generator_averages"]
    for gen in sorted(set(h_avg) | set(j_avg)):
        hv, jv = h_avg.get(gen), j_avg.get(gen)
        if hv is None or jv is None:
            continue
        lines.append(f"| {gen} | {hv:.1f} | {jv:.1f} | {jv - hv:+.1f} |")

    h_ranked = heuristic_comparison["ranked"]
    j_ranked = judge_comparison["ranked"]
    h_top = h_ranked[0]["label"] if h_ranked else None
    j_top = j_ranked[0]["label"] if j_ranked else None
    agree = h_top is not None and h_top == j_top

    stats = judge_comparison["stats"]
    lines += [
        "",
        f"- **Meilleur script (évaluateur heuristique)** : {h_top}",
        f"- **Meilleur script (LLM-judge)** : {j_top}",
        f"- **Accord sur le meilleur script** : {'Oui' if agree else 'Non'}",
        f"- **Scripts évalués par le LLM-judge** : {judge_comparison['total_scripts']}",
        f"- **Coût total de l'évaluation LLM-judge** : ${stats['total_cost_usd']:.4f} "
        f"({stats['llm_calls']} appels, {stats['llm_failures']} échecs)",
        "",
    ]

    return markdown + "\n".join(lines)


def _add_duration_section(markdown: str, comparison, duration_records: Dict[str, List[Dict[str, Any]]], provider: str = "deepseek") -> str:
    """
    Ajoute une section d'analyse des durées au rapport Markdown.
    """
    lines = [
        "",
        "## Analyse des durées",
        "",
        "Comparaison entre la durée cible (CreativeBrief), la durée estimée (script généré) et l'écart relatif.",
        "",
        "| # | Générateur | Titre | Target Duration | Estimated Duration | Différence (%) |",
        "|---|-----------|-------|----------------:|-------------------:|---------------:|",
    ]

    rank = 1
    for r in comparison["ranked"]:
        label = r["label"]
        title = r["title"][:50]
        gen = r["generator"]
        label_parts = label.split(" #")
        gen_key = label_parts[0]
        if len(label_parts) > 1:
            rest = label_parts[1]
            for sep in [" — ", " - ", " – "]:
                if sep in rest:
                    num_str = rest.split(sep)[0]
                    try:
                        idx = int(num_str) - 1
                        break
                    except ValueError:
                        continue
            else:
                idx = 0
        else:
            idx = 0

        recs = duration_records.get(gen_key, [])
        if idx < len(recs):
            rec = recs[idx]
            target = rec["target"]
            estimated = rec["estimated"]
        else:
            target = estimated = 0

        diff = estimated - target
        diff_pct = (diff / target * 100) if target > 0 else 0.0
        diff_str = f"{diff:+d}s ({diff_pct:+.1f}%)"

        lines.append(
            f"| {rank} | {gen_key} | {title} | {target}s | {estimated}s | {diff_str} |"
        )
        rank += 1

    lines += [
        "",
        "### Moyenne par générateur",
        "",
        "| Générateur | Moyenne Target | Moyenne Estimated | Écart moyen | Écart moyen % |",
        "|-----------|--------------:|------------------:|-----------:|-------------:|",
    ]

    for gen_key in ["heuristic", f"{provider}_v1", f"{provider}_v2", f"{provider}_v3"]:
        recs = duration_records.get(gen_key, [])
        if not recs:
            continue
        avg_target = sum(r["target"] for r in recs) / len(recs)
        avg_est = sum(r["estimated"] for r in recs) / len(recs)
        avg_diff = avg_est - avg_target
        avg_diff_pct = (avg_diff / avg_target * 100) if avg_target > 0 else 0.0
        lines.append(
            f"| {gen_key} | {avg_target:.0f}s | {avg_est:.0f}s | {avg_diff:+.0f}s | {avg_diff_pct:+.1f}% |"
        )

    lines.append("")
    return markdown + "\n".join(lines)


def save_scripts(results: Dict[str, List[Script]]):
    """Sauvegarde tous les scripts dans outputs/scripts/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for label, scripts in results.items():
        data = {
            "generated_at": timestamp,
            "generator": label,
            "count": len(scripts),
            "scripts": [_script_to_dict(s) for s in scripts],
        }
        path = OUTPUT_DIR / f"scripts_{label}_{timestamp}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Scripts '%s' sauvegardés → %s (%d scripts)", label, path, len(scripts))


def save_report(comparison, markdown: str):
    """Sauvegarde le rapport Markdown."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"benchmark_scripts_{timestamp}.md"
    path.write_text(markdown, encoding="utf-8")
    logger.info("Rapport sauvegardé → %s", path)

    # Aussi en JSON pour traitement automatique
    json_path = REPORT_DIR / f"benchmark_scripts_{timestamp}.json"
    # Enlever les objets Script non sérialisables
    json_data = {
        "generated_at": timestamp,
        "total_scripts": comparison["total_scripts"],
        "generator_averages": comparison["generator_averages"],
        "ranked": [
            {
                "label": r["label"],
                "generator": r["generator"],
                "title": r["title"],
                "composite_score": r["score"].composite_score,
                "hook_score": r["score"].hook_score,
                "curiosity_score": r["score"].curiosity_score,
                "clarity_score": r["score"].clarity_score,
                "rhythm_score": r["score"].rhythm_score,
                "cta_score": r["score"].cta_score,
                "retention_score": r["score"].retention_score,
                "emotion_score": r["score"].emotion_score,
                "originality_score": r["score"].originality_score,
            }
            for r in comparison["ranked"]
        ],
    }
    json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Rapport JSON sauvegardé → %s", json_path)


# ── Fonction principale ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark — Scripts Heuristique vs LLM (Sprint 21)"
    )
    parser.add_argument("--brand", type=str, default="ia_fr",
                        help="Identifiant de la marque (défaut: ia_fr)")
    parser.add_argument("--top", type=int, default=20,
                        help="Nombre d'opportunités à traiter (défaut: 20)")
    parser.add_argument("--provider", type=str, default="deepseek",
                        help="Provider LLM (deepseek, groq) pour la génération (défaut: deepseek)")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Modèle LLM pour la génération (défaut: deepseek-chat ou groq-llama3-8b selon provider)")
    parser.add_argument("--output-only", action="store_true",
                        help="Ne pas regénérer — utiliser les scripts déjà sauvegardés")
    parser.add_argument("--csv", type=str, default="data/videos.csv",
                        help="Chemin du fichier CSV (défaut: data/videos.csv)")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Évaluer aussi tous les scripts avec LLMScriptEvaluator "
                             "(Sprint 21) — appels LLM supplémentaires, facturés")
    parser.add_argument("--judge-provider", type=str, default=None,
                        help="Provider LLM pour le juge (défaut: identique à --provider)")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Modèle LLM pour le juge (défaut du provider si non précisé)")
    args = parser.parse_args()

    print()
    print("=" * 72)
    print(f"  SCRIPT BENCHMARK — Heuristique vs {args.provider.upper()} (Sprint 21)")
    print("=" * 72)
    print(f"  Marque       : {args.brand}")
    print(f"  Top N        : {args.top}")
    print(f"  Provider     : {args.provider}")
    print(f"  Modèle LLM   : {args.llm_model or '(défaut du provider)'}")
    print(f"  CSV          : {args.csv}")
    print("=" * 72)
    print()

    t_start = time.time()

    # ── 1. Chargement des données ─────────────────────────────────────────────
    csv_path = Path(args.csv)
    timelines, profiles, kb, opportunities = load_data(csv_path)

    # ── 2. Brand ──────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== CHARGEMENT DE LA MARQUE ===")
    be = BrandEngine()
    brand = be.load(args.brand)
    if brand is None:
        disponibles = [p.id for p in be.list()]
        logger.error("Marque '%s' introuvable. Disponibles : %s",
                     args.brand, ", ".join(disponibles))
        sys.exit(1)
    logger.info("Marque : %s (%s)", brand.name, brand.id)

    # ── 3. Génération des scripts ─────────────────────────────────────────────
    results, duration_records = generate_scripts(
        opportunities=opportunities,
        brand=brand,
        top_n=args.top,
        provider=args.provider,
        llm_model=args.llm_model,
    )

    total_generated = sum(len(s) for s in results.values())
    if total_generated == 0:
        logger.error("Aucun script généré. Abandon.")
        sys.exit(1)

    # ── 4. Sauvegarde des scripts ─────────────────────────────────────────────
    save_scripts(results)

    # ── 5. Évaluation ─────────────────────────────────────────────────────────
    comparison = evaluate_all_scripts(results, provider=args.provider)

    if comparison is None:
        logger.error("Évaluation impossible.")
        sys.exit(1)

    # ── 5bis. Évaluation LLM-judge (optionnelle, Sprint 21) ──────────────────
    judge_comparison = None
    if args.llm_judge:
        judge_comparison = evaluate_with_llm_judge(
            results,
            provider=args.provider,
            judge_provider=args.judge_provider or args.provider,
            judge_model=args.judge_model,
        )

    # ── 6. Rapport Markdown (enrichi avec l'analyse des durées) ─────────────
    evaluator = ScriptEvaluator()
    markdown = evaluator.generate_markdown_report(comparison)
    # Ajouter la section d'analyse des durées
    markdown = _add_duration_section(markdown, comparison, duration_records, provider=args.provider)
    # Ajouter la comparaison Heuristique vs LLM-judge si demandée
    markdown = _add_llm_judge_section(markdown, comparison, judge_comparison)
    save_report(comparison, markdown)

    # ── 7. Affichage du rapport ───────────────────────────────────────────────
    # Le fichier est deja sauvegarde en UTF-8 (save_report) ; l'affichage
    # console peut echouer sur un terminal Windows en cp1252 (emojis, etc.).
    print()
    try:
        print(markdown)
    except UnicodeEncodeError:
        print(markdown.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))

    t_elapsed = time.time() - t_start
    print()
    print("=" * 72)
    print(f"  BENCHMARK TERMINÉ — {t_elapsed:.1f}s")
    print(f"  Scripts générés : {total_generated}")
    print(f"  Scripts évalués : {comparison['total_scripts']}")
    print(f"  Rapport         : reports/benchmark_scripts_*.md")
    print(f"  Scripts         : outputs/scripts/")
    print("=" * 72)


# ── Utilitaires ───────────────────────────────────────────────────────────────

def _script_to_dict(script: Script) -> dict:
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
