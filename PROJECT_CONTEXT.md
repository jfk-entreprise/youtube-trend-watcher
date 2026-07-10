# YouTube Trend Watcher — Contexte Technique du Projet
>
> **Phase :** Sprint 23 — LLM Script Evaluator, Rewrite Engine, LLM Image Prompt Generator  
> **586 tests unitaires (projet complet) — ✅ 100 % passent** (2026-07-08, `pytest tests/ -q`)

---

## 1. Vision du projet

**YouTube Trend Watcher** est une plateforme modulaire d'**intelligence de marché YouTube**. Elle ne se contente pas de collecter des vidéos — elle construit une boucle complète **Collecte → Analyse → Décision → Création de contenu**.

Le système transforme des données brutes issues de l'API YouTube Data v3 en :

| Étape | Sortie | Usage |
|-------|--------|-------|
| **Collecte** | `VideoSnapshot` | État d'une vidéo à un instant T |
| **Virality Engine** | Score viral composite (statique + temporel) | Classement par potentiel viral |
| **Content Understanding** | `ContentProfile` (sujet, audience, émotion, pérennité) | Analyse sémantique |
| **Knowledge Engine** | `KnowledgeBase` (distributions, combinaisons) | Apprentissage marché |
| **Niche Analyzer** | Analyse sectorielle par mot-clé | Intelligence concurrentielle |
| **Opportunity Engine** | `Opportunity` classées par 7 critères | Détection d'opportunités |
| **Creative Engine** | `CreativeBrief` (hooks, structures, CTA) | Production de contenu |
| **Brand Engine** | `BrandProfile` (identité éditoriale) | Cohérence de marque |

> **Ce n'est pas un outil pour regarder YouTube, mais pour décider quoi y publier.**

---

## 2. Architecture — Flux de données

```
  YouTube API Data v3
       ↓ (search.list / videos.list)
  ┌─────────────┐  ┌──────────────┐
  │KeywordAgent │  │TrendingAgent │     ← Collecteurs
  └──────┬──────┘  └──────┬───────┘
         │                │
         ▼                ▼
  ┌──────────────────────────────┐
  │     Storage Backend          │     ← CsvStorage / SupabaseStorage
  │  (FallbackStorage possible)  │
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │     ViralityEngine           │     ← Score viral (6 critères modulaires)
  │     (VideoTimeline)          │        Statique + Temporel (Time Engine)
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │ ContentUnderstandingEngine   │     ← 7 analyseurs modulaires
  │     → ContentProfile         │        (Topic, Language, Emotion, etc.)
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │   NicheAnalyzer / Knowledge  │     ← Analyse sectorielle + agrégation
  │   Engine → KnowledgeBase     │        Distributions, combinaisons
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │   OpportunityEngine          │     ← 7 critères d'opportunité
  │     → Opportunités classées  │
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │   CreativeEngine             │     ← 3-5 briefs créatifs/opportunité
  │     → CreativeBriefs         │        (HeuristicGenerator)
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │   ScriptEngine               │     ← Script complet découpé en scènes
  │     → Script                 │        (HeuristicScriptGenerator)
  │       ├── Hook → Contexte →  │
  │       │   Développement → …  │
  │       │   → Conclusion → CTA │
  │       └── Scènes avec        │
  │           image_prompt,      │
  │           animation_notes,   │
  │           sound_effects      │
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │   LearningEngine             │     ← Boucle d'amélioration continue
  │     → LearningProfile        │        7 dimensions (hook, angle, durée,
  │   PerformanceMetrics (input) │        CTA, style, émotion, format)
  │                              │
  │   API : best_hook(),         │
  │         best_angle(),        │
  │         best_duration(),     │
  │         best_cta(),          │
  │         best_style(),        │
  │         best_emotion(),      │
  │         best_format()        │
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │   LLM Provider               │     ← Couche d'abstraction unique
  │     → OpenAIProvider         │       pour tous les modèles IA
  │     → GeminiProvider         │       (generate, json_mode, cost)
  │     → ClaudeProvider         │
  │     → OllamaProvider (stub)  │
  │     → DeepSeekProvider (API) │     ← Implémentation REST réelle
  └──────────────────────────────┘

                │ (Script)
                ▼
  ┌──────────────────────────────┐
  │   Visual Engine               │    ← Script → VisualPlan (par scène)
  │     → VisualPlan             │        Aucune dépendance moteur (Script only)
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │   Image Engine                │    ← VisualPlan → GeneratedImage
  │     → GeneratedImage         │        (prompt final, negative_prompt, seed…)
  │                               │        (Futur) FluxProvider/SdProvider/FalProvider/…
  └──────────────────────────────┘

  ┌──────────────────────────────┐
  │   Script Evaluator (Sprint 20)│    ← Note un Script sur 8 critères /10
  │     → ScriptScore            │        Compare heuristique vs LLM (Groq/DeepSeek)
  │     ← BaseEvaluator (ABC)    │        Interface commune polymorphe
  └─────────────┬────────────────┘
                │
                ▼
  ┌──────────────────────────────┐
  │ LLM Script Evaluator (Sprint 21)│  ← LLM-as-judge, 8 critères /10
  │     → LLMScriptScore         │       (hook, curiosité, storytelling, rythme,
  │       + strengths/weaknesses │        clarté, CTA, rétention, viral)
  │       + suggestions          │       global_score toujours recalculé serveur
  └─────────────┬────────────────┘
                │ (Script + LLMScriptScore)
                ▼
  ┌──────────────────────────────┐
  │   Rewrite Engine (Sprint 22) │    ← Réécrit UNIQUEMENT hook/rythme/
  │     → Script (amélioré ou    │       storytelling/CTA/rétention
  │       original conservé)     │       Sujet/marque/durée/scènes intouchés
  │                               │       Garde la version si score ↑, sinon
  │                               │       conserve l'ancienne
  └──────────────────────────────┘

  ┌──────────────────────────────┐
  │ LLM Image Generator (Sprint 23)│  ← ScriptScene+VisualScene+BrandProfile
  │     → GeneratedImage         │       → prompt professionnel (Flux/SDXL/MJ)
  │     (implémente ImageGenerator)│     Dimensions/seed réutilisent
  │                               │       HeuristicImageGenerator (DRY)
  └──────────────────────────────┘

  (Futur) Animation Engine → Video Engine
```

---

## 3. Contrats d'interface

| Étape | Entrée | Sortie | Contrat |
|-------|--------|--------|---------|
| **Collecte** | API YouTube | `list[VideoSnapshot]` | Modèle immuable |
| **Stockage** | `list[VideoSnapshot]` | Persistance append-only | `StorageBackend` (ABC) |
| **Viralité** | CSV → `list[VideoTimeline]` | Rapport texte + scores | `ViralityEngine.run()` |
| **Sémantique** | `VideoSnapshot` + `VideoTimeline` | `ContentProfile` | 7 `BaseAnalyzer` |
| **Connaissance** | `list[ContentProfile]` | `KnowledgeBase` | `KnowledgeEngine.build()` |
| **Opportunité** | `ContentProfile` + `VideoTimeline` + `KnowledgeBase` | `list[Opportunity]` | `OpportunityEngine.build()` |
| **Création** | `Opportunity` | `list[CreativeBrief]` | `CreativeGenerator.generate()` |
| **Script** | `Opportunity` + `CreativeBrief` + `BrandProfile` | `Script` | `ScriptGenerator.generate()` |
| **Apprentissage** | `Opportunity` + `CreativeBrief` + `BrandProfile` + `Script` + `PerformanceMetrics` | `LearningProfile` | `LearningEngine.record()` |
| **LLM** | `list[LLMMessage]` | `LLMResponse` | `LLMProvider.generate()` |
| **Marque** | Fichier JSON | `BrandProfile` | `BrandStore` |

---

## 4. Modules (dépendances)

```
src/
├── models.py                   # VideoSnapshot (dataclass, pivot du système)
├── utils.py                    # Utilitaires partagés (parse, format, CSV)
├── collector.py                # YouTubeCollector (façade API)
├── storage.py                  # CsvStorage, SupabaseStorage, FallbackStorage
├── agents/
│   ├── base.py                 # BaseAgent (ABC)
│   ├── keyword_agent.py        # KeywordAgent (search.list)
│   └── trending_agent.py       # TrendingAgent (videos.list)
├── virality_engine.py          # ViralityEngine + VideoTimeline + ScoringCriterion
├── content_understanding.py    # ContentUnderstandingEngine + 7 BaseAnalyzer
├── knowledge_engine.py         # KnowledgeEngine + KnowledgeBase + JsonKnowledgeStore
├── niche_intelligence.py       # NicheAnalyzer + Niche
├── opportunity_engine.py       # OpportunityEngine + 7 OpportunityCriterion
├── creative_engine.py          # CreativeEngine + HeuristicCreativeGenerator
├── brand_engine.py             # BrandEngine + BrandProfile + JsonBrandStore
├── script_engine.py            # ScriptEngine + ScriptScene + HeuristicScriptGenerator
├── learning_engine.py          # LearningEngine + PerformanceMetrics + LearningProfile + JsonLearningStore
├── llm.py                     # LLM Provider — couche d'abstraction IA
├── llm_script_generator.py     # LLMScriptGenerator — script via LLM avec validation durée
├── visual_engine.py            # VisualEngine — Script → VisualPlan (par scène)
├── image_engine.py             # ImageEngine — VisualPlan → GeneratedImage
├── script_evaluator.py         # BaseEvaluator (ABC) + ScriptEvaluator (Sprint 20) — notation 8 critères
├── llm_script_evaluator.py     # LLMScriptEvaluator (Sprint 21) — LLM-as-judge, LLMScriptScore
├── rewrite_engine.py           # RewriteEngine (Sprint 22) — réécriture ciblée + garde la meilleure version
└── llm_image_generator.py      # LLMImageGenerator (Sprint 23) — prompts image pro via LLM
```

Docs complémentaires (non dupliquées ici) :
- `docs/content_understanding_architecture.md` — guide d'intégration LLM du Content Understanding Engine
- `docs/github-actions-guide.md` — configuration des secrets et du pipeline CI
- `docs/supabase_deployment.md` — déploiement du backend Supabase

Répertoires de données/sorties :
- `brands/*.json` — profils de marque (`business_fr`, `histoire_fr`, `ia_fr`)
- `data/learning/*.json` — `LearningProfile` persistés par marque
- `data/videos.csv` — snapshots collectés (fallback CSV)
- `outputs/scripts/` — scripts générés (heuristique et LLM)
- `reports/` — rapports Markdown/JSON horodatés (viralité, benchmarks, opportunités…)

### Dépendances entre modules

```
utils → (utilisé par : virality_engine, niche_intelligence, knowledge_engine, storage, collector, etc.)

models ← collector, storage, agents/*, virality_engine, content_understanding
virality_engine ← niche_intelligence (utilise VideoTimeline)
content_understanding ← knowledge_engine, opportunity_engine
knowledge_engine ← opportunity_engine
opportunity_engine ← creative_engine
brand_engine ← (indépendant — utilisé par creative_engine, script_engine)
script_engine ← opportunity_engine (via brief), creative_engine (via brief), brand_engine (via profile)
llm ← (indépendant — ne dépend d'aucun moteur)

Le LLM Provider ne dépend JAMAIS des moteurs du projet (httpx seulement).
Le Script Engine ne dépend JAMAIS des moteurs internes (ViralityEngine,
ContentUnderstandingEngine, KnowledgeEngine, Collector, Storage).
```

### Découplage volontaire (points forts)

- `StorageBackend` ne connaît **aucun** moteur (swap CSV/Supabase sans effet de bord)
- `ViralityEngine` ne connaît **pas** `ContentUnderstandingEngine`
- `ContentUnderstandingEngine` ne connaît **pas** `KnowledgeEngine`
- `KnowledgeEngine` ne connaît **pas** `OpportunityEngine`
- `CreativeEngine` ne lit **que** les `Opportunity` — pas les `VideoSnapshot`
- `BrandEngine` est un module **totalement indépendant**
- `LLM Provider` ne dépend d'aucun moteur (seulement `httpx`)

---

## 5. État d'avancement

### ✅ Complètement fonctionnel (Sprints 1-12)

| Module | Fonctionnalités |
|--------|----------------|
| `models.py` | `VideoSnapshot` — dataclass avec valeurs par défaut optionnelles |
| `utils.py` | `parse_iso_duration`, `safe_int`, `parse_dt`, `age_days`, `fmt_duration`, `fmt_views`, `csv_snapshots_to_timelines` |
| `collector.py` | `YouTubeCollector` — search + videos/list, parsing ISO, safe_int |
| `storage.py` | `CsvStorage`, `SupabaseStorage`, `FallbackStorage`, `build_storage()` |
| `agents/` | `BaseAgent` (ABC), `KeywordAgent`, `TrendingAgent` |
| `virality_engine.py` | `VideoTimeline`, `TemporalMetrics`, 6 `ScoringCriterion`, Time Engine adaptatif |
| `content_understanding.py` | 7 `BaseAnalyzer` (Topic, Language, ContentType, Audience, Emotion, Evergreen, Trend) |
| `knowledge_engine.py` | `KnowledgeBase`, `FrequencyDiscoverer`, `JsonKnowledgeStore` |
| `niche_intelligence.py` | `NicheAnalyzer` — regroupement par mot-clé |
| `opportunity_engine.py` | 7 `OpportunityCriterion` |
| `creative_engine.py` | `HeuristicCreativeGenerator` avec templates riches |
| `brand_engine.py` | `BrandProfile`, `JsonBrandStore`, validation |
| `brands/*.json` | 3 profils de marque complets (business_fr, histoire_fr, ia_fr) |
| `scripts/*.py` | 13 scripts couvrant tous les moteurs |
| `.github/workflows/collect.yml` | Pipeline CI complet (6h) |

### 🔶 Implémenté récemment

| Élément | Ce qui a été fait |
|---------|------------------|
| `src/llm.py` | Module complet (500+ lignes) : `LLMMessage`, `LLMResponse`, `LLMProvider` (ABC), `OpenAIProvider`, `GeminiProvider`, `ClaudeProvider`, `OllamaProvider` (stub), `DeepSeekProvider` (API réelle httpx), `build_llm()` factory, `estimate_cost()` |
| `scripts/test_llm.py` | Script de test : envoi "Réponds uniquement : OK", affichage provider/modèle/temps/coût/tokens |
| `tests/test_llm_groq.py` | 20 tests unitaires **GroqProvider** : appel API réel (mock httpx), pricing, tokens, erreurs |
| `tests/test_llm_deepseek.py` | 20 tests unitaires **DeepSeekProvider** : appel API réel (mock httpx), JSON mode, pricing, erreurs |
| `tests/test_llm.py` | 71 tests : messages, réponses, coûts, 3 providers (mock httpx), 2 stubs, factory (auto-détection, fallback, priorité), découplage complet |
| Priorité détection | `build_llm()` : **1. DEEPSEEK_API_KEY** > 2. ANTHROPIC_API_KEY > 3. OPENAI_API_KEY > 4. GEMINI_API_KEY |
| `src/llm_script_generator.py` | `LLMScriptGenerator` — génération de scripts via LLM avec validation durée : cible → estimation → écart < 50% → correction automatique via seconde génération |
| Tarifs officiels | `_MODEL_PRICING` : GPT-4o, Claude 3, Gemini 1.5, DeepSeek, Groq — prêt à l'emploi |
| Mode dégradé | Erreur API → LLMResponse avec finish_reason="error" (pas de crash) |
| Mode JSON | `json_mode=True` → response_format json_object (OpenAI/Gemini) |
| `src/learning_engine.py` | Module complet : `PerformanceMetrics`, `LearningSignal`, `LearningProfile`, `LearningStore` (ABC), `JsonLearningStore`, `LearningEngine` |
| `scripts/run_learning.py` | Script de démonstration : pipeline complet → 7 dimensions → classement |
| `tests/test_learning_engine.py` | 44 tests : métriques, signaux, profil, API (best_*), store, découplage |
| Boucle d'apprentissage | `record()` → 7 signaux (hook, angle, durée, CTA, style, émotion, format) → `build()` → profil queryable |
| Persistance | `JsonLearningStore` — save/load/list/delete, roundtrip JSON |
| `src/script_engine.py` | Module complet : `ScriptScene`, `Script`, `ScriptGenerator` (ABC), `HeuristicScriptGenerator`, `ScriptEngine` |
| `scripts/run_script.py` | Script de démonstration : pipeline Opportunity → Creative → Brand → Script |
| `scripts/run_script_benchmark.py` | Benchmark DeepSeek/Groq vs heuristique : `--provider deepseek\|groq`, `--llm-model`, `--top N`, rapports Markdown+JSON |
| `scripts/test_llm.py` | Script de test unifié : `--provider openai\|gemini\|claude\|groq\|deepseek`, `--all` |
| `tests/test_script_engine.py` | 27 tests : création, génération (5 angles), découplage, 108 total |
| Champs futurs moteurs | Chaque `ScriptScene` contient `image_prompt`, `visual_description`, `animation_notes`, `sound_effects` pour Visual/Animation/Video Engine |
| `src/utils.py` | Module d'utilitaires partagés — élimine la duplication de `parse_iso_duration`, `safe_int`, `fmt_duration`, `fmt_views`, `parse_dt`, `age_days` |
| `utils.csv_snapshots_to_timelines()` | Fonction partagée de chargement CSV — élimine la duplication entre `ViralityEngine._load_timelines()` et `NicheAnalyzer._load_timelines()` |
| `JsonKnowledgeStore` | Implémentation complète : `save()`, `load()`, `load_history()` + désérialisation |
| `models.py` | Valeurs par défaut pour `view_count`, `like_count`, `comment_count` (champs optionnels en dernier) |
| `src/visual_engine.py` | Module complet (721 lignes) : `VisualPlan`, `VisualScene`, `VisualGenerator` (ABC) — transforme un `Script` en plan visuel structuré. Ne dépend QUE de `Script`/`ScriptScene` |
| `src/image_engine.py` | Module complet (585 lignes) : `GeneratedImage`, `ImageGenerator` (ABC) — transforme un `VisualPlan` en prompt final prêt pour génération (Flux/SD/Fal/DALL-E à venir). Ne dépend QUE de `VisualPlan`/`VisualScene` |
| `src/script_evaluator.py` | Module complet (684 lignes, Sprint 20) : `ScriptEvaluator`, `ScriptScore` — note un `Script` sur 8 critères (/10 chacun : hook, curiosité, clarté, rythme, CTA, rétention, émotion, originalité), `compare()` heuristique vs LLM |
| `scripts/run_visual.py`, `run_image.py` | Scripts de démonstration Visual Engine / Image Engine |
| `scripts/audit_benchmark.py` | Audit des rapports de benchmark existants dans `reports/` |
| `src/script_evaluator.py` | Ajout de `BaseEvaluator` (ABC, Sprint 21) : interface commune à `ScriptEvaluator` et `LLMScriptEvaluator`, comparaison polymorphe via `.global_score`/`.name` |
| `src/llm_script_evaluator.py` | `LLMScriptEvaluator` (Sprint 21) — LLM-as-judge sur 8 critères (hook, curiosité, storytelling, rythme, clarté, CTA, rétention, viral) + `strengths`/`weaknesses`/`suggestions`. `global_score` toujours **recalculé côté serveur** (somme des 8 critères), jamais celui renvoyé par le LLM |
| `scripts/run_script_benchmark.py --llm-judge` | Évalue aussi tous les scripts avec `LLMScriptEvaluator` (optionnel, coûte des appels LLM en plus) et ajoute une section de comparaison Heuristique vs LLM-judge au rapport |
| `src/rewrite_engine.py` | `RewriteEngine` (Sprint 22) — réécrit uniquement hook/introduction/narration/conclusion/CTA à partir d'un `LLMScriptScore`. Sujet, marque, durée (`estimated_duration` + `duration_seconds` par scène) et nombre/ordre des scènes **jamais envoyés au LLM**, toujours recopiés depuis l'original ; toute divergence de structure invalide la réécriture. Pipeline : réécriture → ré-évaluation → garde la nouvelle version seulement si le score augmente strictement |
| `src/llm_image_generator.py` | `LLMImageGenerator` (Sprint 23) — implémente `ImageGenerator`, transforme `ScriptScene + VisualScene + BrandProfile` en prompt professionnel (sujet, composition, caméra, lumière, ambiance, couleurs, profondeur, style cinématographique, détail) pour Flux/SDXL/Midjourney. Dimensions/aspect_ratio/seed réutilisent les helpers déterministes de `HeuristicImageGenerator` (pas dupliqués). Fallback automatique vers l'heuristique si le LLM échoue |
| `scripts/run_image_prompt_benchmark.py` | Benchmark Prompt Heuristique vs Prompt LLM — score de couverture /8 par mots-clés sur les 8 dimensions attendues. Résultat mesuré (2026-07-08, DeepSeek) : **LLM 7.0/8 vs Heuristique 3.2/8** |

### 🔴 Reste à développer (priorité décroissante)

| Module | Sprint | Effort estimé |
|--------|--------|---------------|
| **LLM Script Generators** | 17 | `ClaudeScriptGenerator`, `GPTScriptGenerator` — utilisation de `build_llm()` (DeepSeek/Groq déjà couverts par `LLMScriptGenerator`) |
| **LLM Creative Generators** | 17 | `LLMCreativeGenerator` — hooks, angles, promesses via LLM |
| **Image Generation réelle** | 19+ | `FluxProvider`/`SdProvider`/`FalProvider`/`OpenAIProvider`/`GeminiProvider` — implémentations réelles pour `ImageEngine` (aujourd'hui : prompt professionnel prêt via `HeuristicImageGenerator`/`LLMImageGenerator`, mais aucun appel API de rendu d'image) |
| **Animation Engine** | 17+ | Génération d'animations à partir des `animation_notes` |
| **Video Engine** | 18+ | Montage vidéo automatisé |

| **OllamaProvider** | 17 | Implémentation REST réelle (llama3, mistral local) |
| **Learning → Creative bridge** | 17 | CreativeGenerator utilise `profile.best_hook()`, `profile.best_angle()` |
| **Learning → Script bridge** | 17 | ScriptGenerator utilise `profile.best_duration()`, `profile.best_style()` |
| **SupabaseLearningStore** | 17+ | Persistance Supabase des LearningProfile |
| **SupabaseKnowledgeStore** | 11+ | Persistance Supabase de la KnowledgeBase |
| **AprioriDiscoverer** | 11+ | Algorithme Apriori pour les combinaisons avancées |

| **Quota tracker** | — | Budget des appels API YouTube |
| **Retry decorator** | — | Résilience API (`tenacity`) |

---

## 6. Dette technique résolue

| Problème | Solution | Fichiers concernés |
|----------|----------|-------------------|
| `parse_iso_duration()` dupliqué 3× | Centralisé dans `src.utils.parse_iso_duration()` | `collector.py`, `trending_agent.py`, `experiment_strategies.py` |
| `safe_int()` dupliqué 4× | Centralisé dans `src.utils.safe_int()` | `collector.py`, `virality_engine.py`, `niche_intelligence.py`, `storage.py` |
| `parse_dt()` dupliqué 3× | Centralisé dans `src.utils.parse_dt()` | `virality_engine.py`, `experiment_strategies.py`, `analyze_collection.py` |
| `fmt_duration()` / `fmt_views()` dupliqué 4× | Centralisé dans `src.utils.fmt_duration()` / `fmt_views()` | `virality_engine.py`, `knowledge_engine.py`, `niche_intelligence.py`, divers scripts |
| `_load_timelines()` dupliqué 2× | Centralisé dans `utils.csv_snapshots_to_timelines()` | `virality_engine.py`, `niche_intelligence.py` |
| `_age_days()` dupliqué 2× | Centralisé dans `src.utils.age_days()` | `virality_engine.py`, `content_understanding.py` |
| Imports morts (`random`, `datetime`, `timezone`, `re`, `field`, `Optional`) dans 6 fichiers | Supprimés (détectés via `pyflakes`) | `image_engine.py`, `llm.py`, `niche_intelligence.py`, `script_engine.py`, `virality_engine.py`, `agents/trending_agent.py` |
| `knowledge_engine.py` : `Path` utilisé en annotation de type sans être importé (`NameError` latent) | Ajout de `from pathlib import Path` | `knowledge_engine.py` |
| Variables locales mortes `scenes_text`, `intro` dans `ScriptEvaluator.evaluate()` | Supprimées | `script_evaluator.py` |
| Fichier parasite `./=2.0.0` à la racine (résidu d'une commande `pip install "pkg>=2.0.0"` mal échappée) | Supprimé | racine du projet |

---

## 7. Tests

```
tests/
├── __init__.py
├── test_utils.py                # 35 tests — parse_iso_duration, safe_int, parse_dt,
│                                #           age_days, fmt_duration, fmt_views
├── test_models.py               # 4 tests — création, valeurs, égalité
├── test_virality_engine.py      # 26 tests — VideoTimeline, 6 ScoringCriterion,
│                                #           score composite
├── test_knowledge_engine.py     # 13 tests — KnowledgeFact, KnowledgeBase,
│                                #           FrequencyDiscoverer, KnowledgeEngine,
│                                #           JsonKnowledgeStore
├── test_script_engine.py        # 27 tests — ScriptScene, Script,
│                                #           HeuristicScriptGenerator,
│                                #           ScriptEngine, découplage
├── test_learning_engine.py      # 44 tests — PerformanceMetrics, LearningSignal,
│                                #           LearningProfile (best_* API),
│                                #           LearningEngine, JsonLearningStore,
│                                #           découplage
├── test_llm.py                  # 71 tests — LLMMessage, LLMResponse,
│                                #           estimate_cost, _get_env_key,
│                                #           OpenAIProvider (mock httpx),
│                                #           GeminiProvider (mock httpx),
│                                #           ClaudeProvider, OllamaProvider,
│                                #           DeepSeekProvider, build_llm(),
│                                #           découplage complet
├── test_llm_groq.py             # 20 tests — GroqProvider : appel API réel,
│                                #           pricing Groq, tokens, timeouts,
│                                #           modèles, erreurs 4xx/5xx
├── test_llm_deepseek.py         # 20 tests — DeepSeekProvider : appel API réel,
│                                #           pricing DeepSeek, JSON mode,
│                                #           erreurs, fallback, découplage
├── test_llm_script_generator.py # 25 tests — LLMScriptGenerator : prompt,
│                                #           extract_json, validate_json,
│                                #           generate, validation durée,
│                                #           fallback heuristique
├── test_visual_engine.py        # VisualPlan, VisualScene, VisualGenerator, découplage
├── test_image_engine.py         # GeneratedImage, ImageGenerator, découplage
├── test_script_evaluator.py     # ScriptEvaluator : 8 critères, compare(), BaseEvaluator
├── test_llm_script_evaluator.py # 35 tests — LLMScriptScore, LLMScriptEvaluator :
│                                #           prompt, extract_json, validate_json,
│                                #           recalcul serveur du global_score,
│                                #           échec sans provider, polymorphisme
│                                #           avec ScriptEvaluator, découplage
├── test_rewrite_engine.py       # 26 tests — RewriteEngine : prompt, extract_json,
│                                #           build_script_from_json (préserve
│                                #           sujet/marque/durée/scènes), décision
│                                #           garder/rejeter selon le score, résilience,
│                                #           découplage
└── test_llm_image_generator.py  # 28 tests — LLMImageGenerator : prompt,
                                 #           extract_json, validate_json,
                                 #           build_generated_image (réutilise les
                                 #           helpers heuristiques), fallback,
                                 #           interface ImageGenerator conservée
```

**Couverture actuelle :** 586 tests, ✅ 100 % passent (vérifié le 2026-07-08).

> Remarque : `pytest` n'est pas listé dans `requirements.txt` (il n'est présent
> que dans l'installation Python globale, pas dans `.venv`). Pour lancer les
> tests depuis `.venv`, installer d'abord `pytest` dedans, ou utiliser
> l'interpréteur global qui l'a déjà.

**Pour exécuter :**

```bash
cd c:\dev\youtube-trend-watcher
python -m pytest tests/ -v
```

---

## 8. Pipeline CI (GitHub Actions)

Le workflow `.github/workflows/collect.yml` s'exécute toutes les 6h (00:00, 06:00, 12:00, 18:00 UTC) :

1. **Collecte** : `scripts/test_agents.py` → KeywordAgent + TrendingAgent
2. **Analyse virale** : `scripts/run_virality.py` → ViralityEngine
3. **Rapport** : `scripts/generate_report.py` → Markdown horodaté
4. **Artefacts** : Sauvegarde des rapports (30 jours)

**Secrets requis :**

- `YOUTUBE_API_KEY` — clé API YouTube Data v3
- `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` — optionnels (repli CSV automatique)

---

## 9. Guide de contribution rapide

### Ajouter un nouveau critère de scoring viral

```python
from src.virality_engine import ScoringCriterion, VideoTimeline

class MyCriterion(ScoringCriterion):
    @property
    def name(self): return "my_criterion"
    def score(self, timeline: VideoTimeline) -> float:
        # Logique ici
        return 0.42
```

### Ajouter un nouveau backend de stockage

```python
from src.storage import StorageBackend
from src.models import VideoSnapshot

class MyStorage(StorageBackend):
    def save(self, snapshots: list[VideoSnapshot]) -> int: ...
    def load(self) -> list[VideoSnapshot]: ...
```

### Ajouter un nouvel analyseur sémantique

```python
from src.content_understanding import BaseAnalyzer, ContentProfile

class MyAnalyzer(BaseAnalyzer):
    @property
    def name(self): return "my_analyzer"
    def analyze(self, snapshot, timeline) -> dict:
        return {"my_field": "value"}
```

---

## 10. Commandes utiles

```bash
# Lancer tous les tests
python -m pytest tests/ -v

# Lancer un fichier de test spécifique
python -m pytest tests/test_virality_engine.py -v

# Lancer un test spécifique
python -m pytest tests/test_virality_engine.py::TestVelocityCriterion::test_normal_case -v

# Exécuter le pipeline complet de collecte
python scripts/test_agents.py

# Exécuter l'analyse virale
python scripts/run_virality.py

# Exécuter la génération de rapports
python scripts/generate_report.py

# Valider la connexion Supabase
python scripts/validate_supabase.py

# Explorer les stratégies de collecte
python scripts/experiment_strategies.py

# Construire la KnowledgeBase
python scripts/run_knowledge.py

# Analyser le Content Understanding Engine
python scripts/run_content_understanding.py

# Analyser les niches
python scripts/run_niche.py

# Détecter les opportunités avec marque
python scripts/run_opportunity.py

# Gérer les profils de marque
python scripts/run_brand.py

# Générer des briefs créatifs
python scripts/run_creative.py

# Exécuter le Script Engine
python scripts/run_script.py

# Générer un script via LLM (DeepSeek/Groq)
python scripts/run_llm_script.py

# Exécuter le Visual Engine (Script → VisualPlan)
python scripts/run_visual.py

# Exécuter l'Image Engine (VisualPlan → GeneratedImage)
python scripts/run_image.py

# Auditer les rapports de benchmark existants
python scripts/audit_benchmark.py

# Exécuter le Learning Engine (apprentissage des performances)
python scripts/run_learning.py

# Sauvegarder le profil d'apprentissage
python scripts/run_learning.py --output

# Tester le LLM Provider
python scripts/test_llm.py

# Tester tous les providers configurés
python scripts/test_llm.py --all

# Tester un provider spécifique
python scripts/test_llm.py --provider openai
python scripts/test_llm.py --provider deepseek

# Lancer le benchmark DeepSeek vs heuristique (top 20)
python scripts/run_script_benchmark.py --top 20 --brand ia_fr --provider deepseek

# Lancer le benchmark Groq vs heuristique
python scripts/run_script_benchmark.py --top 20 --brand ia_fr --provider groq

# Benchmark avec modèle spécifique
python scripts/run_script_benchmark.py --top 30 --brand histoire_fr --provider deepseek --llm-model deepseek-chat

# Benchmark avec évaluation LLM-judge en plus de l'heuristique (Sprint 21, appels LLM supplémentaires)
python scripts/run_script_benchmark.py --top 5 --brand ia_fr --provider deepseek --llm-judge

# Benchmark Prompt Heuristique vs Prompt LLM pour les images (Sprint 23)
python scripts/run_image_prompt_benchmark.py --brand ia_fr --provider deepseek
```
