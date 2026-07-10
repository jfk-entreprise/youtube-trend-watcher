# Content Understanding Engine — Architecture & Guide d'Intégration LLM

## 1. Vue d'ensemble

Le **Content Understanding Engine** convertit les métadonnées brutes d'une `VideoSnapshot` en un `ContentProfile` sémantiquement enrichi. Ce profil est le **contrat technique officiel** entre la couche de collecte et les moteurs d'intelligence en aval.

```
VideoSnapshot (brut)
        │
        ▼
ContentUnderstandingEngine
  ├── TopicAnalyzer
  ├── LanguageAnalyzer
  ├── ContentTypeAnalyzer
  ├── AudienceAnalyzer
  ├── EmotionAnalyzer
  ├── EvergreenAnalyzer
  └── TrendAnalyzer
        │
        ▼
ContentProfile (contrat d'échange)
        │
        ├──► Niche Intelligence V2
        ├──► Opportunity Engine
        ├──► Creative Engine
        └──► Script Engine
```

---

## 2. Inventaire des classes et interfaces

### Structures de données

| Classe | Type | Rôle |
|--------|------|------|
| `ContentProfile` | `@dataclass(frozen=True)` | Profil sémantique immuable — sortie standard du moteur |
| `AnalysisResult` | `@dataclass` | Résultat intermédiaire d'un analyseur atomique |

### Interface commune

| Classe | Type | Rôle |
|--------|------|------|
| `BaseAnalyzer` | `ABC` | Contrat d'interface pour tout analyseur (heuristique ou LLM) |

### Analyseurs heuristiques (V1)

| Classe | Champ(s) produit(s) | Signal principal |
|--------|---------------------|-----------------|
| `TopicAnalyzer` | `primary_topic`, `secondary_topics` | Taxonomie de 12 catégories × mots-clés |
| `LanguageAnalyzer` | `language` | Comptage de marqueurs linguistiques (fr/en/es/pt) |
| `ContentTypeAnalyzer` | `content_type` | Durée + mots-clés de genre |
| `AudienceAnalyzer` | `target_audience` | Patterns thématiques ordonnés par spécificité |
| `EmotionAnalyzer` | `emotion` | Marqueurs d'affect (5 tonalités + Neutre) |
| `EvergreenAnalyzer` | `evergreen_score` | Thématique de base + modificateurs format/actualité |
| `TrendAnalyzer` | `trend_score` | Âge + source trending + vélocité temporelle (si disponible) |

### Orchestrateur

| Classe | Méthodes publiques | Rôle |
|--------|--------------------|------|
| `ContentUnderstandingEngine` | `analyze(snapshot, timeline)` → `ContentProfile` | Analyse une vidéo |
| | `analyze_all(timelines)` → `list[ContentProfile]` | Analyse en batch |

---

## 3. Schéma d'architecture logicielle

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ContentUnderstandingEngine                        │
│                                                                      │
│   analyzers: list[BaseAnalyzer]  ◄── injectés dans le constructeur  │
│                                                                      │
│   analyze(snapshot, timeline):                                       │
│     for analyzer in analyzers:                                       │
│       result = analyzer.analyze(snapshot, timeline)                  │
│       merged_fields.update(result.fields)           ─┐              │
│       merged_metadata[name] = result.metadata        │ fusion        │
│       confidences.append(result.confidence)         ─┘              │
│     return ContentProfile(**merged_fields, confidence=mean(...))     │
└──────────────────────────────────────────────────────────────────────┘
         │                              │
         │ hérite de BaseAnalyzer       │ hérite de BaseAnalyzer
         ▼                              ▼
┌──────────────────┐          ┌──────────────────────┐
│  TopicAnalyzer   │          │  LLMTopicAnalyzer    │  ← V2 (à venir)
│  (heuristique)   │          │  (Claude / GPT-4o)   │
│                  │          │                      │
│  name: "topic"   │          │  name: "topic"       │
│  analyze() → ... │          │  analyze() → ...     │
└──────────────────┘          └──────────────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
              AnalysisResult
                fields: {"primary_topic": "IA", ...}
                confidence: 0.82
                metadata: {"topic_scores": {...}}
```

### Flux de données détaillé

```
1. VideoSnapshot  ────────────────────────────────────────────────────►
                                                                       │
2.               ┌─ TopicAnalyzer ──────► AnalysisResult ─────────────┤
                 ├─ LanguageAnalyzer ───► AnalysisResult ─────────────┤
                 ├─ ContentTypeAnalyzer ► AnalysisResult ─────────────┤ merge
                 ├─ AudienceAnalyzer ───► AnalysisResult ─────────────┤ fields
                 ├─ EmotionAnalyzer ────► AnalysisResult ─────────────┤
                 ├─ EvergreenAnalyzer ──► AnalysisResult ─────────────┤
                 └─ TrendAnalyzer ──────► AnalysisResult ─────────────┤
                                                                       │
3.                                                         ContentProfile (frozen)
```

---

## 4. Guide d'intégration d'un analyseur LLM

### 4.1 Principe

L'orchestrateur ne connaît que `BaseAnalyzer`. Substituer `TopicAnalyzer` par un analyseur LLM se résume à :
1. Créer une sous-classe de `BaseAnalyzer`.
2. Implémenter `name` et `analyze()`.
3. L'injecter dans le constructeur de `ContentUnderstandingEngine`.

**Aucune modification** de l'orchestrateur, du `ContentProfile`, ni des moteurs consommateurs.

---

### 4.2 Exemple : `LLMTopicAnalyzer` (Claude via l'API Anthropic)

```python
# src/analyzers/llm_topic_analyzer.py
import anthropic
import json
from src.content_understanding import BaseAnalyzer, AnalysisResult
from src.models import VideoSnapshot
from src.virality_engine import VideoTimeline
from typing import Optional

PROMPT_TEMPLATE = """
Analyse cette vidéo YouTube et classe-la.

Titre : {title}
Description (extrait) : {description}

Réponds UNIQUEMENT en JSON valide avec ce schéma :
{{
  "primary_topic": "<sujet principal en français>",
  "secondary_topics": ["<sujet 2>", "<sujet 3>"],
  "confidence": <float entre 0.0 et 1.0>
}}
"""

class LLMTopicAnalyzer(BaseAnalyzer):
    """
    Analyseur de sujet basé sur Claude — remplace TopicAnalyzer en V2.
    Identique à l'interface : aucune modification en aval requise.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self._client = anthropic.Anthropic()
        self._model = model

    @property
    def name(self) -> str:
        return "topic"   # même clé que TopicAnalyzer → remplacement transparent

    def analyze(
        self,
        snapshot: VideoSnapshot,
        timeline: Optional[VideoTimeline] = None,
    ) -> AnalysisResult:
        prompt = PROMPT_TEMPLATE.format(
            title=snapshot.title,
            description=snapshot.description[:300],
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        parsed = json.loads(raw)

        return AnalysisResult(
            fields={
                "primary_topic": parsed.get("primary_topic", "Divers"),
                "secondary_topics": parsed.get("secondary_topics", []),
            },
            confidence=float(parsed.get("confidence", 0.7)),
            metadata={"model": self._model, "raw_response": raw},
        )
```

### 4.3 Activation de l'analyseur LLM

```python
# scripts/run_content_understanding_v2.py
from src.content_understanding import (
    ContentUnderstandingEngine,
    LanguageAnalyzer, ContentTypeAnalyzer, AudienceAnalyzer,
    EmotionAnalyzer, EvergreenAnalyzer, TrendAnalyzer,
)
from src.analyzers.llm_topic_analyzer import LLMTopicAnalyzer

engine = ContentUnderstandingEngine(analyzers=[
    LLMTopicAnalyzer(),          # ← LLM remplace l'heuristique
    LanguageAnalyzer(),          # ← inchangés
    ContentTypeAnalyzer(),
    AudienceAnalyzer(),
    EmotionAnalyzer(),
    EvergreenAnalyzer(),
    TrendAnalyzer(),
])

# Utilisation identique — ContentProfile inchangé
profile = engine.analyze(snapshot, timeline)
```

### 4.4 Évolutions progressives possibles

| Étape | Analyseur | Valeur ajoutée |
|-------|-----------|----------------|
| V1 | `TopicAnalyzer` (heuristique) | Rapide, aucun coût API |
| V2 | `LLMTopicAnalyzer` (Claude Haiku) | Précision sémantique, multilingue |
| V3 | `HybridTopicAnalyzer` | LLM pour les cas ambigus, heuristique en fallback |

---

## 5. Conventions de développement

### Ajouter un nouvel analyseur

```python
class MySentimentAnalyzer(BaseAnalyzer):
    """Exemple d'analyseur personnalisé."""

    @property
    def name(self) -> str:
        return "sentiment"   # clé unique dans ContentProfile.metadata

    def analyze(self, snapshot: VideoSnapshot, timeline=None) -> AnalysisResult:
        # ... logique ...
        return AnalysisResult(
            fields={"sentiment": "positif"},   # nouveau champ dans metadata
            confidence=0.75,
            metadata={"detail": "..."},
        )
```

> **Note :** Pour ajouter `sentiment` à `ContentProfile`, il suffit d'ajouter le champ au dataclass. Les analyseurs existants ne sont pas affectés.

### Désactiver un analyseur

```python
engine = ContentUnderstandingEngine(analyzers=[
    a for a in DEFAULT_ANALYZERS if a.name != "emotion"
])
```

### Tester un analyseur en isolation

```python
from src.content_understanding import TopicAnalyzer, VideoSnapshot

analyzer = TopicAnalyzer()
# snapshot de test minimal
snap = VideoSnapshot(
    video_id="test", title="Comment investir en bourse", description="",
    channel_id="", channel_title="", published_at="2026-01-01T00:00:00Z",
    duration_iso="PT10M", duration_seconds=600,
    view_count=5000, like_count=200, comment_count=30,
    keyword="finance", source="keyword",
)
result = analyzer.analyze(snap)
print(result.fields)         # {"primary_topic": "Finance & Investissement", ...}
print(result.confidence)     # 0.59
```
