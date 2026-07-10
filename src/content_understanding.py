"""
Content Understanding Engine v1 — Moteur de compréhension sémantique.

Architecture à analyseurs modulaires :
  - ContentProfile             : contrat de données immuable (sortie standard).
  - AnalysisResult             : résultat intermédiaire d'un analyseur atomique.
  - BaseAnalyzer               : interface commune à tous les analyseurs.
  - TopicAnalyzer              : sujet principal + sujets secondaires (taxonomie).
  - LanguageAnalyzer           : langue dominante (marqueurs linguistiques).
  - ContentTypeAnalyzer        : format du contenu (Short, Tutorial, Actualité…).
  - AudienceAnalyzer           : public cible (patterns thématiques).
  - EmotionAnalyzer            : tonalité émotionnelle (marqueurs d'affect).
  - EvergreenAnalyzer          : pérennité du contenu (0.0 éphémère → 1.0 intemporel).
  - TrendAnalyzer              : dynamique de tendance (0.0 stable → 1.0 viral).
  - ContentUnderstandingEngine : orchestrateur.

Extensibilité : substituer un analyseur heuristique par un analyseur LLM
sans modifier ni l'orchestrateur ni les moteurs consommateurs en aval
(Niche Intelligence V2, Opportunity Engine, Creative Engine, Script Engine).
"""

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.models import VideoSnapshot
from src.utils import age_days as _age_days
from src.virality_engine import VideoTimeline

logger = logging.getLogger(__name__)


# ── Structures de données ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContentProfile:
    """
    Profil sémantique d'une vidéo — contrat technique officiel d'échange de données.

    Aucun moteur en aval (Opportunity, Creative, Script) ne doit consommer
    les métadonnées brutes d'un VideoSnapshot ; il consomme exclusivement
    des instances de ContentProfile.
    """
    video_id: str
    primary_topic: str
    secondary_topics: List[str]
    language: str
    content_type: str
    target_audience: str
    emotion: str
    evergreen_score: float           # 0.0 (éphémère) → 1.0 (intemporel)
    trend_score: float               # 0.0 (stable) → 1.0 (viral actuel)
    confidence: float                # confiance globale de l'analyse (0.0–1.0)
    metadata: Dict[str, Any]         # données de débogage et d'extension par analyseur


@dataclass
class AnalysisResult:
    """Résultat intermédiaire produit par un BaseAnalyzer."""
    fields: Dict[str, Any]           # champs contribués au ContentProfile
    confidence: float                # confiance de cet analyseur (0.0–1.0)
    metadata: Dict[str, Any]         # informations complémentaires → ContentProfile.metadata


# ── Interface commune ─────────────────────────────────────────────────────────

class BaseAnalyzer(ABC):
    """
    Interface commune à tout analyseur du Content Understanding Engine.

    Contrat :
      - Entrée  : VideoSnapshot + VideoTimeline optionnel (données temporelles).
      - Sortie  : AnalysisResult avec les champs que cet analyseur sait remplir.

    Pour créer un analyseur LLM, sous-classer BaseAnalyzer et implémenter
    `analyze()` avec un appel API — l'orchestrateur l'accepte sans modification.
    Exemple de substitution :
        TopicAnalyzer (heuristique) → LLMTopicAnalyzer (Claude / GPT-4o)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifiant unique utilisé comme clé dans ContentProfile.metadata."""
        ...

    @abstractmethod
    def analyze(
        self,
        snapshot: VideoSnapshot,
        timeline: Optional[VideoTimeline] = None,
    ) -> AnalysisResult:
        """
        Analyse un snapshot et produit un AnalysisResult.

        Args:
            snapshot : dernier état connu de la vidéo (titre, stats, durée…).
            timeline : historique complet — None si un seul snapshot disponible.
        """
        ...


# ── Analyseurs heuristiques ───────────────────────────────────────────────────

class TopicAnalyzer(BaseAnalyzer):
    """
    Identifie le sujet principal et les sujets secondaires via une taxonomie
    de mots-clés appliquée sur le titre, la description et le mot-clé de collecte.
    """

    _TAXONOMY: Dict[str, List[str]] = {
        "Intelligence Artificielle": [
            "ia", "intelligence artificielle", "ai", "gpt", "llm", "chatgpt",
            "machine learning", "deep learning", "neural", "claude", "gemini",
            "copilot", "algorithme", "automatisation", "robot",
        ],
        "Business & Entrepreneuriat": [
            "business", "startup", "entreprise", "entrepreneur", "marketing",
            "management", "stratégie", "leadership", "productivité", "productivity",
        ],
        "Finance & Investissement": [
            "argent", "money", "finance", "investissement", "bourse", "bitcoin",
            "crypto", "économie", "trading", "revenue", "profit", "richesse", "budget",
        ],
        "Histoire & Culture": [
            "histoire", "historical", "history", "guerre", "révolution", "antique",
            "moyen age", "patrimoine", "civilisation", "archéologie",
        ],
        "Technologie": [
            "technologie", "technology", "tech", "innovation", "gadget", "smartphone",
            "ordinateur", "logiciel", "hardware", "software", "ios", "android", "iphone",
        ],
        "Gaming & Esports": [
            "gaming", "game", "gamer", "roblox", "minecraft", "fortnite",
            "playstation", "xbox", "esport", "stream", "gameplay",
        ],
        "Divertissement": [
            "funny", "humour", "drôle", "comedy", "entertainment", "prank",
            "challenge", "reaction", "vlog", "shorts",
        ],
        "Sports": [
            "football", "sport", "basket", "basketball", "tennis", "fifa", "ligue",
            "match", "championnat", "athlete", "messi", "ronaldo", "soccer",
        ],
        "Musique & Clips": [
            "music", "musique", "chanson", "album", "mv", "clip", "concert",
            "officiel", "official", "remix", "lyrics", "song",
        ],
        "Cinéma & Séries": [
            "film", "movie", "trailer", "cinéma", "série", "episode", "saison",
            "bande annonce", "review", "critique", "series",
        ],
        "Éducation & Science": [
            "éducation", "education", "apprendre", "tutorial", "how to", "science",
            "physique", "chimie", "biologie", "mathématiques", "cours", "explication",
        ],
        "Voyage & Lifestyle": [
            "voyage", "travel", "lifestyle", "aventure", "tourisme",
            "food", "cuisine", "restaurant", "découverte",
        ],
    }

    @property
    def name(self) -> str:
        return "topic"

    def analyze(self, snapshot: VideoSnapshot, timeline=None) -> AnalysisResult:
        text = _normalize(f"{snapshot.title} {snapshot.description} {snapshot.keyword}")
        words = set(re.findall(r"\b\w+\b", text))

        scores: Dict[str, int] = {}
        for topic, kws in self._TAXONOMY.items():
            count = 0
            for kw in kws:
                # Multi-word phrases: substring match; single words: whole-word match
                count += 1 if (" " in kw and kw in text) or (" " not in kw and kw in words) else 0
            if count > 0:
                scores[topic] = count

        if not scores:
            fallback = snapshot.keyword.strip().capitalize() or "Divers"
            return AnalysisResult(
                fields={"primary_topic": fallback, "secondary_topics": []},
                confidence=0.3,
                metadata={"topic_scores": {}},
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = ranked[0][0]
        secondary = [t for t, _ in ranked[1:4]]
        confidence = min(0.35 + ranked[0][1] * 0.12, 0.95)

        return AnalysisResult(
            fields={"primary_topic": primary, "secondary_topics": secondary},
            confidence=confidence,
            metadata={"topic_scores": {t: s for t, s in ranked[:5]}},
        )


class LanguageAnalyzer(BaseAnalyzer):
    """
    Détecte la langue principale par comptage de marqueurs linguistiques
    dans le titre et la description.
    """

    _MARKERS: Dict[str, List[str]] = {
        "fr": [
            "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
            "il", "elle", "ce", "qui", "que", "pour", "avec", "sur", "dans",
            "est", "sont", "pas", "mais", "plus", "aussi",
        ],
        "en": [
            "the", "a", "an", "is", "are", "how", "what", "why", "to", "of",
            "in", "for", "and", "or", "you", "this", "that", "with", "from",
            "has", "have", "was", "were", "its",
        ],
        "es": [
            "el", "los", "del", "que", "una", "con", "por", "para", "como",
            "pero", "más", "esto", "ese", "cuando", "gato",
        ],
        "pt": [
            "de", "que", "para", "uma", "com", "não", "por", "mais",
            "seu", "sua", "ser", "está",
        ],
    }

    @property
    def name(self) -> str:
        return "language"

    def analyze(self, snapshot: VideoSnapshot, timeline=None) -> AnalysisResult:
        text = _normalize(f"{snapshot.title} {snapshot.description}")
        words = set(re.findall(r"\b\w+\b", text))

        lang_scores = {
            lang: sum(1 for m in markers if m in words)
            for lang, markers in self._MARKERS.items()
        }

        best_lang = max(lang_scores, key=lambda k: lang_scores[k])
        best_score = lang_scores[best_lang]

        if best_score == 0:
            return AnalysisResult(
                fields={"language": "inconnu"},
                confidence=0.2,
                metadata={"lang_scores": lang_scores},
            )

        total = sum(lang_scores.values())
        confidence = min(0.40 + (best_score / max(total, 1)) * 0.50, 0.95)

        return AnalysisResult(
            fields={"language": best_lang},
            confidence=confidence,
            metadata={"lang_scores": lang_scores},
        )


class ContentTypeAnalyzer(BaseAnalyzer):
    """
    Classifie le format du contenu : Short, Tutorial, Actualité, Clip Musical,
    Gameplay, Analyse, Divertissement ou Standard.
    Priorité : durée (signal fort) → genre détecté par mots-clés.
    """

    _TYPE_KEYWORDS: Dict[str, List[str]] = {
        "Tutorial": [
            "tutorial", "how to", "guide", "apprendre", "étapes", "formation",
            "cours", "tuto", "méthode", "astuce", "tips",
        ],
        "Actualité": [
            "live", "breaking", "news", "actualité", "aujourd'hui", "urgent",
            "flash", "direct", "en direct", "bulletin",
        ],
        "Clip Musical": [
            "official video", "music video", "mv", "clip officiel",
            "lyric video", "audio", "official audio",
        ],
        "Gameplay": [
            "gameplay", "let's play", "walkthrough", "speedrun",
            "gaming session",
        ],
        "Analyse": [
            "analyse", "explication", "pourquoi", "raison", "comprendre",
            "deep dive", "breakdown", "réalité de",
        ],
        "Divertissement": [
            "prank", "challenge", "réaction", "reaction", "vlog",
            "humour", "drôle", "funny", "sketch",
        ],
    }

    @property
    def name(self) -> str:
        return "content_type"

    def analyze(self, snapshot: VideoSnapshot, timeline=None) -> AnalysisResult:
        if snapshot.duration_seconds <= 60:
            return AnalysisResult(
                fields={"content_type": "Short"},
                confidence=0.95,
                metadata={"duration_s": snapshot.duration_seconds},
            )

        text = _normalize(f"{snapshot.title} {snapshot.description}")
        for content_type, kws in self._TYPE_KEYWORDS.items():
            if any(kw in text for kw in kws):
                return AnalysisResult(
                    fields={"content_type": content_type},
                    confidence=0.70,
                    metadata={"duration_s": snapshot.duration_seconds},
                )

        label = "Long" if snapshot.duration_seconds > 600 else "Standard"
        return AnalysisResult(
            fields={"content_type": label},
            confidence=0.50,
            metadata={"duration_s": snapshot.duration_seconds},
        )


class AudienceAnalyzer(BaseAnalyzer):
    """
    Déduit le public cible à partir de marqueurs thématiques dans le texte
    (titre, description, mot-clé de collecte).
    """

    # Ordered by specificity — first match wins
    _PATTERNS: List[tuple[List[str], str, float]] = [
        (["roblox", "minecraft", "fortnite", "gaming", "gamer", "gameplay"], "Gamers", 0.80),
        (["business", "startup", "entrepreneur", "management", "stratégie", "leadership"], "Professionnels", 0.75),
        (["investissement", "bourse", "trading", "bitcoin", "crypto", "finance"], "Professionnels", 0.75),
        (["ia", "intelligence artificielle", "ai", "gpt", "llm", "technologie", "tech"], "Passionnés Tech", 0.70),
        (["tutorial", "how to", "apprendre", "cours", "formation", "étude", "explication"], "Étudiants", 0.70),
        (["histoire", "historique", "archéologie", "civilisation", "patrimoine"], "Curieux & Étudiants", 0.65),
        (["film", "série", "movie", "trailer", "cinéma"], "Cinéphiles", 0.60),
    ]

    @property
    def name(self) -> str:
        return "audience"

    def analyze(self, snapshot: VideoSnapshot, timeline=None) -> AnalysisResult:
        text = _normalize(f"{snapshot.title} {snapshot.description} {snapshot.keyword}")
        words = set(re.findall(r"\b\w+\b", text))

        for keywords, audience, confidence in self._PATTERNS:
            if any(kw in words if " " not in kw else kw in text for kw in keywords):
                return AnalysisResult(
                    fields={"target_audience": audience},
                    confidence=confidence,
                    metadata={},
                )

        return AnalysisResult(
            fields={"target_audience": "Grand public"},
            confidence=0.40,
            metadata={},
        )


class EmotionAnalyzer(BaseAnalyzer):
    """
    Identifie la tonalité émotionnelle dominante via des marqueurs textuels d'affect.
    """

    _EMOTION_KEYWORDS: Dict[str, List[str]] = {
        "Inspirant": [
            "motivation", "success", "réussir", "amazing", "incroyable",
            "inspiration", "champion", "victoire", "achieve",
        ],
        "Éducatif": [
            "tutorial", "apprendre", "comprendre", "explication", "guide",
            "how to", "cours", "formation", "étude",
        ],
        "Divertissant": [
            "funny", "drôle", "humour", "comedy", "prank", "fun",
            "lol", "hilarant", "sketch",
        ],
        "Nostalgique": [
            "histoire", "vintage", "ancien", "throwback", "nostalgie",
            "autrefois", "souvenir", "classique",
        ],
        "Sensationnel": [
            "choc", "shocking", "incroyable", "unbelievable", "fou", "crazy",
            "secret", "révèle", "découverte", "breaking", "scandale",
        ],
    }

    @property
    def name(self) -> str:
        return "emotion"

    def analyze(self, snapshot: VideoSnapshot, timeline=None) -> AnalysisResult:
        text = _normalize(f"{snapshot.title} {snapshot.description}")

        scores = {
            emotion: sum(1 for kw in kws if kw in text)
            for emotion, kws in self._EMOTION_KEYWORDS.items()
        }

        best = max(scores, key=lambda k: scores[k])
        if scores[best] == 0:
            return AnalysisResult(
                fields={"emotion": "Neutre"},
                confidence=0.40,
                metadata={"emotion_scores": scores},
            )

        confidence = min(0.40 + scores[best] * 0.15, 0.90)
        return AnalysisResult(
            fields={"emotion": best},
            confidence=confidence,
            metadata={"emotion_scores": scores},
        )


class EvergreenAnalyzer(BaseAnalyzer):
    """
    Estime la pérennité du contenu (0.0 éphémère → 1.0 intemporel).
    Score de base par thématique, modulé par le format et les signaux temporels.
    """

    # (keywords, base_score) — évalués dans l'ordre, premier match retenu
    _TOPIC_SCORES: List[tuple[List[str], float]] = [
        (["histoire", "historical", "history", "archéologie", "civilisation"], 0.85),
        (["tutorial", "apprendre", "guide", "how to", "formation"], 0.80),
        (["finance", "investissement", "économie", "budget", "richesse"], 0.70),
        (["business", "startup", "entrepreneur", "stratégie"], 0.65),
        (["ia", "intelligence artificielle", "ai", "gpt", "machine learning"], 0.60),
        (["technologie", "tech", "innovation", "gadget"], 0.55),
        (["voyage", "travel", "lifestyle", "cuisine"], 0.55),
        (["music", "musique", "chanson", "album"], 0.50),
        (["film", "movie", "série", "cinema"], 0.45),
        (["gaming", "game", "roblox", "minecraft", "gameplay"], 0.30),
        (["humour", "drôle", "funny", "comedy", "prank"], 0.25),
        (["sport", "football", "match", "championnat"], 0.20),
    ]

    @property
    def name(self) -> str:
        return "evergreen"

    def analyze(self, snapshot: VideoSnapshot, timeline=None) -> AnalysisResult:
        text = _normalize(f"{snapshot.title} {snapshot.description} {snapshot.keyword}")

        base = 0.50
        for keywords, score in self._TOPIC_SCORES:
            if any(kw in text for kw in keywords):
                base = score
                break

        if snapshot.duration_seconds <= 60:
            base -= 0.15
        elif snapshot.duration_seconds > 600:
            base += 0.10

        if any(kw in text for kw in ["news", "live", "today", "aujourd'hui", "urgent", "breaking"]):
            base -= 0.30

        evergreen_score = round(max(0.0, min(1.0, base)), 3)
        return AnalysisResult(
            fields={"evergreen_score": evergreen_score},
            confidence=0.65,
            metadata={"base_before_modifiers": base},
        )


class TrendAnalyzer(BaseAnalyzer):
    """
    Estime la dynamique de tendance actuelle (0.0 stable → 1.0 viral).
    Combine : âge de publication, source (trending chart), vélocité temporelle.
    """

    @property
    def name(self) -> str:
        return "trend"

    def analyze(
        self,
        snapshot: VideoSnapshot,
        timeline: Optional[VideoTimeline] = None,
    ) -> AnalysisResult:
        score = 0.0
        meta: Dict[str, Any] = {}

        # Signal 1 — âge de publication (poids 35 %)
        age_val = _age_days(snapshot.published_at)
        age_score = max(0.0, 1.0 - age_val / 30.0)
        score += age_score * 0.35
        meta["age_days"] = round(age_val, 1)

        # Signal 2 — validation algorithmique YouTube (poids 30 %)
        if snapshot.source == "trending":
            score += 0.30
            meta["source_bonus"] = True

        # Signal 3 — vélocité mesurée (poids 35 %, actif si ≥ 2 snapshots)
        if timeline and timeline.metrics:
            vph = timeline.metrics.views_per_hour
            velocity_score = math.log1p(vph) / math.log1p(10_000)
            score += min(velocity_score, 0.35)
            meta["views_per_hour"] = round(vph, 1)

        trend_score = round(min(score, 1.0), 3)
        confidence = 0.70 if (timeline and timeline.metrics) else 0.50

        return AnalysisResult(
            fields={"trend_score": trend_score},
            confidence=confidence,
            metadata=meta,
        )


# ── Analyseurs par défaut ─────────────────────────────────────────────────────

DEFAULT_ANALYZERS: List[BaseAnalyzer] = [
    TopicAnalyzer(),
    LanguageAnalyzer(),
    ContentTypeAnalyzer(),
    AudienceAnalyzer(),
    EmotionAnalyzer(),
    EvergreenAnalyzer(),
    TrendAnalyzer(),
]


# ── Moteur d'orchestration ────────────────────────────────────────────────────

class ContentUnderstandingEngine:
    """
    Orchestre les analyseurs et produit un ContentProfile par vidéo.

    Exemple minimal :
        engine = ContentUnderstandingEngine()
        profile = engine.analyze(snapshot)

    Avec analyseurs personnalisés (ex. remplacement par LLM) :
        engine = ContentUnderstandingEngine(analyzers=[
            LLMTopicAnalyzer(),      # remplace TopicAnalyzer
            LanguageAnalyzer(),
            ContentTypeAnalyzer(),
            AudienceAnalyzer(),
            EmotionAnalyzer(),
            EvergreenAnalyzer(),
            TrendAnalyzer(),
        ])
    """

    def __init__(self, analyzers: Optional[List[BaseAnalyzer]] = None) -> None:
        self._analyzers = analyzers if analyzers is not None else DEFAULT_ANALYZERS

    def analyze(
        self,
        snapshot: VideoSnapshot,
        timeline: Optional[VideoTimeline] = None,
    ) -> ContentProfile:
        """Analyse un VideoSnapshot et retourne son ContentProfile."""
        merged_fields: Dict[str, Any] = {}
        merged_metadata: Dict[str, Any] = {}
        confidences: List[float] = []

        for analyzer in self._analyzers:
            try:
                result = analyzer.analyze(snapshot, timeline)
                merged_fields.update(result.fields)
                merged_metadata[analyzer.name] = result.metadata
                confidences.append(result.confidence)
            except Exception as exc:
                logger.warning(
                    "Analyseur '%s' échoué sur %s : %s",
                    analyzer.name, snapshot.video_id, exc,
                )

        global_confidence = round(
            sum(confidences) / len(confidences) if confidences else 0.0, 3
        )

        return ContentProfile(
            video_id=snapshot.video_id,
            primary_topic=merged_fields.get("primary_topic", "Divers"),
            secondary_topics=merged_fields.get("secondary_topics", []),
            language=merged_fields.get("language", "inconnu"),
            content_type=merged_fields.get("content_type", "Standard"),
            target_audience=merged_fields.get("target_audience", "Grand public"),
            emotion=merged_fields.get("emotion", "Neutre"),
            evergreen_score=merged_fields.get("evergreen_score", 0.5),
            trend_score=merged_fields.get("trend_score", 0.5),
            confidence=global_confidence,
            metadata=merged_metadata,
        )

    def analyze_all(self, timelines: List[VideoTimeline]) -> List[ContentProfile]:
        """Analyse un ensemble de VideoTimeline et retourne leur ContentProfile."""
        profiles = [self.analyze(tl.latest, tl) for tl in timelines]
        logger.info("%d profils sémantiques générés.", len(profiles))
        return profiles


# ── Utilitaires (privés au module) ────────────────────────────────────────────

def _normalize(text: str) -> str:
    return text.lower()


# _age_days est importé de src.utils depuis le haut du fichier
