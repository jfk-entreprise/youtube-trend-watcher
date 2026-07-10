"""
Learning Engine v1 — Apprentissage des performances des contenus publiés.

Ferme la boucle du pipeline de création :
  Collecte → Analyse → Décision → Création → Publication → ← Learning Engine

Le Learning Engine reçoit les retours de performance (views, likes, etc.)
après publication et en extrait des LearningSignal exploitables par
les moteurs créatifs en amont.

Architecture à responsabilité unique :
  - Il ne fait QUE mesurer et mémoriser les performances.
  - Il ne modifie AUCUN autre moteur.
  - Il ne dépend que des données qu'on lui fournit.

Composants :
  - PerformanceMetrics   : données brutes de performance d'une vidéo publiée.
  - LearningSignal       : signal d'apprentissage (un couple cause→effet).
  - LearningProfile      : profil d'apprentissage complet pour un groupe.
  - LearningStore        : interface de persistance (ABC).
  - JsonLearningStore    : implémentation JSON sur disque (V1).
  - LearningEngine       : orchestrateur.

Boucle d'amélioration continue (Sprint 16+):
  learning_engine.get_best_hook(brand_id)  → CreativeGenerator utilise ce hook
  learning_engine.get_best_angle(niche)    → CreativeGenerator priorise cet angle
  learning_engine.get_best_duration()      → ScriptGenerator ajuste les durées

Découplage :
  - Le Learning Engine ne connaît AUCUN moteur interne.
  - Il ne manipule que : Opportunity, CreativeBrief, BrandProfile, Script.
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.brand_engine import BrandProfile
from src.creative_engine import CreativeBrief
from src.opportunity_engine import Opportunity
from src.script_engine import Script

logger = logging.getLogger(__name__)


# ── PerformanceMetrics ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Données brutes de performance d'une vidéo publiée.

    Ce sont les métriques qu'un créateur peut lire depuis YouTube Studio
    après la publication d'une vidéo.

    Champs :
      - views          : nombre total de vues à date de mesure.
      - likes          : nombre de likes.
      - comments       : nombre de commentaires.
      - retention      : rétention moyenne [0.0 – 1.0] (ex: 0.52 = 52%).
      - watch_time     : temps de visionnage total en heures.
      - impressions_ctr: taux de clic sur les impressions [0.0 – 1.0].
      - shares         : nombre de partages.
      - subscribers_gained : abonnés gagnés.
      - collected_at   : date ISO de la mesure (default_factory = now).

    Métriques calculées (propriétés) :
      - engagement_rate : ratio (likes + comments) / max(views, 1).
      - views_per_share : vues par partage.
    """

    video_id: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    retention: float = 0.0
    watch_time: float = 0.0
    impressions_ctr: float = 0.0
    shares: int = 0
    subscribers_gained: int = 0
    collected_at: str = ""

    @property
    def engagement_rate(self) -> float:
        """Ratio de likes + commentaires par vue."""
        return round((self.likes + self.comments) / max(self.views, 1), 4)

    @property
    def views_per_share(self) -> float:
        """Vues par partage — viralité sociale."""
        return round(self.views / max(self.shares, 1), 2)

    @property
    def performance_score(self) -> float:
        """Score composite simplifié [0.0 – 1.0] pour classer les performances.

        Pondérations heuristiques :
          - views (log)     : ×0.35 — portée brute
          - engagement      : ×0.25 — qualité de l'audience
          - rétention       : ×0.20 — qualité du contenu
          - CTR             : ×0.10 — efficacité miniature/titre
          - abonnés gagnés  : ×0.10 — croissance de chaîne
        """
        log_vues = math.log1p(self.views) / math.log1p(1_000_000)  # normalisation ~1M
        eng = min(self.engagement_rate * 10, 1.0)  # 10% engagement = 1.0
        ret = min(self.retention * 2.0, 1.0)  # 50% retention = 1.0
        ctr = min(self.impressions_ctr * 5.0, 1.0)  # 20% CTR = 1.0
        subs = math.log1p(self.subscribers_gained) / math.log1p(1000)  # normalisation ~1000

        score = (
            0.35 * log_vues
            + 0.25 * eng
            + 0.20 * ret
            + 0.10 * ctr
            + 0.10 * subs
        )
        return round(min(score, 1.0), 4)


# ── LearningSignal ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LearningSignal:
    """
    Signal d'apprentissage — un couple (dimension, valeur) → performance.

    Structure immutable qui lie une décision créative à son résultat mesuré.

    Exemples :
      - dimension="hook", value="Voici pourquoi X va tout changer",
        performance_score=0.72
      - dimension="angle", value="Liste", views=14520, retention=0.58
      - dimension="duration_seconds", value=482, engagement_rate=0.034

    Champs :
      - dimension          : dimension mesurée (hook, angle, duration, etc.).
      - value              : valeur de cette dimension (texte ou numérique).
      - performance_score  : score composite [0.0 – 1.0].
      - views, likes, etc. : métriques brutes pour analyse fine.
      - brand_id           : identifiant de la marque (pour filtrage).
      - niche              : niche de la vidéo.
      - opportunity_id     : id de l'opportunité source.
      - metadata           : données extensibles.
    """

    dimension: str
    value: str
    performance_score: float

    views: int = 0
    likes: int = 0
    comments: int = 0
    retention: float = 0.0

    brand_id: str = ""
    niche: str = ""
    opportunity_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── LearningProfile ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LearningProfile:
    """
    Profil d'apprentissage complet — agrège tous les signaux pour
    répondre aux questions du système.

    Le LearningProfile est le contrat de sortie du Learning Engine.
    Les moteurs créatifs (CreativeEngine, ScriptEngine) le consomment
    pour améliorer leurs décisions.

    Méthodes de requête (API publique) :
      - best_hook(niche=None, brand_id=None)     → (hook, score)
      - best_duration(niche=None)                → (seconds, score)
      - best_angle(niche=None, brand_id=None)    → (angle_name, score)
      - best_cta(brand_id=None)                  → (cta_text, score)
      - best_style(brand_id=None)                → (style, score)
      - best_emotion(niche=None)                 → (emotion, score)
      - best_format(niche=None)                  → (format, score)

    Chaque méthode peut être filtrée par niche ou marque.

    Stockage interne :
      - signals      : liste brute de tous les LearningSignal.
      - _cache       : dictionnaire de scores moyens par (dimension, valeur).
    """

    brand_id: str
    signals: List[LearningSignal]
    _cache: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Construit le cache de scores à l'initialisation."""
        cache: Dict[str, Dict[str, List[float]]] = {}
        for sig in self.signals:
            dim = sig.dimension
            val = sig.value
            if dim not in cache:
                cache[dim] = {}
            if val not in cache[dim]:
                cache[dim][val] = []
            cache[dim][val].append(sig.performance_score)
        # On utilise object.__setattr__ car frozen=True
        object.__setattr__(self, "_cache", cache)

    # ── API publique ──────────────────────────────────────────────────────────

    def best_hook(
        self,
        niche: Optional[str] = None,
        brand_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Retourne le hook le plus performant, filtré optionnellement.

        Returns:
            (hook_text, avg_performance_score)
        """
        return self._best("hook", niche, brand_id)

    def best_duration(self, niche: Optional[str] = None) -> Tuple[str, float]:
        """
        Retourne la durée la plus performante.

        Returns:
            ("{seconds}", avg_score)
        """
        return self._best("duration_seconds", niche)

    def best_angle(
        self,
        niche: Optional[str] = None,
        brand_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Retourne l'angle le plus performant."""
        return self._best("angle", niche, brand_id)

    def best_cta(self, brand_id: Optional[str] = None) -> Tuple[str, float]:
        """Retourne le CTA le plus performant."""
        return self._best("cta", brand_id=brand_id)

    def best_style(self, brand_id: Optional[str] = None) -> Tuple[str, float]:
        """Retourne le style rédactionnel le plus performant."""
        return self._best("style", brand_id=brand_id)

    def best_emotion(self, niche: Optional[str] = None) -> Tuple[str, float]:
        """Retourne l'émotion la plus performante."""
        return self._best("emotion", niche)

    def best_format(self, niche: Optional[str] = None) -> Tuple[str, float]:
        """Retourne le format de contenu le plus performant."""
        return self._best("format", niche)

    # ── Statistiques ──────────────────────────────────────────────────────────

    def top_hooks(self, n: int = 3) -> List[Tuple[str, float]]:
        """Retourne les n meilleurs hooks."""
        return self._top_n("hook", n)

    def top_angles(self, n: int = 3) -> List[Tuple[str, float]]:
        """Retourne les n meilleurs angles."""
        return self._top_n("angle", n)

    def top_ctas(self, n: int = 3) -> List[Tuple[str, float]]:
        """Retourne les n meilleurs CTA."""
        return self._top_n("cta", n)

    @property
    def total_signals(self) -> int:
        return len(self.signals)

    @property
    def dimensions(self) -> List[str]:
        """Liste des dimensions disponibles dans le profil."""
        return sorted(self._cache.keys())

    def summary(self) -> Dict[str, Any]:
        """Résumé lisible du profil d'apprentissage."""
        return {
            "brand_id": self.brand_id,
            "total_signals": self.total_signals,
            "dimensions": self.dimensions,
            "best_hook": self.best_hook(),
            "best_angle": self.best_angle(),
            "best_duration": self.best_duration(),
            "best_cta": self.best_cta(),
            "best_style": self.best_style(),
            "best_emotion": self.best_emotion(),
            "best_format": self.best_format(),
        }

    # ── Méthodes internes ─────────────────────────────────────────────────────

    def _best(
        self,
        dimension: str,
        niche: Optional[str] = None,
        brand_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """Trouve la meilleure valeur pour une dimension donnée."""
        candidates = self._get_candidates(dimension, niche, brand_id)
        if not candidates:
            return ("", 0.0)

        # candidates est un dict {valeur: score_moyen}
        best_val = max(candidates, key=candidates.get)  # type: ignore
        return (best_val, candidates[best_val])

    def _top_n(
        self,
        dimension: str,
        n: int = 3,
        niche: Optional[str] = None,
        brand_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Retourne les n meilleures valeurs pour une dimension."""
        candidates = self._get_candidates(dimension, niche, brand_id)
        if not candidates:
            return []

        # candidates est un dict {valeur: score_moyen}
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def _get_candidates(
        self,
        dimension: str,
        niche: Optional[str] = None,
        brand_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Calcule les scores moyens pour chaque valeur d'une dimension.

        Applique les filtres niche et brand_id si fournis.
        """
        scores: Dict[str, List[float]] = {}

        for sig in self.signals:
            if sig.dimension != dimension:
                continue
            if niche is not None and sig.niche != niche:
                continue
            if brand_id is not None and sig.brand_id != brand_id:
                continue

            val = sig.value
            if val not in scores:
                scores[val] = []
            scores[val].append(sig.performance_score)

        return {val: round(sum(s) / len(s), 4) for val, s in scores.items()}


# ── LearningStore ────────────────────────────────────────────────────────────

class LearningStore(ABC):
    """
    Interface abstraite de persistance des LearningProfile.

    Implémentations prévues :
      - JsonLearningStore    : fichiers JSON locaux (V1).
      - SupabaseLearningStore : table Supabase (Sprint 16+).
    """

    @abstractmethod
    def save(self, profile: LearningProfile) -> None:
        """Persiste un profil d'apprentissage."""
        ...

    @abstractmethod
    def load(self, brand_id: str) -> Optional[LearningProfile]:
        """Charge un profil par identifiant de marque. None si inexistant."""
        ...

    @abstractmethod
    def list_brands(self) -> List[str]:
        """Retourne la liste des brand_id disponibles."""
        ...

    @abstractmethod
    def delete(self, brand_id: str) -> bool:
        """Supprime un profil. True si supprimé."""
        ...


# ── JsonLearningStore ────────────────────────────────────────────────────────

class JsonLearningStore(LearningStore):
    """
    Persistance JSON sur disque local.

    Convention : un fichier <brand_id>_learning.json par profil.
    Chargement dynamique — un nouveau fichier est immédiatement disponible.
    """

    def __init__(self, directory: Any) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    # ── Sauvegarde ────────────────────────────────────────────────────────────

    def save(self, profile: LearningProfile) -> None:
        path = self._dir / f"{profile.brand_id}_learning.json"
        data = _profile_to_dict(profile)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("LearningProfile '%s' sauvegardé → %s", profile.brand_id, path)

    # ── Chargement ────────────────────────────────────────────────────────────

    def load(self, brand_id: str) -> Optional[LearningProfile]:
        path = self._dir / f"{brand_id}_learning.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _profile_from_dict(data)
        except Exception as exc:
            logger.error("Erreur lecture '%s' : %s", path, exc)
            return None

    # ── Liste ─────────────────────────────────────────────────────────────────

    def list_brands(self) -> List[str]:
        brands: List[str] = []
        for path in sorted(self._dir.glob("*_learning.json")):
            brand_id = path.stem.replace("_learning", "")
            brands.append(brand_id)
        return brands

    # ── Suppression ───────────────────────────────────────────────────────────

    def delete(self, brand_id: str) -> bool:
        path = self._dir / f"{brand_id}_learning.json"
        if path.exists():
            path.unlink()
            logger.debug("LearningProfile '%s' supprimé.", brand_id)
            return True
        return False


# ── Sérialisation ────────────────────────────────────────────────────────────

def _signal_to_dict(sig: LearningSignal) -> Dict[str, Any]:
    return {
        "dimension": sig.dimension,
        "value": sig.value,
        "performance_score": sig.performance_score,
        "views": sig.views,
        "likes": sig.likes,
        "comments": sig.comments,
        "retention": sig.retention,
        "brand_id": sig.brand_id,
        "niche": sig.niche,
        "opportunity_id": sig.opportunity_id,
        "metadata": dict(sig.metadata),
    }


def _signal_from_dict(data: Dict[str, Any]) -> LearningSignal:
    return LearningSignal(
        dimension=str(data["dimension"]),
        value=str(data["value"]),
        performance_score=float(data.get("performance_score", 0.0)),
        views=int(data.get("views", 0)),
        likes=int(data.get("likes", 0)),
        comments=int(data.get("comments", 0)),
        retention=float(data.get("retention", 0.0)),
        brand_id=str(data.get("brand_id", "")),
        niche=str(data.get("niche", "")),
        opportunity_id=str(data.get("opportunity_id", "")),
        metadata=dict(data.get("metadata", {})),
    )


def _profile_to_dict(profile: LearningProfile) -> Dict[str, Any]:
    return {
        "brand_id": profile.brand_id,
        "signals": [_signal_to_dict(s) for s in profile.signals],
    }


def _profile_from_dict(data: Dict[str, Any]) -> LearningProfile:
    signals = [_signal_from_dict(s) for s in data.get("signals", [])]
    return LearningProfile(
        brand_id=str(data["brand_id"]),
        signals=signals,
    )


# ── LearningEngine ────────────────────────────────────────────────────────────

class LearningEngine:
    """
    Orchestrateur du Learning Engine.

    Reçoit les données de performance après publication et construit
    un LearningProfile capable de répondre à :
      - Quel hook fonctionne le mieux ?
      - Quelle durée est optimale ?
      - Quel angle performe le mieux ?
      - Quel CTA convertit le plus ?

    Exemple minimal :
        engine = LearningEngine()
        signals = engine.record(opportunity, brief, brand, script, metrics)
        profile = engine.build(signals)

    Avec persistance :
        engine = LearningEngine(store=JsonLearningStore("data/learning"))
        engine.save("ia_fr", profile)
        profile = engine.load("ia_fr")

    Boucle d'amélioration :
        best_hook, score = profile.best_hook(brand_id="ia_fr")
        best_angle, score = profile.best_angle(niche="IA")
        best_dur, score = profile.best_duration()
    """

    def __init__(self, store: Optional[LearningStore] = None) -> None:
        self._store = store

    # ── Enregistrement d'une performance ──────────────────────────────────────

    def record(
        self,
        opportunity: Opportunity,
        creative_brief: CreativeBrief,
        brand_profile: BrandProfile,
        script: Script,
        metrics: PerformanceMetrics,
    ) -> List[LearningSignal]:
        """
        Transforme une vidéo publiée en signaux d'apprentissage.

        Extrait les dimensions suivantes depuis les données fournies :
          - hook             : texte du hook
          - angle            : angle narratif
          - duration_seconds : durée du script
          - cta              : texte du CTA
          - style            : style rédactionnel
          - emotion          : tonalité émotionnelle
          - format           : format de contenu

        Args:
            opportunity   : l'opportunité source.
            creative_brief: le brief créatif utilisé.
            brand_profile : le profil de marque utilisé.
            script        : le script généré.
            metrics       : les métriques de performance (post-publication).

        Returns:
            Liste de LearningSignal, un par dimension.
        """
        signals: List[LearningSignal] = []

        score = metrics.performance_score
        brand_id = brand_profile.id
        niche = opportunity.niche
        opp_id = opportunity.source_video_id

        # Dimensions extraites
        dimensions = self._extract_dimensions(
            creative_brief, brand_profile, script,
        )

        for dim, val in dimensions.items():
            sig = LearningSignal(
                dimension=dim,
                value=str(val),
                performance_score=score,
                views=metrics.views,
                likes=metrics.likes,
                comments=metrics.comments,
                retention=metrics.retention,
                brand_id=brand_id,
                niche=niche,
                opportunity_id=opp_id,
                metadata={
                    "engagement_rate": metrics.engagement_rate,
                    "watch_time": metrics.watch_time,
                    "impressions_ctr": metrics.impressions_ctr,
                    "shares": metrics.shares,
                    "subscribers_gained": metrics.subscribers_gained,
                },
            )
            signals.append(sig)

        logger.info(
            "Enregistrement Learning : %d signaux pour '%s' (score: %.4f, niche: %s)",
            len(signals),
            opp_id[:8],
            score,
            niche,
        )
        return signals

    # ── Construction d'un profil ─────────────────────────────────────────────

    def build(
        self,
        signals: List[LearningSignal],
        brand_id: str = "default",
    ) -> LearningProfile:
        """
        Construit un LearningProfile à partir d'une liste de signaux.

        Le profil est prêt à répondre aux requêtes best_hook, best_angle, etc.

        Args:
            signals : signaux d'apprentissage bruts.
            brand_id: identifiant de la marque associée.

        Returns:
            LearningProfile avec cache de scores pré-calculé.
        """
        if not signals:
            logger.warning("Aucun signal fourni — profil vide créé.")
            return LearningProfile(brand_id=brand_id, signals=[])

        profile = LearningProfile(brand_id=brand_id, signals=signals)

        logger.info(
            "LearningProfile '%s' : %d signaux, %d dimensions (%s)",
            brand_id,
            profile.total_signals,
            len(profile.dimensions),
            ", ".join(profile.dimensions),
        )
        return profile

    # ── Persistance ─────────────────────────────────────────────────────────

    def save(self, brand_id: str, profile: LearningProfile) -> None:
        """Sauvegarde un profil via le store configuré."""
        if self._store is None:
            logger.warning("Aucun store configuré — profil non sauvegardé.")
            return
        self._store.save(profile)
        logger.info("LearningProfile '%s' sauvegardé.", brand_id)

    def load(self, brand_id: str) -> Optional[LearningProfile]:
        """Charge un profil depuis le store. None si inexistant."""
        if self._store is None:
            logger.warning("Aucun store configuré.")
            return None
        profile = self._store.load(brand_id)
        if profile is None:
            logger.info("Aucun LearningProfile pour '%s'.", brand_id)
        return profile

    # ── Extraction des dimensions ───────────────────────────────────────────

    @staticmethod
    def _extract_dimensions(
        brief: CreativeBrief,
        brand: BrandProfile,
        script: Script,
    ) -> Dict[str, str]:
        """Extrait les dimensions à mesurer depuis les entrées créatives."""
        dims: Dict[str, str] = {}

        # Depuis le script
        dims["hook"] = script.hook
        dims["duration_seconds"] = str(script.estimated_duration)
        dims["cta"] = script.call_to_action
        dims["style"] = script.style

        # Depuis le brief
        dims["angle"] = brief.angle
        dims["emotion"] = brief.emotion
        dims["format"] = brief.format

        return dims
