"""
Knowledge Engine v1 — Moteur d'apprentissage marché.

Transforme une liste de ContentProfile en une KnowledgeBase structurée.
Aucune lecture directe de VideoSnapshot — le moteur consomme uniquement
des ContentProfile (contrat établi au Sprint 9).

Composants :
  - KnowledgeFact          : unité de connaissance atomique et immuable.
  - KnowledgeBase          : conteneur des connaissances agrégées + accesseurs.
  - CombinationDiscoverer  : interface pour les algorithmes de découverte.
  - FrequencyDiscoverer    : V1 — co-occurrences par comptage de fréquences.
  - KnowledgeStore         : interface de persistance (Supabase — Sprint 11).
  - KnowledgeEngine        : orchestrateur.

Extensibilité :
  - Nouvel algorithme    : sous-classer CombinationDiscoverer.
  - Persistance          : sous-classer KnowledgeStore et implémenter save/load.
  - Nouveau type de fait : ajouter un champ dans KnowledgeBase + _compute_*().
  - Analyse temporelle   : passer plusieurs KnowledgeBase à un comparateur.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median as stat_median
from typing import Any, Callable, Optional

from src.content_understanding import ContentProfile
from src.utils import fmt_duration as _fmt_dur

logger = logging.getLogger(__name__)


# ── KnowledgeFact ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KnowledgeFact:
    """
    Unité atomique de connaissance découverte automatiquement.

    Immuable (frozen=True) — les moteurs consommateurs ne peuvent que lire.
    Le champ `value` peut contenir n'importe quel type sérialisable (dict, float, str…).
    Note : hashing non supporté si `value` contient un dict ou une liste.
    """
    name: str                    # identifiant unique (ex: "topic.IA")
    description: str             # libellé lisible
    value: Any                   # donnée principale (dict, float, str…)
    confidence: float            # confiance de la mesure (0.0–1.0)
    observations: int            # nombre de ContentProfile ayant contribué
    updated_at: datetime         # horodatage de la mise à jour
    metadata: dict[str, Any]     # informations annexes libres


# ── KnowledgeBase ─────────────────────────────────────────────────────────────

@dataclass
class KnowledgeBase:
    """
    Conteneur de toutes les connaissances calculées par le KnowledgeEngine.

    Chaque moteur aval (Opportunity, Creative, Script) consomme uniquement
    cette classe — jamais les ContentProfile ou VideoSnapshot directement.

    Méthodes publiques :
      top(dimension, n)  → liste des N premiers faits par observations
      to_dict()          → dict JSON-serializable (préparation Supabase)
      to_json()          → string JSON (sauvegarde fichier)
    """
    generated_at: datetime
    total_profiles: int

    # Distributions catégorielles
    topics: dict[str, KnowledgeFact]
    emotions: dict[str, KnowledgeFact]
    audiences: dict[str, KnowledgeFact]
    content_types: dict[str, KnowledgeFact]
    languages: dict[str, KnowledgeFact]

    # Analyses continues
    evergreen: KnowledgeFact
    trend: KnowledgeFact
    durations: KnowledgeFact

    # Co-occurrences
    combinations: list[KnowledgeFact]

    # ── Accesseurs ─────────────────────────────────────────────────────────────

    def top(self, dimension: str, n: int = 5) -> list[KnowledgeFact]:
        """Retourne les N KnowledgeFact d'une dimension triés par observations."""
        registry: dict[str, dict[str, KnowledgeFact]] = {
            "topics": self.topics,
            "emotions": self.emotions,
            "audiences": self.audiences,
            "content_types": self.content_types,
            "languages": self.languages,
        }
        facts = registry.get(dimension, {})
        return sorted(facts.values(), key=lambda f: f.observations, reverse=True)[:n]

    def top_combinations(self, n: int = 10) -> list[KnowledgeFact]:
        """Retourne les N combinaisons les plus fréquentes."""
        return sorted(self.combinations, key=lambda f: f.observations, reverse=True)[:n]

    # ── Sérialisation (préparation Supabase) ───────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Sérialise en dictionnaire JSON-serializable."""
        def _fact(f: KnowledgeFact) -> dict:
            return {
                "name": f.name,
                "description": f.description,
                "value": f.value,
                "confidence": f.confidence,
                "observations": f.observations,
                "updated_at": f.updated_at.isoformat(),
                "metadata": f.metadata,
            }

        return {
            "generated_at": self.generated_at.isoformat(),
            "total_profiles": self.total_profiles,
            "topics": {k: _fact(v) for k, v in self.topics.items()},
            "emotions": {k: _fact(v) for k, v in self.emotions.items()},
            "audiences": {k: _fact(v) for k, v in self.audiences.items()},
            "content_types": {k: _fact(v) for k, v in self.content_types.items()},
            "languages": {k: _fact(v) for k, v in self.languages.items()},
            "evergreen": _fact(self.evergreen),
            "trend": _fact(self.trend),
            "durations": _fact(self.durations),
            "combinations": [_fact(c) for c in self.combinations],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ── CombinationDiscoverer (interface extensible) ──────────────────────────────

class CombinationDiscoverer(ABC):
    """
    Interface pour les algorithmes de découverte de combinaisons.

    V1 : FrequencyDiscoverer (co-occurrences simples).
    Prochaines versions : Apriori, FP-Growth, Association Rules.

    Pour implémenter un nouvel algorithme :
        class AprioriDiscoverer(CombinationDiscoverer):
            def discover(self, profiles, now): ...
    """

    @abstractmethod
    def discover(
        self,
        profiles: list[ContentProfile],
        now: datetime,
    ) -> list[KnowledgeFact]:
        """Retourne les KnowledgeFact de combinaisons triés par fréquence."""
        ...


class FrequencyDiscoverer(CombinationDiscoverer):
    """
    V1 : détection de co-occurrences par comptage de fréquences.

    Génère des paires et des triples à partir de :
      - (topic, content_type)
      - (topic, audience)
      - (topic, emotion)
      - (content_type, audience)
      - (topic, content_type, audience)  [triple principal]
      - (topic, emotion, content_type)   [triple éditorial]

    Architecture préparée pour l'ajout d'Apriori :
    remplacer FrequencyDiscoverer par AprioriDiscoverer dans KnowledgeEngine.
    """

    _PAIR_DIMENSIONS: list[tuple[str, ...]] = [
        ("primary_topic", "content_type"),
        ("primary_topic", "target_audience"),
        ("primary_topic", "emotion"),
        ("content_type", "target_audience"),
    ]

    _TRIPLE_DIMENSIONS: list[tuple[str, ...]] = [
        ("primary_topic", "content_type", "target_audience"),
        ("primary_topic", "emotion", "content_type"),
    ]

    def __init__(self, min_freq: int = 2, top_n: int = 25) -> None:
        self._min_freq = min_freq
        self._top_n = top_n

    def discover(
        self,
        profiles: list[ContentProfile],
        now: datetime,
    ) -> list[KnowledgeFact]:
        total = len(profiles)
        facts: list[KnowledgeFact] = []

        for dims in self._PAIR_DIMENSIONS + self._TRIPLE_DIMENSIONS:
            scope = "pair" if len(dims) == 2 else "triple"
            counter: Counter = Counter()
            for p in profiles:
                combo = tuple(getattr(p, d) for d in dims)
                counter[combo] += 1

            for i, (combo, freq) in enumerate(
                counter.most_common(self._top_n), start=len(facts)
            ):
                if freq < self._min_freq:
                    break
                label = " + ".join(str(v) for v in combo)
                facts.append(KnowledgeFact(
                    name=f"combination.{scope}.{i}",
                    description=label,
                    value={
                        "keys": dict(zip(dims, combo)),
                        "frequency": freq,
                        "pct": round(freq / total * 100, 1),
                    },
                    confidence=0.70,
                    observations=freq,
                    updated_at=now,
                    metadata={"dimensions": list(dims), "scope": scope},
                ))

        return sorted(facts, key=lambda f: f.observations, reverse=True)


# ── KnowledgeStore (interface — persistance Sprint 11) ────────────────────────

class KnowledgeStore(ABC):
    """
    Interface de persistance de la KnowledgeBase.

    Implémentations :
      - JsonKnowledgeStore   : fichier JSON local (développement).
      - SupabaseKnowledgeStore : table `knowledge_bases` dans Supabase (Sprint 11).

    La méthode load_history() est prévue pour l'analyse temporelle (jour/semaine/mois).
    """

    @abstractmethod
    def save(self, kb: KnowledgeBase) -> None:
        """Persiste une KnowledgeBase."""
        ...

    @abstractmethod
    def load(self) -> Optional[KnowledgeBase]:
        """Charge la dernière KnowledgeBase disponible."""
        ...

    @abstractmethod
    def load_history(self, days: int = 30) -> list[KnowledgeBase]:
        """Charge l'historique des KnowledgeBase pour analyse temporelle."""
        ...


class JsonKnowledgeStore(KnowledgeStore):
    """
    Persistance locale de la KnowledgeBase en fichier JSON.

    Stocke chaque KnowledgeBase dans un fichier JSON horodaté et
    maintient un index des fichiers disponibles pour load_history().

    Exemple :
        store = JsonKnowledgeStore(Path("reports"))
        store.save(kb)
        latest = store.load()
        history = store.load_history(days=7)
    """

    def __init__(self, directory: Path, prefix: str = "knowledge_") -> None:
        self._directory = directory
        self._prefix = prefix
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(self, kb: KnowledgeBase) -> None:
        timestamp = kb.generated_at.strftime("%Y%m%d_%H%M%S")
        filename = f"{self._prefix}{timestamp}.json"
        path = self._directory / filename
        path.write_text(kb.to_json(), encoding="utf-8")
        logger.info("KnowledgeBase sauvegardée → %s", path.name)

    def load(self) -> Optional[KnowledgeBase]:
        """Charge la KnowledgeBase la plus récente."""
        files = sorted(self._directory.glob(f"{self._prefix}*.json"), reverse=True)
        if not files:
            logger.warning("Aucun fichier KnowledgeBase trouvé dans %s", self._directory)
            return None
        return self._load_file(files[0])

    def load_history(self, days: int = 30) -> list[KnowledgeBase]:
        """Charge toutes les KnowledgeBase des N derniers jours."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        files = sorted(self._directory.glob(f"{self._prefix}*.json"), reverse=True)
        result: list[KnowledgeBase] = []
        for path in files:
            # Extraire timestamp du nom de fichier
            stem = path.stem
            try:
                ts_str = stem.replace(self._prefix, "")
                file_dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
                if file_dt >= cutoff:
                    kb = self._load_file(path)
                    if kb:
                        result.append(kb)
            except (ValueError, IndexError):
                continue
        return result

    def _load_file(self, path: Path) -> Optional[KnowledgeBase]:
        """Charge et désérialise une KnowledgeBase depuis un fichier JSON."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            def _fact(d: dict) -> KnowledgeFact:
                return KnowledgeFact(
                    name=d["name"],
                    description=d["description"],
                    value=d["value"],
                    confidence=d["confidence"],
                    observations=d["observations"],
                    updated_at=datetime.fromisoformat(d["updated_at"]),
                    metadata=d.get("metadata", {}),
                )

            def _dist(d: dict) -> dict[str, KnowledgeFact]:
                return {k: _fact(v) for k, v in d.items()}

            kb = KnowledgeBase(
                generated_at=datetime.fromisoformat(data["generated_at"]),
                total_profiles=data["total_profiles"],
                topics=_dist(data["topics"]),
                emotions=_dist(data["emotions"]),
                audiences=_dist(data["audiences"]),
                content_types=_dist(data["content_types"]),
                languages=_dist(data["languages"]),
                evergreen=_fact(data["evergreen"]),
                trend=_fact(data["trend"]),
                durations=_fact(data["durations"]),
                combinations=[_fact(c) for c in data["combinations"]],
            )
            logger.info("KnowledgeBase chargée depuis %s", path.name)
            return kb
        except Exception as exc:
            logger.error("Impossible de charger %s : %s", path.name, exc)
            return None


# ── KnowledgeEngine ───────────────────────────────────────────────────────────

class KnowledgeEngine:
    """
    Transforme une liste de ContentProfile en KnowledgeBase.

    Exemple minimal :
        engine = KnowledgeEngine()
        kb = engine.build(profiles)

    Avec découvreur personnalisé :
        engine = KnowledgeEngine(discoverer=AprioriDiscoverer(min_support=0.05))

    Avec persistance :
        engine = KnowledgeEngine(store=SupabaseKnowledgeStore())
        kb = engine.build(profiles)   # → sauvegardé automatiquement
    """

    def __init__(
        self,
        discoverer: Optional[CombinationDiscoverer] = None,
        store: Optional[KnowledgeStore] = None,
    ) -> None:
        self._discoverer = discoverer if discoverer is not None else FrequencyDiscoverer()
        self._store = store

    # ── Interface publique ─────────────────────────────────────────────────────

    def build(self, profiles: list[ContentProfile]) -> KnowledgeBase:
        """
        Construit une KnowledgeBase à partir des ContentProfile.
        Ne lit aucune VideoSnapshot.
        """
        if not profiles:
            raise ValueError("Aucun ContentProfile fourni — impossible de construire la KnowledgeBase.")

        now = datetime.now(timezone.utc)
        total = len(profiles)
        logger.info("Construction de la KnowledgeBase — %d profils…", total)

        kb = KnowledgeBase(
            generated_at=now,
            total_profiles=total,
            topics=self._compute_distribution(
                "topic", profiles, lambda p: p.primary_topic, now
            ),
            emotions=self._compute_distribution(
                "emotion", profiles, lambda p: p.emotion, now
            ),
            audiences=self._compute_distribution(
                "audience", profiles, lambda p: p.target_audience, now
            ),
            content_types=self._compute_distribution(
                "content_type", profiles, lambda p: p.content_type, now
            ),
            languages=self._compute_distribution(
                "language", profiles, lambda p: p.language, now
            ),
            evergreen=self._compute_evergreen(profiles, now),
            trend=self._compute_trend(profiles, now),
            durations=self._compute_durations(profiles, now),
            combinations=self._discoverer.discover(profiles, now),
        )

        logger.info(
            "KnowledgeBase construite — %d sujets | %d combinaisons | %.0f ms",
            len(kb.topics),
            len(kb.combinations),
            (datetime.now(timezone.utc) - now).total_seconds() * 1000,
        )

        if self._store:
            self._store.save(kb)
            logger.info("KnowledgeBase persistée via %s.", type(self._store).__name__)

        return kb

    # ── Calculs de distributions catégorielles ─────────────────────────────────

    def _compute_distribution(
        self,
        dimension: str,
        profiles: list[ContentProfile],
        get_value: Callable[[ContentProfile], str],
        now: datetime,
    ) -> dict[str, KnowledgeFact]:
        total = len(profiles)
        buckets: dict[str, list[ContentProfile]] = defaultdict(list)
        for p in profiles:
            buckets[get_value(p)].append(p)

        result: dict[str, KnowledgeFact] = {}
        for val, ps in buckets.items():
            n = len(ps)
            avg_conf = _mean(p.confidence for p in ps)
            avg_trend = _mean(p.trend_score for p in ps)
            avg_ev = _mean(p.evergreen_score for p in ps)

            result[val] = KnowledgeFact(
                name=f"{dimension}.{val}",
                description=f"{val}",
                value={
                    "frequency": n,
                    "pct": round(n / total * 100, 1),
                    "avg_trend_score": round(avg_trend, 3),
                    "avg_evergreen_score": round(avg_ev, 3),
                    "avg_confidence": round(avg_conf, 3),
                },
                confidence=round(avg_conf, 3),
                observations=n,
                updated_at=now,
                metadata={},
            )

        return result

    # ── Calculs continus ───────────────────────────────────────────────────────

    def _compute_evergreen(
        self, profiles: list[ContentProfile], now: datetime
    ) -> KnowledgeFact:
        scores = [p.evergreen_score for p in profiles]
        avg = _mean(iter(scores))
        med = stat_median(scores)

        # Sujets les plus evergreen (moyenne par topic)
        topic_scores: dict[str, list[float]] = defaultdict(list)
        for p in profiles:
            topic_scores[p.primary_topic].append(p.evergreen_score)
        top_topics = sorted(
            {t: _mean(iter(s)) for t, s in topic_scores.items()}.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]

        return KnowledgeFact(
            name="evergreen.global",
            description="Analyse de la pérennité du contenu",
            value={
                "mean": round(avg, 3),
                "median": round(med, 3),
                "top_topics": [{"topic": t, "avg_score": round(s, 3)} for t, s in top_topics],
            },
            confidence=0.75,
            observations=len(profiles),
            updated_at=now,
            metadata={},
        )

    def _compute_trend(
        self, profiles: list[ContentProfile], now: datetime
    ) -> KnowledgeFact:
        scores = [p.trend_score for p in profiles]
        avg = _mean(iter(scores))

        # Sujets les plus en tendance
        topic_scores: dict[str, list[float]] = defaultdict(list)
        for p in profiles:
            topic_scores[p.primary_topic].append(p.trend_score)
        top_topics = sorted(
            {t: _mean(iter(s)) for t, s in topic_scores.items()}.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]

        # Répartition haute / basse tendance
        high = sum(1 for s in scores if s >= 0.7)
        low = sum(1 for s in scores if s < 0.4)

        return KnowledgeFact(
            name="trend.global",
            description="Analyse de la dynamique de tendance",
            value={
                "mean": round(avg, 3),
                "high_trend_count": high,        # ≥ 0.7
                "low_trend_count": low,           # < 0.4
                "top_topics": [{"topic": t, "avg_score": round(s, 3)} for t, s in top_topics],
            },
            confidence=0.70,
            observations=len(profiles),
            updated_at=now,
            metadata={},
        )

    def _compute_durations(
        self, profiles: list[ContentProfile], now: datetime
    ) -> KnowledgeFact:
        # La durée brute est mise en cache dans ContentProfile.metadata par ContentTypeAnalyzer
        raw_durations = [
            _duration_from_profile(p)
            for p in profiles
            if _duration_from_profile(p) is not None
        ]

        if not raw_durations:
            # Fallback : classification par content_type si métadonnées absentes
            return KnowledgeFact(
                name="duration.global",
                description="Analyse des durées (données brutes indisponibles)",
                value={"note": "ContentProfile.metadata manquant — utiliser content_type"},
                confidence=0.3,
                observations=len(profiles),
                updated_at=now,
                metadata={},
            )

        avg_s = _mean(iter(raw_durations))
        med_s = stat_median(raw_durations)

        # Répartition par plage de durée
        buckets = {
            "Short (≤60s)": [d for d in raw_durations if d <= 60],
            "Court (1–5min)": [d for d in raw_durations if 61 <= d <= 300],
            "Moyen (5–10min)": [d for d in raw_durations if 301 <= d <= 600],
            "Long (>10min)": [d for d in raw_durations if d > 600],
        }
        ranges = {label: len(ds) for label, ds in buckets.items()}
        best_range = max(ranges, key=lambda k: ranges[k])

        # Performance par plage : avg trend_score
        def _avg_trend_in_range(pred: Callable[[int], bool]) -> float:
            matched = [p for p in profiles if (d := _duration_from_profile(p)) and pred(d)]
            return _mean(p.trend_score for p in matched) if matched else 0.0

        range_perf = {
            "Short (≤60s)": round(_avg_trend_in_range(lambda d: d <= 60), 3),
            "Court (1–5min)": round(_avg_trend_in_range(lambda d: 61 <= d <= 300), 3),
            "Moyen (5–10min)": round(_avg_trend_in_range(lambda d: 301 <= d <= 600), 3),
            "Long (>10min)": round(_avg_trend_in_range(lambda d: d > 600), 3),
        }

        return KnowledgeFact(
            name="duration.global",
            description="Analyse des durées de contenu",
            value={
                "mean_seconds": round(avg_s),
                "median_seconds": round(med_s),
                "mean_fmt": _fmt_dur(round(avg_s)),
                "median_fmt": _fmt_dur(round(med_s)),
                "best_range_by_volume": best_range,
                "ranges": ranges,
                "avg_trend_by_range": range_perf,
            },
            confidence=0.80,
            observations=len(raw_durations),
            updated_at=now,
            metadata={},
        )


# ── Utilitaires (privés au module) ────────────────────────────────────────────

def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _duration_from_profile(profile: ContentProfile) -> Optional[int]:
    """Extrait la durée mise en cache dans ContentProfile.metadata par ContentTypeAnalyzer."""
    raw = profile.metadata.get("content_type", {}).get("duration_s")
    return int(raw) if raw is not None else None


# _fmt_dur est importé de src.utils depuis le haut du fichier"
