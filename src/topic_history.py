"""
Topic History — anti-duplication de sujet/histoire d'un jour à l'autre
(Sprint 33).

Problème : NicheSelector (Sprint 28, src/niche_selector.py) persiste des
NICHES actives (ex. "IA", "Histoire") mais ne dit rien sur le SUJET précis
produit chaque jour à l'intérieur d'une niche active. Sans garde-fou
supplémentaire, rien n'empêche le pipeline de produire un script quasi
identique deux jours de suite dans la même niche.

Ce module ajoute une fenêtre glissante de sujets récemment produits (table
`topic_history`) et une heuristique de similarité lexicale (aucune
dépendance ML/embedding — stdlib uniquement, cohérent avec les fallbacks
déterministes déjà en place ailleurs dans ce projet) :
  - similarité trop forte (>= DUPLICATE_THRESHOLD)  -> le candidat est
    écarté (une autre opportunité est choisie à la place).
  - similarité modérée (>= SEQUEL_THRESHOLD)        -> le candidat est
    conservé mais annoté (Opportunity.metadata["sequel_of"]) pour que
    LLMScriptGenerator construise explicitement une SUITE plutôt que de
    répéter la même histoire.
  - sinon                                            -> sujet neuf, aucune
    annotation.

Architecture — même style que NicheSelectionStore (src/niche_selector.py) :
  TopicHistoryStore (ABC)
        ├── JsonTopicHistoryStore      (fichier local — tests, fallback)
        └── SupabaseTopicHistoryStore  (table `topic_history`)
"""

import dataclasses
import difflib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from src.opportunity_engine import Opportunity

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 5
DUPLICATE_THRESHOLD = 0.60
SEQUEL_THRESHOLD = 0.35
DEFAULT_JSON_FALLBACK_PATH = Path(".cache/topic_history.json")

_STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "a", "au", "aux",
    "en", "dans", "sur", "pour", "par", "avec", "sans", "ce", "cette", "ces", "que",
    "qui", "quoi", "comment", "pourquoi", "est", "sont", "va", "vont", "plus", "tout",
    "tous", "toutes", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are",
}


def _normalize_words(text: str) -> List[str]:
    words = re.findall(r"[a-zà-ÿ0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 2]


def topic_similarity(title_a: str, title_b: str) -> float:
    """
    Similarité lexicale [0, 1] entre deux titres — combine Jaccard (mots
    significatifs partagés) et difflib.SequenceMatcher (proximité de
    surface) pour limiter à la fois les faux négatifs (reformulation d'un
    même sujet) et les faux positifs (mots communs mais sujets différents).
    """
    words_a = set(_normalize_words(title_a))
    words_b = set(_normalize_words(title_b))
    jaccard = (len(words_a & words_b) / len(words_a | words_b)) if (words_a and words_b) else 0.0
    ratio = difflib.SequenceMatcher(None, title_a.lower(), title_b.lower()).ratio()
    return (jaccard + ratio) / 2


# ── TopicRecord ──────────────────────────────────────────────────────────────

@dataclass
class TopicRecord:
    """Un sujet déjà produit — utilisé pour la comparaison des jours suivants."""
    title: str
    niche: str
    brand_id: str
    produced_date: str   # ISO (YYYY-MM-DD)
    source_video_id: str
    market: str = "FR"   # Marché de la vidéo produite — Sprint 34 : une même
                         # niche peut être active simultanément sur plusieurs
                         # marchés sans que ce soit un doublon.
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── TopicHistoryStore ────────────────────────────────────────────────────────

class TopicHistoryStore(ABC):
    """Interface abstraite de persistance des sujets déjà produits."""

    @abstractmethod
    def load_recent(self, days: int = DEFAULT_LOOKBACK_DAYS, today: Optional[date] = None) -> List[TopicRecord]:
        """Retourne les sujets produits dans les `days` derniers jours."""
        ...

    @abstractmethod
    def save_topic(self, record: TopicRecord) -> None:
        """Enregistre un nouveau sujet produit (append — jamais d'écrasement)."""
        ...


class JsonTopicHistoryStore(TopicHistoryStore):
    """Persistance JSON locale — utile pour les tests et en l'absence de Supabase."""

    def __init__(self, path: Any) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _load_all(self) -> List[TopicRecord]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Lecture de '%s' impossible (%s) — historique considéré vide.", self._path, exc)
            return []
        return [TopicRecord(**row) for row in data]

    def load_recent(self, days: int = DEFAULT_LOOKBACK_DAYS, today: Optional[date] = None) -> List[TopicRecord]:
        cutoff = (today or date.today()) - timedelta(days=days)
        return [r for r in self._load_all() if date.fromisoformat(r.produced_date) >= cutoff]

    def save_topic(self, record: TopicRecord) -> None:
        records = self._load_all()
        records.append(record)
        payload = [dataclasses.asdict(r) for r in records]
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SupabaseTopicHistoryStore(TopicHistoryStore):
    """
    Persistance Supabase — table `topic_history` (sql/create_topic_history.sql).

    Requiert SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY. Utiliser
    build_topic_history_store() pour bénéficier du repli automatique sur JSON.
    """

    def __init__(self, url: str, key: str, table: str = "topic_history") -> None:
        from supabase import create_client
        self._client = create_client(url, key)
        self._table = table

    def load_recent(self, days: int = DEFAULT_LOOKBACK_DAYS, today: Optional[date] = None) -> List[TopicRecord]:
        cutoff = (today or date.today()) - timedelta(days=days)
        result = (
            self._client.table(self._table)
            .select("*")
            .gte("produced_date", cutoff.isoformat())
            .execute()
        )
        records: List[TopicRecord] = []
        for row in result.data:
            try:
                records.append(TopicRecord(
                    title=row["title"],
                    niche=row["niche"],
                    brand_id=row["brand_id"],
                    produced_date=str(row["produced_date"]),
                    source_video_id=row["source_video_id"],
                    market=str(row.get("market") or "FR"),
                    metadata=dict(row.get("metadata") or {}),
                ))
            except Exception as exc:
                logger.warning("Ligne topic_history ignorée (%s)", exc)
        return records

    def save_topic(self, record: TopicRecord) -> None:
        self._client.table(self._table).insert({
            "title": record.title,
            "niche": record.niche,
            "brand_id": record.brand_id,
            "produced_date": record.produced_date,
            "source_video_id": record.source_video_id,
            "market": record.market,
            "metadata": record.metadata,
        }).execute()


class FallbackTopicHistoryStore(TopicHistoryStore):
    """
    Tente le store primaire ; en cas d'erreur (ex. table `topic_history` pas
    encore créée via sql/create_topic_history.sql) bascule sur le secondaire.
    Même pattern que FallbackNicheSelectionStore (src/niche_selector.py).
    """

    def __init__(self, primary: TopicHistoryStore, fallback: TopicHistoryStore) -> None:
        self._primary = primary
        self._fallback = fallback

    def load_recent(self, days: int = DEFAULT_LOOKBACK_DAYS, today: Optional[date] = None) -> List[TopicRecord]:
        try:
            return self._primary.load_recent(days, today)
        except Exception as exc:
            logger.warning("Store primaire en échec (%s) — repli sur le store secondaire.", exc)
            return self._fallback.load_recent(days, today)

    def save_topic(self, record: TopicRecord) -> None:
        try:
            self._primary.save_topic(record)
        except Exception as exc:
            logger.warning("Store primaire en échec (%s) — repli sur le store secondaire.", exc)
            self._fallback.save_topic(record)


def build_topic_history_store(json_fallback_path: Optional[Path] = None) -> TopicHistoryStore:
    """
    Construit le store selon la configuration disponible dans l'environnement
    — même règle que build_niche_selection_store() (src/niche_selector.py).
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    path = json_fallback_path or DEFAULT_JSON_FALLBACK_PATH
    json_store = JsonTopicHistoryStore(path)

    if url and key:
        try:
            supabase_store = SupabaseTopicHistoryStore(url, key)
            logger.info("Store actif (historique des sujets) : Supabase → %s", url)
            return FallbackTopicHistoryStore(primary=supabase_store, fallback=json_store)
        except Exception as exc:
            logger.warning("Impossible d'initialiser Supabase (%s) — repli JSON.", exc)

    logger.info("Store actif (historique des sujets) : JSON → %s", path)
    return json_store


# ── TopicHistoryFilter ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class TopicClassification:
    """Résultat de la classification d'une opportunité face à l'historique."""
    status: str  # "new" | "sequel" | "duplicate"
    matched_title: Optional[str] = None
    matched_date: Optional[str] = None
    similarity: float = 0.0


class TopicHistoryFilter:
    """
    Classe chaque Opportunity candidate d'une niche par rapport aux sujets
    récemment produits DANS LA MÊME NICHE (comparer des sujets de niches
    différentes n'a pas de sens — seule la similarité intra-niche compte).

    - "duplicate" (similarité >= duplicate_threshold) : le candidat rejoue
      trop fidèlement une histoire récente -> écarté si une alternative
      existe dans la niche.
    - "sequel" (similarité >= sequel_threshold) : le candidat prolonge un
      sujet récent sans le répéter -> conservé, mais annoté pour que le
      script généré construise une VRAIE suite plutôt qu'un remake.
    - "new" : aucun recoupement notable.
    """

    def __init__(
        self,
        store: Optional[TopicHistoryStore] = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        duplicate_threshold: float = DUPLICATE_THRESHOLD,
        sequel_threshold: float = SEQUEL_THRESHOLD,
    ) -> None:
        self._store = store or build_topic_history_store()
        self._lookback_days = lookback_days
        self._duplicate_threshold = duplicate_threshold
        self._sequel_threshold = sequel_threshold

    def classify(
        self, title: str, niche: str, market: str = "FR", today: Optional[date] = None
    ) -> TopicClassification:
        recent = [
            r for r in self._store.load_recent(self._lookback_days, today)
            if r.niche == niche and r.market == market
        ]
        best_match: Optional[Tuple[float, TopicRecord]] = None
        for record in recent:
            score = topic_similarity(title, record.title)
            if best_match is None or score > best_match[0]:
                best_match = (score, record)

        if best_match is None:
            return TopicClassification(status="new")

        score, record = best_match
        if score >= self._duplicate_threshold:
            return TopicClassification(
                status="duplicate", matched_title=record.title,
                matched_date=record.produced_date, similarity=score,
            )
        if score >= self._sequel_threshold:
            return TopicClassification(
                status="sequel", matched_title=record.title,
                matched_date=record.produced_date, similarity=score,
            )
        return TopicClassification(status="new", similarity=score)

    def filter_opportunities(
        self, opportunities: List["Opportunity"], niche: str, market: str = "FR",
        today: Optional[date] = None,
    ) -> List["Opportunity"]:
        """
        Retourne les opportunités filtrées/annotées pour une niche :
          - les doublons sont écartés SAUF si écarter tous les candidats
            laisserait la niche sans aucune opportunité (auto-piégeage
            évité : on garde alors le moins similaire, annoté "sequel",
            plutôt que de faire échouer la production du jour).
          - les candidats "sequel" restent, avec metadata["sequel_of"] renseigné.

        `market` isole la comparaison (Sprint 34) : une niche active à la
        fois côté US et côté FR n'est jamais un doublon d'elle-même — seule
        la similarité intra-marché compte.
        """
        if not opportunities:
            return []

        classified = [(opp, self.classify(opp.title, niche, market, today)) for opp in opportunities]
        kept = [(opp, c) for opp, c in classified if c.status != "duplicate"]

        if not kept:
            opp, c = min(classified, key=lambda pair: pair[1].similarity)
            logger.warning(
                "  Niche '%s' : toutes les opportunités ressemblent à un sujet récent — "
                "'%s' conservée malgré tout (similarité=%.2f avec '%s' du %s), traitée en suite.",
                niche, opp.title[:60], c.similarity, c.matched_title, c.matched_date,
            )
            kept = [(opp, replace(c, status="sequel"))]

        result: List["Opportunity"] = []
        for opp, c in kept:
            if c.status == "sequel":
                new_metadata = dict(opp.metadata)
                new_metadata["sequel_of"] = {
                    "title": c.matched_title, "date": c.matched_date,
                    "similarity": round(c.similarity, 2),
                }
                result.append(replace(opp, metadata=new_metadata))
                logger.info(
                    "  '%s' traité comme une SUITE de '%s' (%.0f%% similaire, %s).",
                    opp.title[:60], c.matched_title, c.similarity * 100, c.matched_date,
                )
            else:
                result.append(opp)

        dropped = len(opportunities) - len(result)
        if dropped:
            logger.info("  %d opportunité(s) écartée(s) comme doublon(s) récent(s) dans la niche '%s'.", dropped, niche)

        return result
