"""
Niche Selector — persistance des niches actives (Sprint 28, Studio de production).

Règles business (vision produit) :
  1. Identifier les niches ayant le meilleur potentiel de croissance durable.
  2. Conserver les niches déjà sélectionnées tant qu'elles restent parmi les
     meilleures.
  3. Ne remplacer une niche que lorsqu'une nouvelle niche présente un
     potentiel significativement supérieur (seuil REPLACEMENT_THRESHOLD).

On ne change PAS de niche chaque jour : NicheAnalyzer recalcule un classement
complet à chaque run (aucune modification de ce moteur), mais NicheSelector
compare ce classement à l'état persisté (table `active_niches`) pour décider
ce qui change réellement d'un jour à l'autre.

Architecture — même style que BrandStore/BrandEngine (src/brand_engine.py) :
  NicheSelectionStore (ABC)
        ├── JsonNicheSelectionStore      (fichier local — tests, fallback)
        └── SupabaseNicheSelectionStore  (table `active_niches`, Sprint 28)

  NicheSelector : logique de sélection, indépendante du store injecté.
"""

import dataclasses
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.niche_intelligence import Niche

logger = logging.getLogger(__name__)

REPLACEMENT_THRESHOLD = 0.15  # une nouvelle niche doit dépasser l'active de 15% pour la remplacer
DEFAULT_MAX_NICHES = 2
DEFAULT_JSON_FALLBACK_PATH = Path(".cache/active_niches.json")


# ── ActiveNicheRecord ─────────────────────────────────────────────────────────

@dataclass
class ActiveNicheRecord:
    """État persisté d'une niche actuellement active pour une chaîne."""
    niche_name: str
    niche_score: float
    first_selected_date: str   # ISO (YYYY-MM-DD) — date de première sélection
    last_confirmed_date: str   # ISO (YYYY-MM-DD) — dernier run où elle a été confirmée
    market: str = "FR"         # Marché ciblé ('FR' | 'US'...) — Sprint 34 : une même
                               # niche peut être active simultanément sur plusieurs marchés.
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── NicheSelectionStore ───────────────────────────────────────────────────────

class NicheSelectionStore(ABC):
    """Interface abstraite de persistance de l'état des niches actives."""

    @abstractmethod
    def load_active(self) -> List[ActiveNicheRecord]:
        """Retourne l'état actif persisté (peut être vide au premier run)."""
        ...

    @abstractmethod
    def save_active(self, records: List[ActiveNicheRecord]) -> None:
        """Remplace l'état actif persisté par `records` (les niches absentes sont retirées)."""
        ...


class JsonNicheSelectionStore(NicheSelectionStore):
    """Persistance JSON locale — utile pour les tests et en l'absence de Supabase."""

    def __init__(self, path: Any) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load_active(self) -> List[ActiveNicheRecord]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Lecture de '%s' impossible (%s) — état actif considéré vide.", self._path, exc)
            return []
        return [ActiveNicheRecord(**row) for row in data]

    def save_active(self, records: List[ActiveNicheRecord]) -> None:
        payload = [dataclasses.asdict(r) for r in records]
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class SupabaseNicheSelectionStore(NicheSelectionStore):
    """
    Persistance Supabase — table `active_niches` (sql/create_active_niches.sql).

    Requiert SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY. Utiliser
    build_niche_selection_store() pour bénéficier du repli automatique sur JSON.
    """

    def __init__(self, url: str, key: str, table: str = "active_niches") -> None:
        from supabase import create_client
        self._client = create_client(url, key)
        self._table = table

    def load_active(self) -> List[ActiveNicheRecord]:
        result = self._client.table(self._table).select("*").execute()
        records: List[ActiveNicheRecord] = []
        for row in result.data:
            try:
                records.append(ActiveNicheRecord(
                    niche_name=row["niche_name"],
                    niche_score=float(row["niche_score"]),
                    first_selected_date=str(row["first_selected_date"]),
                    last_confirmed_date=str(row["last_confirmed_date"]),
                    market=str(row.get("market") or "FR"),
                    metadata=dict(row.get("metadata") or {}),
                ))
            except Exception as exc:
                logger.warning("Ligne active_niches ignorée (%s)", exc)
        return records

    def save_active(self, records: List[ActiveNicheRecord]) -> None:
        # Clé naturelle (niche_name, market) — Sprint 34 : une même niche peut
        # être active simultanément sur plusieurs marchés, donc niche_name seul
        # ne suffit plus à identifier une ligne.
        current_keys = {(r.niche_name, r.market) for r in records}
        existing_rows = self._client.table(self._table).select("niche_name,market").execute().data
        existing_keys = {(row["niche_name"], row.get("market") or "FR") for row in existing_rows}

        if records:
            rows = [
                {
                    "niche_name": r.niche_name,
                    "niche_score": r.niche_score,
                    "first_selected_date": r.first_selected_date,
                    "last_confirmed_date": r.last_confirmed_date,
                    "market": r.market,
                    "metadata": r.metadata,
                }
                for r in records
            ]
            self._client.table(self._table).upsert(rows, on_conflict="niche_name,market").execute()

        for name, market in existing_keys - current_keys:
            self._client.table(self._table).delete().eq("niche_name", name).eq("market", market).execute()


class FallbackNicheSelectionStore(NicheSelectionStore):
    """
    Tente le store primaire ; en cas d'erreur (ex. table `active_niches` pas
    encore créée via sql/create_active_niches.sql) bascule sur le secondaire.
    Couplage standard : SupabaseNicheSelectionStore (primaire) → JsonNicheSelectionStore (secours).
    Même pattern que FallbackStorage (src/storage.py).
    """

    def __init__(self, primary: NicheSelectionStore, fallback: NicheSelectionStore) -> None:
        self._primary = primary
        self._fallback = fallback

    def load_active(self) -> List[ActiveNicheRecord]:
        try:
            return self._primary.load_active()
        except Exception as exc:
            logger.warning("Store primaire en échec (%s) — repli sur le store secondaire.", exc)
            return self._fallback.load_active()

    def save_active(self, records: List[ActiveNicheRecord]) -> None:
        try:
            self._primary.save_active(records)
        except Exception as exc:
            logger.warning("Store primaire en échec (%s) — repli sur le store secondaire.", exc)
            self._fallback.save_active(records)


def build_niche_selection_store(
    json_fallback_path: Optional[Path] = None,
) -> NicheSelectionStore:
    """
    Construit le store selon la configuration disponible dans l'environnement.

    - SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY présents → FallbackNicheSelectionStore
      (SupabaseNicheSelectionStore en primaire, JsonNicheSelectionStore en secours —
      utile tant que sql/create_active_niches.sql n'a pas encore été exécuté).
    - Variables absentes → JsonNicheSelectionStore seul.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    path = json_fallback_path or DEFAULT_JSON_FALLBACK_PATH
    json_store = JsonNicheSelectionStore(path)

    if url and key:
        try:
            supabase_store = SupabaseNicheSelectionStore(url, key)
            logger.info("Store actif (niches actives) : Supabase → %s", url)
            return FallbackNicheSelectionStore(primary=supabase_store, fallback=json_store)
        except Exception as exc:
            logger.warning("Impossible d'initialiser Supabase (%s) — repli JSON.", exc)

    logger.info("Store actif (niches actives) : JSON → %s", path)
    return json_store


# ── NicheSelector ─────────────────────────────────────────────────────────────

class NicheSelector:
    """
    Sélectionne les niches actives du jour à partir du classement recalculé
    par NicheAnalyzer et de l'état persisté (règles business 1-3).

    Exemple minimal :
        selector = NicheSelector()
        niches_du_jour = selector.select_daily_niches(all_niches_candidates)
    """

    def __init__(
        self,
        store: Optional[NicheSelectionStore] = None,
        max_niches: int = DEFAULT_MAX_NICHES,
        replacement_threshold: float = REPLACEMENT_THRESHOLD,
    ) -> None:
        self._store = store or build_niche_selection_store()
        self._max_niches = max_niches
        self._replacement_threshold = replacement_threshold

    def select_daily_niches(
        self, candidates: List[Niche], today: Optional[date] = None, market: str = "FR"
    ) -> List[Niche]:
        """
        Retourne au maximum `max_niches` Niche pour le run du jour, en gardant
        les niches déjà actives sauf si une niche non-active les dépasse
        significativement (> replacement_threshold).

        `market` isole cet appel des autres marchés (Sprint 34) : seuls les
        `ActiveNicheRecord` de ce marché sont considérés pour la conservation/
        remplacement, et les enregistrements des AUTRES marchés sont toujours
        repassés tels quels à `save_active()` (qui remplace l'état persisté
        dans son intégralité) — un appel pour le marché US ne doit jamais
        effacer l'état déjà sauvegardé pour le marché FR, et réciproquement.
        """
        if not candidates:
            raise RuntimeError("Aucune niche candidate à sélectionner.")

        today_str = (today or date.today()).isoformat()
        candidates_by_name = {n.name: n for n in candidates}
        ranked_candidates = sorted(candidates, key=lambda n: n.niche_score, reverse=True)

        all_records = self._store.load_active()
        other_market_records = [r for r in all_records if r.market != market]
        active_records = [r for r in all_records if r.market == market]
        active_names = {r.niche_name for r in active_records}

        kept_names: List[str] = []
        for record in active_records:
            candidate = candidates_by_name.get(record.niche_name)
            if candidate is None:
                logger.info("Niche active '%s' absente des candidates du jour — abandonnée.", record.niche_name)
                continue

            best_new_contender = max(
                (c for c in ranked_candidates if c.name not in active_names),
                key=lambda c: c.niche_score,
                default=None,
            )
            if (
                best_new_contender is not None
                and best_new_contender.niche_score > candidate.niche_score * (1 + self._replacement_threshold)
            ):
                logger.info(
                    "Niche active '%s' (score=%.3f) remplacée par '%s' (score=%.3f, +%.0f%%).",
                    record.niche_name, candidate.niche_score,
                    best_new_contender.name, best_new_contender.niche_score,
                    (best_new_contender.niche_score / candidate.niche_score - 1) * 100,
                )
                continue

            kept_names.append(record.niche_name)

        kept_names = kept_names[: self._max_niches]

        final_names = list(kept_names)
        for candidate in ranked_candidates:
            if len(final_names) >= self._max_niches:
                break
            if candidate.name not in final_names:
                final_names.append(candidate.name)

        selected = sorted(
            (candidates_by_name[name] for name in final_names),
            key=lambda n: n.niche_score,
            reverse=True,
        )

        records_by_name = {r.niche_name: r for r in active_records}
        new_records: List[ActiveNicheRecord] = []
        for niche in selected:
            prior = records_by_name.get(niche.name)
            new_records.append(ActiveNicheRecord(
                niche_name=niche.name,
                niche_score=niche.niche_score,
                first_selected_date=prior.first_selected_date if prior else today_str,
                last_confirmed_date=today_str,
                market=market,
                metadata=dict(prior.metadata) if prior else {},
            ))
        self._store.save_active(other_market_records + new_records)

        for niche in selected:
            status = "conservée" if niche.name in records_by_name else "nouvelle"
            logger.info("  Niche du jour (%s) : %-25s score=%.3f", status, niche.name, niche.niche_score)

        return selected
