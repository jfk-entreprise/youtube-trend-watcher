"""
Couche de persistance — backends interchangeables via StorageBackend.

Pour ajouter un nouveau backend :
    1. Créer une classe qui hérite de StorageBackend.
    2. Implémenter save() et load().
    3. L'injecter dans le script principal via build_storage() ou directement.
"""

import csv
import dataclasses
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from .models import VideoSnapshot

logger = logging.getLogger(__name__)

# Colonnes dans l'ordre des champs du dataclass (source de vérité unique)
CSV_COLUMNS: list[str] = [f.name for f in dataclasses.fields(VideoSnapshot)]


def _to_int(value: Optional[str | int]) -> Optional[int]:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


class StorageBackend(ABC):
    """Contrat minimal que tout backend de stockage doit respecter."""

    @abstractmethod
    def save(self, snapshots: list[VideoSnapshot]) -> int:
        """
        Persiste les snapshots.

        Returns:
            Nombre de lignes/documents effectivement écrits.
        """
        ...

    @abstractmethod
    def load(self) -> list[VideoSnapshot]:
        """
        Charge tous les snapshots disponibles.

        Returns:
            Liste complète des snapshots, dans l'ordre de stockage.
        """
        ...


class CsvStorage(StorageBackend):
    """
    Persistance CSV en mode append-only.

    Chaque appel à save() ajoute de nouvelles lignes — les lignes existantes
    ne sont jamais modifiées. C'est ce qui permet de suivre l'évolution
    des statistiques d'une même vidéo dans le temps.
    """

    def __init__(self, path: Path):
        self._path = path

    def save(self, snapshots: list[VideoSnapshot]) -> int:
        if not snapshots:
            logger.warning("save() appelé avec une liste vide, rien à écrire.")
            return 0

        is_new_file = not self._path.exists() or self._path.stat().st_size == 0

        with open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if is_new_file:
                writer.writeheader()
            for snap in snapshots:
                writer.writerow(dataclasses.asdict(snap))

        logger.info("%d snapshots ajoutés dans %s", len(snapshots), self._path.name)
        return len(snapshots)

    def load(self) -> list[VideoSnapshot]:
        if not self._path.exists():
            logger.warning("Fichier CSV introuvable : %s", self._path)
            return []

        snapshots: list[VideoSnapshot] = []
        skipped = 0
        with open(self._path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    snapshots.append(VideoSnapshot(
                        video_id=row["video_id"],
                        title=row["title"],
                        channel_id=row["channel_id"],
                        channel_title=row["channel_title"],
                        published_at=row["published_at"],
                        description=row.get("description", ""),
                        duration_iso=row["duration_iso"],
                        duration_seconds=int(row.get("duration_seconds") or 0),
                        view_count=_to_int(row.get("view_count")),
                        like_count=_to_int(row.get("like_count")),
                        comment_count=_to_int(row.get("comment_count")),
                        keyword=row["keyword"],
                        source=row.get("source", "keyword"),
                        market=row.get("market") or "FR",
                        collected_at=row["collected_at"],
                    ))
                except Exception as exc:
                    skipped += 1
                    logger.debug("Ligne ignorée lors du chargement CSV (%s)", exc)

        if skipped:
            logger.warning("%d ligne(s) ignorée(s) lors du chargement CSV.", skipped)
        logger.info("%d snapshots chargés depuis %s", len(snapshots), self._path.name)
        return snapshots


class SupabaseStorage(StorageBackend):
    """
    Persistance Supabase — insère et lit les snapshots dans la table video_snapshots.

    Requiert les variables SUPABASE_URL et SUPABASE_SERVICE_ROLE_KEY.
    Utiliser build_storage() pour bénéficier du repli automatique sur CSV.
    """

    def __init__(self, url: str, key: str, table: str = "video_snapshots"):
        from supabase import create_client
        self._client = create_client(url, key)
        self._table = table

    def save(self, snapshots: list[VideoSnapshot]) -> int:
        if not snapshots:
            logger.warning("save() appelé avec une liste vide, rien à insérer.")
            return 0

        rows = [dataclasses.asdict(s) for s in snapshots]
        result = self._client.table(self._table).insert(rows).execute()
        count = len(result.data)
        logger.info("%d snapshots insérés dans Supabase (%s)", count, self._table)
        return count

    def load(self) -> list[VideoSnapshot]:
        result = self._client.table(self._table).select("*").execute()
        snapshots: list[VideoSnapshot] = []
        skipped = 0
        for row in result.data:
            try:
                snapshots.append(VideoSnapshot(
                    video_id=row["video_id"],
                    title=row["title"],
                    channel_id=row["channel_id"],
                    channel_title=row["channel_title"],
                    published_at=row["published_at"],
                    description=row.get("description", ""),
                    duration_iso=row["duration_iso"],
                    duration_seconds=int(row.get("duration_seconds") or 0),
                    view_count=_to_int(row.get("view_count")),
                    like_count=_to_int(row.get("like_count")),
                    comment_count=_to_int(row.get("comment_count")),
                    keyword=row["keyword"],
                    source=row.get("source", "keyword"),
                    market=row.get("market") or "FR",
                    collected_at=row["collected_at"],
                ))
            except Exception as exc:
                skipped += 1
                logger.debug("Ligne ignorée lors du chargement Supabase (%s)", exc)

        if skipped:
            logger.warning("%d ligne(s) ignorée(s) lors du chargement Supabase.", skipped)
        logger.info("%d snapshots chargés depuis Supabase (%s)", len(snapshots), self._table)
        return snapshots


class FallbackStorage(StorageBackend):
    """
    Tente le backend primaire ; en cas d'erreur bascule sur le secondaire.
    Couplage standard : SupabaseStorage (primaire) → CsvStorage (secours).
    """

    def __init__(self, primary: StorageBackend, fallback: StorageBackend):
        self._primary = primary
        self._fallback = fallback

    def save(self, snapshots: list[VideoSnapshot]) -> int:
        try:
            return self._primary.save(snapshots)
        except Exception as exc:
            logger.error(
                "Backend primaire en échec (%s) — repli sur le backend secondaire.", exc
            )
            return self._fallback.save(snapshots)

    def load(self) -> list[VideoSnapshot]:
        try:
            return self._primary.load()
        except Exception as exc:
            logger.error(
                "Backend primaire en échec (%s) — repli sur le backend secondaire.", exc
            )
            return self._fallback.load()


def build_storage(csv_fallback_path: Path) -> StorageBackend:
    """
    Construit le backend selon la configuration disponible dans l'environnement.

    - SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY présents → FallbackStorage
      (SupabaseStorage en primaire, CsvStorage en secours).
    - Variables absentes → CsvStorage seul.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    csv_storage = CsvStorage(csv_fallback_path)

    if url and key:
        try:
            supabase_storage = SupabaseStorage(url, key)
            logger.info("Backend actif : Supabase → %s", url)
            return FallbackStorage(primary=supabase_storage, fallback=csv_storage)
        except Exception as exc:
            logger.warning(
                "Impossible d'initialiser Supabase (%s) — repli sur CSV.", exc
            )

    logger.info("Backend actif : CSV → %s", csv_fallback_path)
    return csv_storage
