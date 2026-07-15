"""
Supabase Storage Uploader — Sprint 30 (remplace Google Drive).

Google Drive a atteint une limitation de plateforme définitive : un compte
de service n'a AUCUN quota de stockage propre (voir l'audit Sprint 29.2 dans
l'historique du projet) — chaque upload de fichier échouait avec
HTTP 403 storageQuotaExceeded, sans solution possible côté code (seule une
migration vers un Shared Drive Google Workspace l'aurait résolu). Supabase
Storage remplace entièrement Google Drive comme backend d'upload des packages
de production, en réutilisant les identifiants Supabase déjà configurés pour
la persistance des données (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY — voir
src/storage.py) : aucun nouveau secret à provisionner.

Architecture identique à l'ancien src/google_drive_uploader.py (Sprint 29) :
    StorageUploader (interface) → NoOpStorageUploader (repli, non configuré) /
    SupabaseStorageUploader (upload réel) — build_storage_uploader() choisit
    automatiquement selon la configuration disponible.

Fiabilité (même principe que Sprint 29.1) : chaque fichier du package est
envoyé individuellement — l'échec de l'un n'interrompt jamais l'envoi des
autres. UploadResult.success ne vaut True que si tous les fichiers attendus
ont été envoyés ET VÉRIFIÉS présents sur Supabase Storage après coup.

Prérequis Supabase (hors code) : créer un bucket de Storage nommé "production"
dans le Dashboard Supabase (Storage > New bucket) — voir docs/supabase_deployment.md.

Sprint 30.5 — un upload() qui ne lève pas d'exception ne prouve pas que
l'objet existe réellement côté serveur (SDK, proxy, réponse tronquée...).
Chaque upload est donc immédiatement vérifié via bucket.exists() : seul un
fichier confirmé présent après coup est compté comme "uploaded". Le préfixe
distant est aussi normalisé pour ne jamais dupliquer le nom du bucket dans
la clé d'objet (ex: bucket "production" + dossier "production/2026-07-10/..."
ne doit jamais produire la clé "production/production/2026-07-10/...").
"""

import logging
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_BUCKET = "production"


@dataclass
class UploadResult:
    success: bool
    uploaded_count: int
    total_count: int
    remote_url: Optional[str]
    error: Optional[str] = None


class StorageUploader(ABC):
    """Interface abstraite d'envoi d'un package de production vers un stockage distant."""

    @abstractmethod
    def upload_package(self, package_dir: Path, remote_folder_name: str) -> UploadResult:
        """Envoie le contenu de `package_dir` sous le préfixe `remote_folder_name`."""
        ...


class NoOpStorageUploader(StorageUploader):
    """Implémentation active tant que Supabase Storage n'est pas configuré."""

    def upload_package(self, package_dir: Path, remote_folder_name: str) -> UploadResult:
        logger.info(
            "Upload Supabase Storage non configuré — package prêt localement : %s (préfixe cible prévu : %s)",
            package_dir, remote_folder_name,
        )
        return UploadResult(
            success=False, uploaded_count=0, total_count=0,
            remote_url=None, error="not_configured",
        )


class SupabaseStorageUploader(StorageUploader):
    """
    Upload réel via Supabase Storage (bucket configurable, "production" par défaut).

    `client` peut être injecté directement (tests, réutilisation d'un client
    existant) ; sinon il est construit depuis `url`/`key`.
    """

    def __init__(
        self,
        url: str = "",
        key: str = "",
        bucket: str = DEFAULT_BUCKET,
        client: Any = None,
    ) -> None:
        self._bucket = bucket
        self._url = url.rstrip("/") if url else url

        if client is not None:
            self._client = client
        else:
            if not url or not key:
                raise ValueError("url et key sont requis (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).")
            from supabase import create_client
            self._client = create_client(url, key)

    def _normalize_remote_folder(self, remote_folder_name: str) -> str:
        """
        Garantit que le préfixe distant ne duplique jamais le nom du bucket.

        Le bucket EST déjà l'espace de nommage racine ("production") : un
        appelant qui construit par erreur un préfixe commençant par
        "{bucket}/..." produirait une clé d'objet "production/production/..."
        une fois combinée à `self._bucket` par l'API Storage. On détecte et
        retire ce préfixe redondant ici, une bonne fois pour toutes, plutôt
        que de compter sur chaque appelant pour ne jamais s'y prendre mal.
        """
        folder = remote_folder_name.strip("/")
        prefix = f"{self._bucket}/"
        if folder == self._bucket or folder.startswith(prefix):
            stripped = folder[len(self._bucket):].lstrip("/")
            logger.warning(
                "Préfixe distant '%s' dupliquait le nom du bucket '%s' — normalisé en '%s'.",
                remote_folder_name, self._bucket, stripped,
            )
            return stripped
        return folder

    def upload_package(self, package_dir: Path, remote_folder_name: str) -> UploadResult:
        package_dir = Path(package_dir)
        files = sorted(f for f in package_dir.rglob("*") if f.is_file())
        total = len(files)
        remote_folder_name = self._normalize_remote_folder(remote_folder_name)
        logger.info(
            "Uploading production package to Supabase Storage...\n\nBucket:\n%s\n\nFound:\n%d files",
            self._bucket, total,
        )

        bucket_api = self._client.storage.from_(self._bucket)
        self._clear_remote_folder(bucket_api, remote_folder_name)
        uploaded = 0
        failures: List[tuple] = []

        for file_path in files:
            relative = file_path.relative_to(package_dir).as_posix()
            remote_path = f"{remote_folder_name}/{relative}" if remote_folder_name else relative
            if remote_path == self._bucket or remote_path.startswith(f"{self._bucket}/"):
                raise ValueError(
                    f"Clé d'objet dupliquant le nom du bucket détectée : "
                    f"bucket='{self._bucket}' path='{remote_path}' — la normalisation a échoué."
                )
            try:
                mime_type, _ = mimetypes.guess_type(str(file_path))
                response = bucket_api.upload(
                    path=remote_path,
                    file=file_path,
                    file_options={"content-type": mime_type or "application/octet-stream", "upsert": "true"},
                )
                logger.info(
                    "Upload API response — bucket=%s path=%s response=%r",
                    self._bucket, remote_path, response,
                )
            except Exception as exc:
                logger.warning(
                    "Échec de l'upload du fichier '%s' (bucket=%s path=%s) : %s",
                    file_path, self._bucket, remote_path, exc,
                )
                failures.append((str(file_path), f"upload_error: {exc}"))
                continue

            verified, verify_detail = self._verify_object_exists(bucket_api, remote_path)
            logger.info(
                "Verification — bucket=%s path=%s exists=%s detail=%s",
                self._bucket, remote_path, verified, verify_detail,
            )
            if verified:
                uploaded += 1
            else:
                logger.warning(
                    "Upload signalé sans exception mais objet introuvable après coup — "
                    "bucket=%s path=%s (%s). Ne compte PAS comme réussi.",
                    self._bucket, remote_path, verify_detail,
                )
                failures.append((str(file_path), f"verification_failed: {verify_detail}"))

        remote_url = self._build_remote_url(remote_folder_name)
        success = total > 0 and uploaded == total
        error = None
        if failures:
            error = f"{len(failures)} fichier(s) en échec : " + "; ".join(
                f"{path} ({reason})" for path, reason in failures
            )
            logger.warning(
                "Upload Supabase Storage incomplet pour '%s' : %d/%d fichier(s) envoyé(s) ET vérifié(s). Échecs : %s",
                remote_folder_name, uploaded, total, failures,
            )
        logger.info(
            "Uploaded (vérifiés):\n%d / %d\n\nStorage URL:\n%s\n\nStatus: %s",
            uploaded, total, remote_url, "SUCCESS" if success else "PARTIAL",
        )
        return UploadResult(
            success=success, uploaded_count=uploaded, total_count=total,
            remote_url=remote_url, error=error,
        )

    @classmethod
    def _list_existing_remote_files(cls, bucket_api: Any, prefix: str) -> List[str]:
        """
        Liste récursivement tous les fichiers déjà présents sous `prefix`
        (l'API Storage ne liste qu'un niveau à la fois — il faut redescendre
        manuellement dans chaque sous-dossier).
        """
        found: List[str] = []
        try:
            items = bucket_api.list(prefix)
        except Exception as exc:
            logger.warning("Impossible de lister le contenu distant existant sous '%s' : %s", prefix, exc)
            return found
        for item in items:
            full_path = f"{prefix}/{item['name']}" if prefix else item["name"]
            is_file = item.get("id") is not None
            if is_file:
                found.append(full_path)
            else:
                found.extend(cls._list_existing_remote_files(bucket_api, full_path))
        return found

    @classmethod
    def _clear_remote_folder(cls, bucket_api: Any, remote_folder_name: str) -> None:
        """
        Supprime tout contenu déjà présent sous `remote_folder_name` avant
        l'upload — sans cela, un deuxième run le même jour (retry manuel,
        double déclenchement) laisse les fichiers de l'ancien run mélangés
        avec les nouveaux (ex: scene_03.json ET scene_03a.json/scene_03b.json
        cohabitant pour deux scripts différents), ce qui rend le dossier
        distant incohérent pour le montage.
        """
        existing = cls._list_existing_remote_files(bucket_api, remote_folder_name)
        if not existing:
            return
        logger.info(
            "Nettoyage de %d fichier(s) existant(s) sous '%s' avant l'upload (éviter les orphelins d'un run précédent).",
            len(existing), remote_folder_name,
        )
        batch_size = 20
        for i in range(0, len(existing), batch_size):
            batch = existing[i : i + batch_size]
            try:
                bucket_api.remove(batch)
            except Exception as exc:
                logger.warning("Échec de suppression d'un lot de fichiers existants (%s) : %s", batch, exc)

    @staticmethod
    def _verify_object_exists(bucket_api: Any, remote_path: str) -> "tuple[bool, str]":
        """
        Confirme qu'un objet existe réellement côté serveur après upload() —
        ne jamais faire confiance à l'absence d'exception seule (Sprint 30.5).

        Utilise `bucket.exists()` (requête HEAD réelle) : certaines versions/
        configurations peuvent lever une exception pour un objet manquant
        (ex: 404) plutôt que de retourner False — les deux cas sont traités
        comme "non vérifié".
        """
        try:
            found = bucket_api.exists(remote_path)
        except Exception as exc:
            return False, f"exists() a levé une exception : {exc}"
        if not found:
            return False, "exists() a retourné False"
        return True, "confirmé par exists()"

    def _build_remote_url(self, remote_folder_name: str) -> Optional[str]:
        """Construit l'URL pointant vers le dossier distant (best effort)."""
        try:
            bucket_api = self._client.storage.from_(self._bucket)
            return bucket_api.get_public_url(remote_folder_name)
        except Exception as exc:
            logger.debug("Impossible d'obtenir l'URL publique Supabase Storage (%s).", exc)
            if self._url:
                return f"{self._url}/storage/v1/object/public/{self._bucket}/{remote_folder_name}"
            return None


def build_storage_uploader() -> StorageUploader:
    """
    Construit l'uploader selon la configuration disponible dans l'environnement.

    - SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY présents → SupabaseStorageUploader.
    - Variables absentes (ou init impossible) → NoOpStorageUploader.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if url and key:
        try:
            uploader = SupabaseStorageUploader(url=url, key=key)
            logger.info("Uploader Supabase Storage actif (bucket=%s).", DEFAULT_BUCKET)
            return uploader
        except Exception as exc:
            logger.warning("Impossible d'initialiser Supabase Storage (%s) — repli NoOp.", exc)

    logger.info("Supabase Storage non configuré — uploader NoOp actif.")
    return NoOpStorageUploader()
