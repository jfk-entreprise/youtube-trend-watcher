"""
Google Drive Uploader — Sprint 29 (intégration réelle, Service Account).

Envoie automatiquement chaque package de production
(outputs/YYYY-MM-DD/niche_XX/) vers un dossier Google Drive configuré, en
recréant l'arborescence locale (image_prompts/, animation_prompts/, ...) sous
un sous-dossier par run (ex. "2026-07-10_IA").

Configuration (voir .env.example) :
    GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON  contenu JSON complet de la clé du
                                        compte de service (Google Cloud Console).
    GOOGLE_DRIVE_FOLDER_ID             ID du dossier Drive racine, partagé en
                                        Éditeur avec l'email du compte de service.

Tant que ces deux variables ne sont pas configurées, build_google_drive_uploader()
retombe sur NoOpGoogleDriveUploader (aucune régression pour les environnements
sans Google Drive — dev local, CI sans secrets, etc.).

RealGoogleDriveUploader n'interrompt jamais le pipeline : toute erreur réseau
ou API est capturée et journalisée, le package restant disponible localement
(voir UploadResult.success=False, message=<détail>).
"""

import json
import logging
import mimetypes
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DRIVE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
_SCOPES = ["https://www.googleapis.com/auth/drive"]


@dataclass
class UploadResult:
    success: bool
    remote_path: Optional[str]
    message: str
    remote_url: Optional[str] = None  # lien Drive cliquable — Sprint 29.1 (NotificationService)


class GoogleDriveUploader(ABC):
    """Interface abstraite d'envoi d'un package de production vers Google Drive."""

    @abstractmethod
    def upload_package(self, package_dir: Path, remote_folder_name: str) -> UploadResult:
        """Envoie le contenu de `package_dir` vers un dossier Drive `remote_folder_name`."""
        ...


class NoOpGoogleDriveUploader(GoogleDriveUploader):
    """Implémentation active tant que Google Drive n'est pas configuré."""

    def upload_package(self, package_dir: Path, remote_folder_name: str) -> UploadResult:
        logger.info(
            "Upload Google Drive non configuré — package prêt localement : %s (dossier cible prévu : %s)",
            package_dir, remote_folder_name,
        )
        return UploadResult(success=False, remote_path=None, message="not_configured")


class RealGoogleDriveUploader(GoogleDriveUploader):
    """
    Upload réel via l'API Google Drive v3, authentifié par compte de service.

    `drive_service` peut être injecté directement (tests, réutilisation d'un
    client existant) ; sinon il est construit depuis `service_account_info`.
    """

    def __init__(
        self,
        service_account_info: Optional[Dict[str, Any]] = None,
        root_folder_id: str = "",
        drive_service: Any = None,
    ) -> None:
        if not root_folder_id:
            raise ValueError("root_folder_id est requis (GOOGLE_DRIVE_FOLDER_ID).")
        self._root_folder_id = root_folder_id

        if drive_service is not None:
            self._service = drive_service
        else:
            if not service_account_info:
                raise ValueError("service_account_info est requis (GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON).")
            from google.oauth2.service_account import Credentials
            from googleapiclient.discovery import build

            credentials = Credentials.from_service_account_info(service_account_info, scopes=_SCOPES)
            self._service = build("drive", "v3", credentials=credentials, cache_discovery=False)

    def upload_package(self, package_dir: Path, remote_folder_name: str) -> UploadResult:
        try:
            remote_folder_id = self._get_or_create_folder(remote_folder_name, self._root_folder_id)
            uploaded = self._upload_directory(Path(package_dir), remote_folder_id)
            remote_url = f"https://drive.google.com/drive/folders/{remote_folder_id}"
            logger.info(
                "Package Google Drive envoyé : %s (%d fichier(s), dossier=%s).",
                remote_folder_name, uploaded, remote_folder_id,
            )
            return UploadResult(
                success=True, remote_path=remote_folder_name,
                message=f"{uploaded} fichier(s) envoyé(s)", remote_url=remote_url,
            )
        except Exception as exc:
            logger.warning(
                "Envoi Google Drive échoué pour '%s' (%s) — package conservé localement uniquement.",
                package_dir, exc,
            )
            return UploadResult(success=False, remote_path=None, message=str(exc))

    # ── Interne ────────────────────────────────────────────────────────────────

    def _upload_directory(self, local_dir: Path, parent_id: str) -> int:
        uploaded = 0
        for entry in sorted(local_dir.iterdir()):
            if entry.is_dir():
                sub_folder_id = self._get_or_create_folder(entry.name, parent_id)
                uploaded += self._upload_directory(entry, sub_folder_id)
            else:
                self._upload_file(entry, parent_id)
                uploaded += 1
        return uploaded

    def _get_or_create_folder(self, name: str, parent_id: str) -> str:
        safe_name = name.replace("'", "\\'")
        query = (
            f"name = '{safe_name}' and '{parent_id}' in parents "
            f"and mimeType = '{_DRIVE_FOLDER_MIME_TYPE}' and trashed = false"
        )
        existing = self._service.files().list(q=query, fields="files(id, name)").execute()
        files = existing.get("files", [])
        if files:
            return files[0]["id"]

        created = self._service.files().create(
            body={"name": name, "mimeType": _DRIVE_FOLDER_MIME_TYPE, "parents": [parent_id]},
            fields="id",
        ).execute()
        return created["id"]

    def _upload_file(self, file_path: Path, parent_id: str) -> str:
        from googleapiclient.http import MediaFileUpload

        mime_type, _ = mimetypes.guess_type(str(file_path))
        media = MediaFileUpload(str(file_path), mimetype=mime_type or "application/octet-stream", resumable=False)
        created = self._service.files().create(
            body={"name": file_path.name, "parents": [parent_id]},
            media_body=media,
            fields="id",
        ).execute()
        return created["id"]


def build_google_drive_uploader() -> GoogleDriveUploader:
    """
    Construit l'uploader selon la configuration disponible dans l'environnement.

    - GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON + GOOGLE_DRIVE_FOLDER_ID présents →
      RealGoogleDriveUploader.
    - Variables absentes (ou clé JSON invalide) → NoOpGoogleDriveUploader.
    """
    service_account_json = os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON")
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

    if service_account_json and folder_id:
        try:
            info = json.loads(service_account_json)
            uploader = RealGoogleDriveUploader(service_account_info=info, root_folder_id=folder_id)
            logger.info("Uploader Google Drive actif (dossier racine=%s).", folder_id)
            return uploader
        except Exception as exc:
            logger.warning("Impossible d'initialiser Google Drive (%s) — repli NoOp.", exc)

    logger.info("Google Drive non configuré — uploader NoOp actif.")
    return NoOpGoogleDriveUploader()
