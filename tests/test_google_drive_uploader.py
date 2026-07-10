from unittest.mock import MagicMock

import pytest

from src.google_drive_uploader import (
    NoOpGoogleDriveUploader,
    RealGoogleDriveUploader,
    UploadResult,
    build_google_drive_uploader,
)


class TestNoOpGoogleDriveUploader:
    def test_returns_not_configured_without_raising(self, tmp_path):
        uploader = NoOpGoogleDriveUploader()

        result = uploader.upload_package(tmp_path, "2026-07-10_ia")

        assert isinstance(result, UploadResult)
        assert result.success is False
        assert result.remote_path is None
        assert result.remote_url is None
        assert result.message == "not_configured"


class _FakeFilesResource:
    """Simule service.files() — list()/create() renvoient des objets avec .execute()."""

    def __init__(self):
        self.created_folders = []
        self.created_files = []
        self._existing_folder_ids = {}

    def list(self, q, fields):
        # Simule : aucun dossier existant ne matche jamais (toujours créé).
        return _ExecConst({"files": []})

    def create(self, body, fields, media_body=None):
        if body.get("mimeType") == "application/vnd.google-apps.folder":
            new_id = f"folder-{len(self.created_folders) + 1}"
            self.created_folders.append((body["name"], body["parents"][0], new_id))
            return _ExecConst({"id": new_id})
        new_id = f"file-{len(self.created_files) + 1}"
        self.created_files.append((body["name"], body["parents"][0]))
        return _ExecConst({"id": new_id})


class _ExecConst:
    def __init__(self, value):
        self._value = value

    def execute(self):
        return self._value


class _FakeDriveService:
    def __init__(self):
        self._files = _FakeFilesResource()

    def files(self):
        return self._files


def _make_package(tmp_path):
    package_dir = tmp_path / "niche_01"
    package_dir.mkdir()
    (package_dir / "final_script.json").write_text("{}", encoding="utf-8")
    (package_dir / "report.md").write_text("# report", encoding="utf-8")
    sub = package_dir / "image_prompts"
    sub.mkdir()
    (sub / "scene_01.json").write_text("{}", encoding="utf-8")
    return package_dir


class TestRealGoogleDriveUploader:
    def test_requires_root_folder_id(self):
        with pytest.raises(ValueError):
            RealGoogleDriveUploader(drive_service=MagicMock(), root_folder_id="")

    def test_requires_service_account_info_when_no_service_injected(self):
        with pytest.raises(ValueError):
            RealGoogleDriveUploader(root_folder_id="root-123")

    def test_uploads_all_files_recursively(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_service = _FakeDriveService()
        uploader = RealGoogleDriveUploader(drive_service=fake_service, root_folder_id="root-123")

        result = uploader.upload_package(package_dir, "2026-07-10_ia")

        assert result.success is True
        assert result.remote_path == "2026-07-10_ia"
        assert "3 fichier(s)" in result.message
        assert len(fake_service._files.created_files) == 3
        # Un dossier racine "2026-07-10_ia" + un sous-dossier "image_prompts"
        folder_names = {name for name, _, _ in fake_service._files.created_folders}
        assert folder_names == {"2026-07-10_ia", "image_prompts"}
        # Le lien Drive pointe vers le dossier racine du package (premier créé).
        root_folder_id = fake_service._files.created_folders[0][2]
        assert result.remote_url == f"https://drive.google.com/drive/folders/{root_folder_id}"

    def test_upload_failure_does_not_raise(self, tmp_path):
        package_dir = _make_package(tmp_path)
        broken_service = MagicMock()
        broken_service.files.return_value.list.side_effect = RuntimeError("quota exceeded")
        uploader = RealGoogleDriveUploader(drive_service=broken_service, root_folder_id="root-123")

        result = uploader.upload_package(package_dir, "2026-07-10_ia")

        assert result.success is False
        assert result.remote_path is None
        assert result.remote_url is None
        assert "quota exceeded" in result.message

    def test_single_file_failure_does_not_abort_whole_package(self, tmp_path):
        """Sprint 29.1 — auparavant, l'échec d'UN fichier interrompait tout
        l'upload (dossiers déjà créés restant vides sur Drive). Désormais,
        les autres fichiers doivent quand même être envoyés."""
        package_dir = _make_package(tmp_path)
        fake_service = _FakeDriveService()
        real_create = fake_service._files.create
        calls = {"n": 0}

        def flaky_create(body, fields, media_body=None):
            if media_body is not None:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient network error")
            return real_create(body, fields, media_body=media_body)

        fake_service._files.create = flaky_create
        uploader = RealGoogleDriveUploader(drive_service=fake_service, root_folder_id="root-123")

        result = uploader.upload_package(package_dir, "2026-07-10_ia")

        # 2 des 3 fichiers envoyés malgré l'échec du premier — pas d'abandon total.
        assert len(fake_service._files.created_files) == 2
        assert result.success is False  # incomplet : pas 3/3
        assert result.remote_url is not None  # le dossier racine existe bien
        assert "2/3" in result.message
        assert result.uploaded_count == 2
        assert result.total_count == 3

    def test_all_files_uploaded_reports_success_status(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_service = _FakeDriveService()
        uploader = RealGoogleDriveUploader(drive_service=fake_service, root_folder_id="root-123")

        result = uploader.upload_package(package_dir, "2026-07-10_ia")

        assert result.success is True
        assert "3/3" in result.message
        assert result.uploaded_count == 3
        assert result.total_count == 3


class TestFactory:
    def test_returns_noop_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", raising=False)
        monkeypatch.delenv("GOOGLE_DRIVE_FOLDER_ID", raising=False)

        uploader = build_google_drive_uploader()

        assert isinstance(uploader, NoOpGoogleDriveUploader)

    def test_returns_noop_on_invalid_json(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "not-json")
        monkeypatch.setenv("GOOGLE_DRIVE_FOLDER_ID", "root-123")

        uploader = build_google_drive_uploader()

        assert isinstance(uploader, NoOpGoogleDriveUploader)
