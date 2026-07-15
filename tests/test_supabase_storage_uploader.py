from unittest.mock import MagicMock

import pytest

from src.supabase_storage_uploader import (
    DEFAULT_BUCKET,
    NoOpStorageUploader,
    SupabaseStorageUploader,
    UploadResult,
    build_storage_uploader,
)


class TestNoOpStorageUploader:
    def test_returns_not_configured_without_raising(self, tmp_path):
        uploader = NoOpStorageUploader()

        result = uploader.upload_package(tmp_path, "2026-07-10/niche_01")

        assert isinstance(result, UploadResult)
        assert result.success is False
        assert result.uploaded_count == 0
        assert result.total_count == 0
        assert result.remote_url is None
        assert result.error == "not_configured"


class _FakeBucketAPI:
    """Simule client.storage.from_(bucket) — upload()/exists()/get_public_url().

    `missing_paths` / `exists_raises_for` permettent de simuler, par chemin,
    un upload() qui ne lève pas mais dont l'objet est en réalité absent (ou
    dont la vérification échoue) — c'est exactement le scénario Sprint 30.5.
    """

    def __init__(self, missing_paths=(), exists_raises_for=(), existing_remote_files=()):
        self.uploaded_paths = []
        self.exists_calls = []
        self.removed_paths = []
        self._missing_paths = set(missing_paths)
        self._exists_raises_for = set(exists_raises_for)
        # Simule un arbre de fichiers déjà présents côté serveur (Sprint 37 —
        # nettoyage avant upload) : {"dossier/sous-dossier": ["a.json", ...]}.
        self._remote_tree = {}
        for path in existing_remote_files:
            parent, _, name = path.rpartition("/")
            self._remote_tree.setdefault(parent, []).append(name)

    def upload(self, path, file, file_options=None):
        self.uploaded_paths.append(path)
        return {"path": path}

    def exists(self, path):
        self.exists_calls.append(path)
        if path in self._exists_raises_for:
            raise RuntimeError(f"HEAD {path} -> 404 (simulated)")
        return path not in self._missing_paths

    def list(self, prefix=""):
        names_at_prefix = self._remote_tree.get(prefix, [])
        # Un "dossier" est toute clé de _remote_tree préfixée par prefix/ —
        # ses fichiers directs ont un "id" (fichier), les sous-dossiers non.
        subfolders = {
            key[len(prefix) + 1 :].split("/")[0]
            for key in self._remote_tree
            if prefix and key.startswith(prefix + "/") and key != prefix
        }
        items = [{"name": name, "id": "fake-id"} for name in names_at_prefix]
        items += [{"name": name, "id": None} for name in subfolders]
        return items

    def remove(self, paths):
        self.removed_paths.extend(paths)
        return {"removed": paths}

    def get_public_url(self, path, options=None):
        return f"https://fake.supabase.co/storage/v1/object/public/{DEFAULT_BUCKET}/{path}"


class _FakeStorageNamespace:
    def __init__(self, bucket_api):
        self._bucket_api = bucket_api

    def from_(self, bucket_name):
        return self._bucket_api


class _FakeSupabaseClient:
    def __init__(self, **bucket_api_kwargs):
        self.bucket_api = _FakeBucketAPI(**bucket_api_kwargs)
        self.storage = _FakeStorageNamespace(self.bucket_api)


def _make_package(tmp_path):
    package_dir = tmp_path / "niche_01"
    package_dir.mkdir()
    (package_dir / "final_script.json").write_text("{}", encoding="utf-8")
    (package_dir / "report.md").write_text("# report", encoding="utf-8")
    sub = package_dir / "image_prompts"
    sub.mkdir()
    (sub / "scene_01.json").write_text("{}", encoding="utf-8")
    return package_dir


class TestSupabaseStorageUploader:
    def test_requires_url_and_key_when_no_client_injected(self):
        with pytest.raises(ValueError):
            SupabaseStorageUploader(url="", key="")

    def test_uploads_all_files_recursively(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is True
        assert result.uploaded_count == 3
        assert result.total_count == 3
        assert result.error is None
        assert result.remote_url == (
            "https://fake.supabase.co/storage/v1/object/public/production/2026-07-10/niche_01"
        )
        assert set(fake_client.bucket_api.uploaded_paths) == {
            "2026-07-10/niche_01/final_script.json",
            "2026-07-10/niche_01/report.md",
            "2026-07-10/niche_01/image_prompts/scene_01.json",
        }

    def test_preserves_folder_hierarchy_in_remote_paths(self, tmp_path):
        package_dir = _make_package(tmp_path)
        (package_dir / "animation_prompts").mkdir()
        (package_dir / "animation_prompts" / "scene_01.json").write_text("{}", encoding="utf-8")
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client)

        uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert "2026-07-10/niche_01/animation_prompts/scene_01.json" in fake_client.bucket_api.uploaded_paths

    def test_single_file_failure_does_not_abort_whole_package(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        real_upload = fake_client.bucket_api.upload
        calls = {"n": 0}

        def flaky_upload(path, file, file_options=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient network error")
            return real_upload(path, file, file_options=file_options)

        fake_client.bucket_api.upload = flaky_upload
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is False  # incomplet : pas 3/3
        assert result.uploaded_count == 2
        assert result.total_count == 3
        assert result.error is not None
        assert result.remote_url is not None  # le préfixe distant reste calculable

    def test_upload_failure_never_raises(self, tmp_path):
        package_dir = _make_package(tmp_path)
        broken_client = MagicMock()
        broken_client.storage.from_.return_value.upload.side_effect = RuntimeError("bucket not found")
        broken_client.storage.from_.return_value.get_public_url.return_value = "https://fake/url"
        uploader = SupabaseStorageUploader(client=broken_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is False
        assert result.uploaded_count == 0
        assert result.total_count == 3
        assert "bucket not found" in result.error
        assert "final_script.json" in result.error

    def test_empty_package_is_not_reported_as_success(self, tmp_path):
        package_dir = tmp_path / "niche_01"
        package_dir.mkdir()
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is False
        assert result.total_count == 0
        assert result.uploaded_count == 0

    def test_remote_url_falls_back_to_constructed_url_when_get_public_url_fails(self, tmp_path):
        package_dir = _make_package(tmp_path)
        client = MagicMock()
        client.storage.from_.return_value.upload.return_value = {"path": "ok"}
        client.storage.from_.return_value.exists.return_value = True
        client.storage.from_.return_value.get_public_url.side_effect = RuntimeError("boom")
        uploader = SupabaseStorageUploader(url="https://proj.supabase.co", key="k", client=client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.remote_url == (
            "https://proj.supabase.co/storage/v1/object/public/production/2026-07-10/niche_01"
        )


# ── Tests : nettoyage du dossier distant avant upload (Sprint 37) ───────────
# Bug de production confirmé : un deuxième run le même jour (retry manuel,
# double déclenchement du workflow) laissait les fichiers de l'ancien run
# mélangés avec les nouveaux — ex: scene_03.json (script 1, court) ET
# scene_03a.json/scene_03b.json (script 2, plus long, découpé) cohabitant
# pour la même scène. upload_package() doit maintenant vider le dossier
# distant avant d'écrire les fichiers du run courant.

class TestClearRemoteFolderBeforeUpload:
    def test_removes_stale_files_not_in_current_package(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient(
            existing_remote_files=[
                "2026-07-10/niche_01/final_script.json",
                "2026-07-10/niche_01/image_prompts/scene_02.json",  # orpheline d'un run precedent
            ],
        )
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is True
        assert set(fake_client.bucket_api.removed_paths) == {
            "2026-07-10/niche_01/final_script.json",
            "2026-07-10/niche_01/image_prompts/scene_02.json",
        }

    def test_no_removal_when_remote_folder_already_empty(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client)

        uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert fake_client.bucket_api.removed_paths == []

    def test_clear_failure_does_not_abort_upload(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient(
            existing_remote_files=["2026-07-10/niche_01/stale.json"],
        )
        fake_client.bucket_api.remove = MagicMock(side_effect=RuntimeError("delete failed"))
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is True
        assert result.uploaded_count == 3

    def test_list_failure_does_not_abort_upload(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        fake_client.bucket_api.list = MagicMock(side_effect=RuntimeError("list failed"))
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is True
        assert result.uploaded_count == 3


# ── Tests : vérification post-upload (Sprint 30.5) ───────────────────────────
# Bug de production confirmé : GitHub Actions rapportait "18/18 uploaded" et
# "20/20 uploaded" alors que le bucket était vide côté dashboard Supabase —
# upload() ne levait aucune exception, mais rien ne garantissait que l'objet
# existait réellement côté serveur après coup. Ces tests verrouillent le
# nouveau comportement : chaque upload est vérifié via exists(), et seul un
# objet CONFIRMÉ présent compte comme réussi.

class TestPostUploadVerification:
    def test_upload_reported_without_exception_but_object_missing_is_not_success(self, tmp_path):
        """Cas exact du bug : upload() ne lève rien, mais l'objet n'existe pas
        réellement — ne doit JAMAIS être compté comme un succès."""
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient(
            missing_paths={"2026-07-10/niche_01/final_script.json"},
        )
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is False
        assert result.uploaded_count == 2  # les 2 fichiers réellement vérifiés
        assert result.total_count == 3
        assert "final_script.json" in result.error
        assert "verification_failed" in result.error
        # upload() a bien été appelé pour les 3 fichiers — ce n'est pas
        # upload() qui a échoué, c'est la vérification après coup.
        assert len(fake_client.bucket_api.uploaded_paths) == 3

    def test_verification_exception_is_treated_as_not_uploaded(self, tmp_path):
        """Si exists() lève (ex: 404 non capturé en amont par le SDK), le
        fichier ne doit pas non plus compter comme réussi."""
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient(
            exists_raises_for={"2026-07-10/niche_01/report.md"},
        )
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is False
        assert result.uploaded_count == 2
        assert "report.md" in result.error
        assert "verification_failed" in result.error

    def test_all_verified_reports_full_success(self, tmp_path):
        """Chemin heureux : upload() réussit ET exists() confirme chaque objet."""
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client)

        result = uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert result.success is True
        assert result.uploaded_count == 3
        assert result.error is None
        # exists() a bien été appelé pour chacun des 3 fichiers uploadés.
        assert sorted(fake_client.bucket_api.exists_calls) == sorted(fake_client.bucket_api.uploaded_paths)


# ── Tests : préfixe distant ne duplique jamais le bucket (Sprint 30.5) ──────

class TestRemoteFolderNormalization:
    def test_duplicated_bucket_prefix_is_stripped(self, tmp_path):
        """Un appelant qui construit par erreur 'production/2026-07-10/niche_01'
        (bucket déjà nommé 'production') ne doit jamais produire la clé
        d'objet 'production/production/2026-07-10/niche_01/...'."""
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client, bucket="production")

        result = uploader.upload_package(package_dir, "production/2026-07-10/niche_01")

        assert result.success is True
        for path in fake_client.bucket_api.uploaded_paths:
            assert not path.startswith("production/production")
        assert "2026-07-10/niche_01/final_script.json" in fake_client.bucket_api.uploaded_paths
        assert result.remote_url == (
            "https://fake.supabase.co/storage/v1/object/public/production/2026-07-10/niche_01"
        )

    def test_bucket_name_alone_as_prefix_normalizes_to_empty(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client, bucket="production")

        uploader.upload_package(package_dir, "production")

        assert "final_script.json" in fake_client.bucket_api.uploaded_paths
        for path in fake_client.bucket_api.uploaded_paths:
            assert not path.startswith("production/")

    def test_non_duplicated_prefix_is_left_untouched(self, tmp_path):
        package_dir = _make_package(tmp_path)
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client, bucket="production")

        uploader.upload_package(package_dir, "2026-07-10/niche_01")

        assert "2026-07-10/niche_01/final_script.json" in fake_client.bucket_api.uploaded_paths


class TestFactory:
    def test_returns_noop_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

        uploader = build_storage_uploader()

        assert isinstance(uploader, NoOpStorageUploader)

    def test_returns_supabase_uploader_when_configured(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")

        def _fake_init(self, url="", key="", bucket=DEFAULT_BUCKET, client=None):
            self._bucket = bucket
            self._url = url
            self._client = _FakeSupabaseClient()

        monkeypatch.setattr(SupabaseStorageUploader, "__init__", _fake_init)

        uploader = build_storage_uploader()

        assert isinstance(uploader, SupabaseStorageUploader)

    def test_returns_noop_when_client_init_raises(self, monkeypatch):
        monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "fake-key")

        def _boom_init(self, url="", key="", bucket=DEFAULT_BUCKET, client=None):
            raise RuntimeError("network unreachable")

        monkeypatch.setattr(SupabaseStorageUploader, "__init__", _boom_init)

        uploader = build_storage_uploader()

        assert isinstance(uploader, NoOpStorageUploader)
