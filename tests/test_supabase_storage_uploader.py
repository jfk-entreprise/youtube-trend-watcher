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

        result = uploader.upload_package(tmp_path, "production/2026-07-10/niche_01")

        assert isinstance(result, UploadResult)
        assert result.success is False
        assert result.uploaded_count == 0
        assert result.total_count == 0
        assert result.remote_url is None
        assert result.error == "not_configured"


class _FakeBucketAPI:
    """Simule client.storage.from_(bucket) — upload()/get_public_url()."""

    def __init__(self):
        self.uploaded_paths = []

    def upload(self, path, file, file_options=None):
        self.uploaded_paths.append(path)
        return {"path": path}

    def get_public_url(self, path, options=None):
        return f"https://fake.supabase.co/storage/v1/object/public/{DEFAULT_BUCKET}/{path}"


class _FakeStorageNamespace:
    def __init__(self, bucket_api):
        self._bucket_api = bucket_api

    def from_(self, bucket_name):
        return self._bucket_api


class _FakeSupabaseClient:
    def __init__(self):
        self.bucket_api = _FakeBucketAPI()
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

        result = uploader.upload_package(package_dir, "production/2026-07-10/niche_01")

        assert result.success is True
        assert result.uploaded_count == 3
        assert result.total_count == 3
        assert result.error is None
        assert result.remote_url == (
            "https://fake.supabase.co/storage/v1/object/public/production/production/2026-07-10/niche_01"
        )
        assert set(fake_client.bucket_api.uploaded_paths) == {
            "production/2026-07-10/niche_01/final_script.json",
            "production/2026-07-10/niche_01/report.md",
            "production/2026-07-10/niche_01/image_prompts/scene_01.json",
        }

    def test_preserves_folder_hierarchy_in_remote_paths(self, tmp_path):
        package_dir = _make_package(tmp_path)
        (package_dir / "animation_prompts").mkdir()
        (package_dir / "animation_prompts" / "scene_01.json").write_text("{}", encoding="utf-8")
        fake_client = _FakeSupabaseClient()
        uploader = SupabaseStorageUploader(client=fake_client)

        uploader.upload_package(package_dir, "production/2026-07-10/niche_01")

        assert "production/2026-07-10/niche_01/animation_prompts/scene_01.json" in fake_client.bucket_api.uploaded_paths

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

        result = uploader.upload_package(package_dir, "production/2026-07-10/niche_01")

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

        result = uploader.upload_package(package_dir, "production/2026-07-10/niche_01")

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

        result = uploader.upload_package(package_dir, "production/2026-07-10/niche_01")

        assert result.success is False
        assert result.total_count == 0
        assert result.uploaded_count == 0

    def test_remote_url_falls_back_to_constructed_url_when_get_public_url_fails(self, tmp_path):
        package_dir = _make_package(tmp_path)
        client = MagicMock()
        client.storage.from_.return_value.upload.return_value = {"path": "ok"}
        client.storage.from_.return_value.get_public_url.side_effect = RuntimeError("boom")
        uploader = SupabaseStorageUploader(url="https://proj.supabase.co", key="k", client=client)

        result = uploader.upload_package(package_dir, "production/2026-07-10/niche_01")

        assert result.remote_url == (
            "https://proj.supabase.co/storage/v1/object/public/production/production/2026-07-10/niche_01"
        )


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
