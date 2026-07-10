from unittest.mock import MagicMock, patch

import requests

from src.notification_service import (
    ChannelSummary,
    DailyProductionSummary,
    LoggingNotificationService,
    NotificationResult,
    TelegramNotificationService,
    _format_duration,
    build_notification_service,
    format_summary_text,
)


def _fake_response(status_code=200, description=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"ok": status_code == 200, "description": description}
    response.text = description or ""
    return response


def _summary(with_links=False, duration=684.0) -> DailyProductionSummary:
    return DailyProductionSummary(
        date="2026-07-10",
        channels=[
            ChannelSummary(
                niche_name="IA", channel_name="IA FR", subject="L'IA va-t-elle nous remplacer ?",
                duration_seconds=90, scene_count=9,
                storage_link="https://proj.supabase.co/storage/v1/object/public/production/abc123" if with_links else None,
            ),
            ChannelSummary(
                niche_name="Histoire", channel_name="Histoire FR", subject="La chute de Rome",
                duration_seconds=120, scene_count=10,
                storage_link="https://proj.supabase.co/storage/v1/object/public/production/def456" if with_links else None,
            ),
        ],
        pipeline_duration_seconds=duration,
    )


class TestFormatDuration:
    def test_minutes_and_seconds(self):
        assert _format_duration(684) == "11m 24s"

    def test_seconds_only_under_a_minute(self):
        assert _format_duration(45) == "45s"

    def test_hours_minutes_seconds(self):
        assert _format_duration(3725) == "1h 2m 5s"


class TestFormatSummaryTextStructure:
    def test_contains_expected_sections(self):
        text = format_summary_text(_summary())

        assert text.startswith("🎬 Daily Production Ready")
        assert "✅ Pipeline completed successfully" in text
        assert "📈 Active niches" in text
        assert "• IA" in text
        assert "• Histoire" in text
        assert "🎥 Videos generated" in text
        assert "\n2\n" in text
        assert "🖼 Image prompts" in text
        assert "🎞 Animation prompts" in text
        assert "Ready" in text
        assert "⏱ Pipeline duration" in text
        assert "11m 24s" in text

    def test_is_compact_no_blank_line_bloat(self):
        text = format_summary_text(_summary())
        # Pas plus d'une ligne vide consécutive (lisible sur mobile).
        assert "\n\n\n" not in text

    def test_omits_duration_section_when_unknown(self):
        summary = _summary()
        summary.pipeline_duration_seconds = None

        text = format_summary_text(summary)

        assert "⏱ Pipeline duration" not in text


class TestFormatSummaryTextStorageLink:
    def test_message_always_names_supabase_storage(self):
        # Le bloc "📦 Storage: / Supabase Storage" doit apparaître même sans lien
        # (upload non configuré/échoué) — le résumé reste utile, jamais silencieux.
        text = format_summary_text(_summary(with_links=False))

        assert "📦 Storage:" in text
        assert "Supabase Storage" in text

    def test_message_with_storage_links(self):
        text = format_summary_text(_summary(with_links=True))

        assert "https://proj.supabase.co/storage/v1/object/public/production/abc123" in text
        assert "https://proj.supabase.co/storage/v1/object/public/production/def456" in text
        # Plusieurs liens → préfixés par niche pour rester lisibles.
        assert "• IA: https://proj.supabase.co/storage/v1/object/public/production/abc123" in text

    def test_message_without_storage_link_omits_link_lines(self):
        text = format_summary_text(_summary(with_links=False))

        assert "supabase.co/storage" not in text

    def test_single_link_shown_without_niche_prefix(self):
        summary = _summary(with_links=False)
        summary.channels = [summary.channels[0]]
        summary.channels[0].storage_link = "https://proj.supabase.co/storage/v1/object/public/production/solo"

        text = format_summary_text(summary)

        lines = text.splitlines()
        idx = lines.index("Supabase Storage")
        assert lines[idx + 1] == "https://proj.supabase.co/storage/v1/object/public/production/solo"


class TestLoggingNotificationService:
    def test_does_not_raise(self):
        service = LoggingNotificationService()
        service.send_daily_summary(_summary())  # no exception = pass

    def test_reports_not_configured_status(self):
        service = LoggingNotificationService(reason="not_configured")
        result = service.send_daily_summary(_summary())
        assert isinstance(result, NotificationResult)
        assert result.success is False
        assert result.status == "not_configured"

    def test_reports_missing_secret_status(self):
        service = LoggingNotificationService(reason="missing_secret")
        result = service.send_daily_summary(_summary())
        assert result.status == "missing_secret"

    def test_formatting_failure_reports_formatting_error(self):
        service = LoggingNotificationService()
        broken_summary = MagicMock()
        broken_summary.channels = None

        result = service.send_daily_summary(broken_summary)

        assert result.success is False
        assert result.status == "formatting_error"


class TestTelegramNotificationService:
    def test_sends_expected_payload_with_storage_link(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")

        with patch("requests.post", return_value=_fake_response(200)) as mock_post:
            result = service.send_daily_summary(_summary(with_links=True))

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.telegram.org/bot123:ABC/sendMessage"
        assert kwargs["data"]["chat_id"] == "42"
        assert "supabase.co/storage" in kwargs["data"]["text"]
        assert "Supabase Storage" in kwargs["data"]["text"]
        assert result.success is True
        assert result.status == "sent"

    def test_sends_expected_payload_without_storage_link(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")

        with patch("requests.post", return_value=_fake_response(200)) as mock_post:
            result = service.send_daily_summary(_summary(with_links=False))

        _, kwargs = mock_post.call_args
        assert "supabase.co/storage" not in kwargs["data"]["text"]
        assert "Supabase Storage" in kwargs["data"]["text"]
        assert result.status == "sent"

    def test_network_failure_reports_network_error(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")

        with patch("requests.post", side_effect=requests.exceptions.ConnectionError("network down")):
            result = service.send_daily_summary(_summary())  # no exception = pass

        assert result.success is False
        assert result.status == "network_error"

    def test_invalid_bot_token_reports_precise_status(self):
        service = TelegramNotificationService(bot_token="bad-token", chat_id="42")

        with patch("requests.post", return_value=_fake_response(401, "Unauthorized")):
            result = service.send_daily_summary(_summary())

        assert result.success is False
        assert result.status == "invalid_bot_token"
        assert "Unauthorized" in result.detail

    def test_invalid_chat_id_reports_precise_status(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="0")

        with patch("requests.post", return_value=_fake_response(400, "Bad Request: chat not found")):
            result = service.send_daily_summary(_summary())

        assert result.success is False
        assert result.status == "invalid_chat_id"
        assert "chat not found" in result.detail

    def test_other_http_error_reports_generic_http_error(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")

        with patch("requests.post", return_value=_fake_response(500, "Internal Server Error")):
            result = service.send_daily_summary(_summary())

        assert result.success is False
        assert result.status == "http_error"

    def test_timeout_reports_timeout_status(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")

        with patch("requests.post", side_effect=requests.exceptions.Timeout("timed out")):
            result = service.send_daily_summary(_summary())  # no exception = pass

        assert result.success is False
        assert result.status == "timeout"

    def test_formatting_failure_does_not_raise(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")
        broken_summary = MagicMock()
        broken_summary.channels = None  # itérer sur None lève TypeError dans format_summary_text

        with patch("requests.post") as mock_post:
            result = service.send_daily_summary(broken_summary)  # no exception = pass

        mock_post.assert_not_called()
        assert result.status == "formatting_error"


class TestFactory:
    def test_returns_telegram_when_configured(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

        service = build_notification_service()

        assert isinstance(service, TelegramNotificationService)

    def test_returns_logging_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        service = build_notification_service()

        assert isinstance(service, LoggingNotificationService)

    def test_returns_logging_with_missing_secret_reason_when_partially_configured(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

        service = build_notification_service()

        assert isinstance(service, LoggingNotificationService)
        result = service.send_daily_summary(_summary())
        assert result.status == "missing_secret"
