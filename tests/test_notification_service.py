from unittest.mock import MagicMock, patch

from src.notification_service import (
    ChannelSummary,
    DailyProductionSummary,
    LoggingNotificationService,
    TelegramNotificationService,
    _format_duration,
    build_notification_service,
    format_summary_text,
)


def _summary(with_links=False, duration=684.0) -> DailyProductionSummary:
    return DailyProductionSummary(
        date="2026-07-10",
        channels=[
            ChannelSummary(
                niche_name="IA", channel_name="IA FR", subject="L'IA va-t-elle nous remplacer ?",
                duration_seconds=90, scene_count=9,
                drive_link="https://drive.google.com/drive/folders/abc123" if with_links else None,
            ),
            ChannelSummary(
                niche_name="Histoire", channel_name="Histoire FR", subject="La chute de Rome",
                duration_seconds=120, scene_count=10,
                drive_link="https://drive.google.com/drive/folders/def456" if with_links else None,
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


class TestFormatSummaryTextDriveLink:
    def test_message_with_drive_links(self):
        text = format_summary_text(_summary(with_links=True))

        assert "📁 Production package" in text
        assert "https://drive.google.com/drive/folders/abc123" in text
        assert "https://drive.google.com/drive/folders/def456" in text
        # Plusieurs liens → préfixés par niche pour rester lisibles.
        assert "• IA: https://drive.google.com/drive/folders/abc123" in text

    def test_message_without_drive_link_omits_section(self):
        text = format_summary_text(_summary(with_links=False))

        assert "📁 Production package" not in text
        assert "drive.google.com" not in text

    def test_single_link_shown_without_niche_prefix(self):
        summary = _summary(with_links=False)
        summary.channels = [summary.channels[0]]
        summary.channels[0].drive_link = "https://drive.google.com/drive/folders/solo"

        text = format_summary_text(summary)

        assert "📁 Production package" in text
        lines = text.splitlines()
        idx = lines.index("📁 Production package")
        assert lines[idx + 1] == "https://drive.google.com/drive/folders/solo"


class TestLoggingNotificationService:
    def test_does_not_raise(self):
        service = LoggingNotificationService()
        service.send_daily_summary(_summary())  # no exception = pass


class TestTelegramNotificationService:
    def test_sends_expected_payload_with_drive_link(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=fake_response) as mock_post:
            service.send_daily_summary(_summary(with_links=True))

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://api.telegram.org/bot123:ABC/sendMessage"
        assert kwargs["data"]["chat_id"] == "42"
        assert "drive.google.com" in kwargs["data"]["text"]

    def test_sends_expected_payload_without_drive_link(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=fake_response) as mock_post:
            service.send_daily_summary(_summary(with_links=False))

        _, kwargs = mock_post.call_args
        assert "drive.google.com" not in kwargs["data"]["text"]

    def test_network_failure_does_not_raise(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")

        with patch("requests.post", side_effect=ConnectionError("network down")):
            service.send_daily_summary(_summary())  # no exception = pass

    def test_http_error_status_does_not_raise(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")
        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = Exception("400 Bad Request")

        with patch("requests.post", return_value=fake_response):
            service.send_daily_summary(_summary())  # no exception = pass

    def test_timeout_does_not_raise(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")

        with patch("requests.post", side_effect=TimeoutError("timed out")):
            service.send_daily_summary(_summary())  # no exception = pass

    def test_formatting_failure_does_not_raise(self):
        service = TelegramNotificationService(bot_token="123:ABC", chat_id="42")
        broken_summary = MagicMock()
        broken_summary.channels = None  # itérer sur None lève TypeError dans format_summary_text

        with patch("requests.post") as mock_post:
            service.send_daily_summary(broken_summary)  # no exception = pass

        mock_post.assert_not_called()


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
