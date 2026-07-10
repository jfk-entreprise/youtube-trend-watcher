"""
Tests unitaires pour le module src.utils.

Couvre :
  - parse_iso_duration : toutes les formes ISO 8601
  - safe_int           : cas nominaux, limites, None
  - parse_dt           : formats avec Z, +00:00, sans fuseau
  - age_days           : valeurs calculées
  - fmt_duration       : formatage lisible
  - fmt_views          : formatage des vues (k, M, N/A)
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.utils import (parse_iso_duration, safe_int, parse_dt,
                        age_days, fmt_duration, fmt_views)


# ── parse_iso_duration ────────────────────────────────────────────────────────

class TestParseIsoDuration:

    def test_simple_seconds(self):
        assert parse_iso_duration("PT45S") == 45

    def test_minutes_and_seconds(self):
        assert parse_iso_duration("PT14M48S") == 888

    def test_hours_minutes_seconds(self):
        assert parse_iso_duration("PT1H30M15S") == 5415

    def test_hours_only(self):
        assert parse_iso_duration("PT2H") == 7200

    def test_zero_duration(self):
        assert parse_iso_duration("PT0S") == 0

    def test_empty_string(self):
        assert parse_iso_duration("") == 0

    def test_none_input(self):
        assert parse_iso_duration(None) == 0  # type: ignore

    def test_invalid_format(self):
        assert parse_iso_duration("abc") == 0

    def test_no_time_part(self):
        assert parse_iso_duration("P0D") == 0


# ── safe_int ──────────────────────────────────────────────────────────────────

class TestSafeInt:

    def test_simple_string(self):
        assert safe_int("42") == 42

    def test_already_int(self):
        assert safe_int(42) == 42

    def test_none_input(self):
        assert safe_int(None) is None

    def test_empty_string(self):
        assert safe_int("") is None

    def test_invalid_string(self):
        assert safe_int("not_a_number") is None

    def test_float_string(self):
        assert safe_int("3.14") is None  # int("3.14") lève ValueError

    def test_zero(self):
        assert safe_int(0) == 0

    def test_negative(self):
        assert safe_int("-100") == -100


# ── parse_dt ─────────────────────────────────────────────────────────────────

class TestParseDt:

    def test_with_z_suffix(self):
        dt = parse_dt("2026-06-21T10:10:13Z")
        assert dt.year == 2026
        assert dt.month == 6
        assert dt.day == 21
        assert dt.tzinfo is not None

    def test_with_offset(self):
        dt = parse_dt("2026-01-15T08:30:00+00:00")
        assert dt.hour == 8
        assert dt.minute == 30

    def test_without_timezone(self):
        dt = parse_dt("2026-03-10T14:45:00")
        assert dt.hour == 14
        # La date naive est traitée comme UTC

    def test_different_formats(self):
        dt1 = parse_dt("2026-06-21T10:10:13Z")
        dt2 = parse_dt("2026-06-21T10:10:13+00:00")
        assert dt1 == dt2


# ── age_days ──────────────────────────────────────────────────────────────────

class TestAgeDays:

    def test_recent_video(self):
        """Une vidéo publiée dans l'heure devrait avoir un âge < 0.05 jours."""
        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        age = age_days(recent)
        assert 0.0 < age < 0.05

    def test_old_video(self):
        """Une vidéo de 10 jours devrait avoir un âge ~10 jours."""
        old = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        age = age_days(old)
        assert 9.5 < age < 10.5

    def test_future_video(self):
        """Une date future ne devrait pas retourner de valeur négative."""
        future = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        age = age_days(future)
        assert age >= 0.0

    def test_invalid_date(self):
        """Une date invalide retourne la valeur de fallback (30.0)."""
        age = age_days("not_a_date")
        assert age == 30.0


# ── fmt_duration ──────────────────────────────────────────────────────────────

class TestFmtDuration:

    def test_seconds_only(self):
        assert fmt_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        assert fmt_duration(125) == "2m05s"

    def test_hours_minutes_seconds(self):
        assert fmt_duration(3661) == "1h01m01s"

    def test_exact_minute(self):
        assert fmt_duration(60) == "1m00s"

    def test_exact_hour(self):
        assert fmt_duration(3600) == "1h00m00s"

    def test_zero(self):
        assert fmt_duration(0) == "0s"

    def test_large_duration(self):
        assert fmt_duration(10000) == "2h46m40s"


# ── fmt_views ─────────────────────────────────────────────────────────────────

class TestFmtViews:

    def test_none(self):
        assert fmt_views(None) == "N/A"

    def test_small_number(self):
        assert fmt_views(42) == "42"

    def test_thousands(self):
        assert fmt_views(1500) == "1.5k"

    def test_exact_thousand(self):
        assert fmt_views(1000) == "1.0k"

    def test_millions(self):
        assert fmt_views(2_500_000) == "2.5M"

    def test_exact_million(self):
        assert fmt_views(1_000_000) == "1.0M"

    def test_billions(self):
        assert fmt_views(1_500_000_000) == "1500.0M"
