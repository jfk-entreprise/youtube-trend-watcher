from src.brand_engine import BrandEngine, JsonBrandStore


def _make_profile_dict(niche_keywords=None):
    data = {
        "id": "test_fr",
        "name": "Test FR",
        "description": "Chaîne de test.",
        "niche": "Test",
        "target_audience": "Testeurs",
        "primary_language": "fr",
        "tone": "Neutre",
        "personality": "Neutre",
        "writing_style": "Neutre",
        "emotion_level": 0.5,
        "humor_level": 0.5,
        "authority_level": 0.5,
        "curiosity_level": 0.5,
        "storytelling_level": 0.5,
        "voice_speed": "Modéré",
        "preferred_video_duration": 600,
        "preferred_formats": ["Analyse"],
        "preferred_hooks": ["Hook ?"],
        "preferred_cta": ["Abonne-toi."],
        "forbidden_words": [],
        "visual_style": "",
        "color_palette": [],
        "typography_style": "",
        "logo_description": "",
        "thumbnail_style": "",
        "metadata": {},
    }
    if niche_keywords is not None:
        data["niche_keywords"] = niche_keywords
    return data


def _make_profile_dict_with_market(market):
    data = _make_profile_dict()
    data["market"] = market
    return data


class TestNicheKeywordsDefault:
    def test_missing_niche_keywords_defaults_to_empty_list(self, tmp_path):
        (tmp_path / "test_fr.json").write_text(
            __import__("json").dumps(_make_profile_dict()), encoding="utf-8"
        )
        engine = BrandEngine(brands_dir=tmp_path)

        profile = engine.load("test_fr")

        assert profile.niche_keywords == []

    def test_existing_brand_files_still_load(self):
        engine = BrandEngine()
        profile = engine.load("ia_fr")

        assert profile is not None
        assert isinstance(profile.niche_keywords, list)


class TestNicheKeywordsRoundtrip:
    def test_niche_keywords_persisted_and_reloaded(self, tmp_path):
        (tmp_path / "test_fr.json").write_text(
            __import__("json").dumps(_make_profile_dict(niche_keywords=["ia", "tech"])),
            encoding="utf-8",
        )
        engine = BrandEngine(brands_dir=tmp_path)

        profile = engine.load("test_fr")
        assert profile.niche_keywords == ["ia", "tech"]

        store = JsonBrandStore(tmp_path)
        store.save(profile)
        reloaded = store.load("test_fr")
        assert reloaded.niche_keywords == ["ia", "tech"]


class TestMarket:
    """Sprint 34 — champ market (marché ciblé par la marque)."""

    def test_missing_market_defaults_to_fr(self, tmp_path):
        (tmp_path / "test_fr.json").write_text(
            __import__("json").dumps(_make_profile_dict()), encoding="utf-8"
        )
        engine = BrandEngine(brands_dir=tmp_path)

        profile = engine.load("test_fr")

        assert profile.market == "FR"

    def test_market_us_is_parsed(self, tmp_path):
        (tmp_path / "test_us.json").write_text(
            __import__("json").dumps(_make_profile_dict_with_market("US")), encoding="utf-8"
        )
        engine = BrandEngine(brands_dir=tmp_path)

        profile = engine.load("test_us")

        assert profile.market == "US"

    def test_market_persisted_and_reloaded(self, tmp_path):
        (tmp_path / "test_us.json").write_text(
            __import__("json").dumps(_make_profile_dict_with_market("US")), encoding="utf-8"
        )
        store = JsonBrandStore(tmp_path)

        profile = store.load("test_us")
        store.save(profile)
        reloaded = store.load("test_us")

        assert reloaded.market == "US"

    def test_existing_brand_files_default_to_fr(self):
        engine = BrandEngine()
        profile = engine.load("ia_fr")

        assert profile is not None
        assert profile.market == "FR"
