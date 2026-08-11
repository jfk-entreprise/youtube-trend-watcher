"""
Tests unitaires pour le Series Engine (src/series_engine.py — Sprint 39).
"""

import json
from unittest.mock import MagicMock
import pytest
from pathlib import Path

from src.series_engine import (
    CharacterSpec,
    EpisodeOutline,
    SeriesConcept,
    JsonSeriesStore,
    SupabaseSeriesStore,
    SeriesPlanner,
    series_concept_to_dict,
    series_concept_from_dict,
    build_series_store,
)


@pytest.fixture
def sample_character():
    return CharacterSpec(
        name="Victor",
        role="Protagoniste",
        visual_description="Homme de 30 ans, veste sombre",
        personality="Déterminé",
        voice_tone="Grave",
        permanent_visual_prompt="30 year old man, dark jacket, cinematic look"
    )


@pytest.fixture
def sample_outline():
    return EpisodeOutline(
        episode_number=1,
        title="Le Début",
        synopsis="Premier épisode d'introduction",
        key_events=["Rencontre", "Découverte"],
        cliffhanger="Qui est l'inconnu ?"
    )


@pytest.fixture
def sample_series(sample_character, sample_outline):
    return SeriesConcept(
        series_id="series_test_001",
        title="Série de Test",
        logline="Une série de test en 12 épisodes.",
        serie="IA",
        brand_id="ia_fr",
        market="FR",
        total_episodes=3,
        season_number=1,
        main_story_arc={"intro": "Début", "climax": "Fin"},
        character_bible=[sample_character],
        world_building={"setting": "Laboratoire"},
        episodes_roadmap=[
            sample_outline,
            EpisodeOutline(episode_number=2, title="Épisode 2", synopsis="Deuxième épisode"),
            EpisodeOutline(episode_number=3, title="Épisode 3", synopsis="Troisième épisode final")
        ],
        current_episode=1,
        status="active"
    )


class TestSeriesConceptModel:
    def test_properties(self, sample_series):
        assert not sample_series.is_completed
        assert sample_series.produced_episodes_count == 0
        assert sample_series.remaining_episodes_count == 3
        assert sample_series.current_episode_outline.episode_number == 1

    def test_serialization_roundtrip(self, sample_series):
        as_dict = series_concept_to_dict(sample_series)
        reconstructed = series_concept_from_dict(as_dict)

        assert reconstructed.series_id == sample_series.series_id
        assert reconstructed.title == sample_series.title
        assert len(reconstructed.character_bible) == 1
        assert reconstructed.character_bible[0].name == "Victor"
        assert len(reconstructed.episodes_roadmap) == 3
        assert reconstructed.episodes_roadmap[0].title == "Le Début"


class TestJsonSeriesStore:
    def test_save_and_load(self, tmp_path, sample_series):
        store_path = tmp_path / "series_concepts.json"
        store = JsonSeriesStore(store_path)

        # Avant sauvegarde
        assert store.load_active_series(serie="IA", market="FR") is None

        # Sauvegarde
        store.save_series(sample_series)

        # Après sauvegarde
        active = store.load_active_series(serie="IA", market="FR")
        assert active is not None
        assert active.series_id == sample_series.series_id
        assert active.title == "Série de Test"

        # Chargement par ID
        by_id = store.load_series_by_id("series_test_001")
        assert by_id is not None
        assert by_id.title == "Série de Test"

        # Liste
        all_series = store.list_series(market="FR")
        assert len(all_series) == 1


class TestSeriesPlanner:
    def test_pitch_heuristic(self, sample_series):
        planner = SeriesPlanner(llm_provider=None)

        opportunity = MagicMock()
        opportunity.niche = "Histoire"
        opportunity.topic = "Les Secrets des Pyramides"
        
        brand_profile = MagicMock()
        brand_profile.id = "histoire_fr"
        brand_profile.market = "FR"

        concept = planner.pitch_new_series(opportunity, brand_profile, niche_name="Histoire", total_episodes=12)

        assert concept is not None
        assert concept.serie == "Histoire"
        assert concept.brand_id == "histoire_fr"
        assert concept.total_episodes == 12
        assert len(concept.episodes_roadmap) == 12
        assert len(concept.character_bible) >= 2
        assert concept.current_episode == 1
        assert concept.status == "active"

    def test_get_next_episode_context(self, sample_series):
        planner = SeriesPlanner()
        ctx = planner.get_next_episode_context(sample_series)

        assert ctx["series_id"] == sample_series.series_id
        assert ctx["current_episode"] == 1
        assert ctx["total_episodes"] == 3
        assert ctx["remaining_episodes"] == 3
        assert "Aucun épisode précédent" in ctx["previous_episodes_recap"]
        assert len(ctx["character_bible"]) == 1
        assert ctx["character_bible"][0]["name"] == "Victor"

    def test_mark_episode_produced_and_progression(self, sample_series):
        planner = SeriesPlanner()

        # Marquer l'épisode 1 comme produit
        updated = planner.mark_episode_produced(sample_series, episode_number=1, script_summary="Victor découvre le labo.")
        assert updated.produced_episodes_count == 1
        assert updated.remaining_episodes_count == 2
        assert updated.current_episode == 2
        assert not updated.is_completed
        assert updated.episodes_roadmap[0].status == "produced"
        assert updated.episodes_roadmap[0].summary_produced == "Victor découvre le labo."

        # Contexte pour l'épisode 2
        ctx_ep2 = planner.get_next_episode_context(updated)
        assert ctx_ep2["current_episode"] == 2
        assert "Épisode 1 (Le Début): Victor découvre le labo." in ctx_ep2["previous_episodes_recap"]

        # Marquer l'épisode 2 puis 3
        planner.mark_episode_produced(updated, episode_number=2, script_summary="Révélation.")
        planner.mark_episode_produced(updated, episode_number=3, script_summary="Confrontation finale.")

        assert updated.produced_episodes_count == 3
        assert updated.remaining_episodes_count == 0
        assert updated.is_completed
        assert updated.status == "completed"

    def test_pitch_with_mocked_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "title": "La Révolution IA",
            "logline": "Une série passionnante sur l'IA.",
            "main_story_arc": {"intro": "Début", "climax": "Fin"},
            "character_bible": [
                {
                    "name": "Sarah Connor",
                    "role": "Résistante",
                    "visual_description": "Femme forte de 30 ans",
                    "personality": "Combative",
                    "voice_tone": "Ferme",
                    "permanent_visual_prompt": "30 year old strong woman, cinematic photorealistic"
                }
            ],
            "world_building": {"setting": "Futur proche"},
            "episodes_roadmap": [
                {
                    "episode_number": 1,
                    "title": "L'Éveil",
                    "synopsis": "Une IA s'éveille",
                    "key_events": ["Éveil"],
                    "cliffhanger": "Que va-t-elle faire ?"
                }
            ]
        })
        mock_llm.generate.return_value = mock_response

        planner = SeriesPlanner(llm_provider=mock_llm)
        opp = MagicMock(niche="IA", topic="Singularité")
        brand = MagicMock(id="ia_fr", name="IA FR", market="FR")

        concept = planner.pitch_new_series(opp, brand, niche_name="IA", total_episodes=1)
        assert concept.title == "La Révolution IA"
        assert len(concept.character_bible) == 1
        assert concept.character_bible[0].name == "Sarah Connor"
        assert len(concept.episodes_roadmap) == 1
