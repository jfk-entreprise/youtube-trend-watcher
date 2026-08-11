"""
Series Engine — Gestion des séries vidéo épisodiques et continuité narrative.

Ce moteur transforme la production de vidéos individuelles en séries complètes
(ex: 12 épisodes par saison) dotées d'une bible de personnages permanente,
d'un arc narratif structuré et d'une continuité d'un épisode à l'autre.

Composants :
  - CharacterSpec      : définition d'un personnage récurrent (visuel + personnalité).
  - EpisodeOutline     : fiche de cadrage d'un épisode dans la roadmap de la série.
  - SeriesConcept      : concept complet d'une série (bible, saison, roadmap, état).
  - SeriesStore (ABC)  : interface de persistance (JsonSeriesStore / SupabaseSeriesStore).
  - SeriesPlanner      : orchestrateur IA / heuristique pour créer et faire avancer les séries.
"""

import dataclasses
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_EPISODES_PER_SEASON = 12
DEFAULT_JSON_FALLBACK_PATH = Path(".cache/series_concepts.json")


# ── CharacterSpec ─────────────────────────────────────────────────────────────

@dataclass
class CharacterSpec:
    """Spécification d'un personnage récurrent dans la série."""
    name: str
    role: str                       # ex: "Protagoniste", "Rival", "Narrateur", "Expert"
    visual_description: str         # Description physique précise (âge, cheveux, vêtements, style)
    personality: str                # Traits de caractère, attitude, valeurs
    voice_tone: str                 # Ton de voix, style d'élocution
    permanent_visual_prompt: str    # Description visuelle anglaise optimisée pour Flux/SDXL


# ── EpisodeOutline ────────────────────────────────────────────────────────────

@dataclass
class EpisodeOutline:
    """Fiche de cadrage d'un épisode au sein de la série."""
    episode_number: int             # 1-indexed (1, 2, ..., N)
    title: str
    synopsis: str                   # Résumé de l'intrigue de l'épisode
    key_events: List[str] = field(default_factory=list)  # Rebondissements clés
    cliffhanger: str = ""           # Élément de tension ou accroche pour l'épisode suivant
    status: str = "pending"         # 'pending' | 'produced'
    summary_produced: Optional[str] = None  # Résumé effectif après production du script
    produced_date: Optional[str] = None     # Date ISO (YYYY-MM-DD) de production


# ── SeriesConcept ─────────────────────────────────────────────────────────────

@dataclass
class SeriesConcept:
    """Concept global et état d'avancement d'une série vidéo."""
    series_id: str                  # Identifiant unique (ex: "series_ai_war_fr_001")
    title: str                      # Titre officiel de la série
    logline: str                    # Pitch accrocheur de la série
    serie: str                      # Identifiant de la série (ex: "IA", "Histoire", "Finance")
    brand_id: str                   # Profil de marque (ex: "ia_fr")
    market: str = "FR"              # Marché cible ("FR" | "US")
    total_episodes: int = DEFAULT_EPISODES_PER_SEASON
    season_number: int = 1
    is_multi_season: bool = False
    main_story_arc: Dict[str, str] = field(default_factory=dict)   # {"intro": ..., "climax": ..., "resolution": ...}
    character_bible: List[CharacterSpec] = field(default_factory=list)
    world_building: Dict[str, str] = field(default_factory=dict)   # {"setting": ..., "visual_theme": ..., "mood": ...}
    episodes_roadmap: List[EpisodeOutline] = field(default_factory=list)
    current_episode: int = 1
    status: str = "active"          # 'active' | 'completed' | 'archived'
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.status == "completed" or self.current_episode > self.total_episodes

    @property
    def produced_episodes_count(self) -> int:
        return sum(1 for ep in self.episodes_roadmap if ep.status == "produced")

    @property
    def remaining_episodes_count(self) -> int:
        return max(0, self.total_episodes - self.produced_episodes_count)

    @property
    def current_episode_outline(self) -> Optional[EpisodeOutline]:
        """Retourne la fiche de l'épisode courant. Cherche d'abord par numéro exact,
        puis par premier épisode encore 'pending' si la roadmap est désynchronisée."""
        for ep in self.episodes_roadmap:
            if ep.episode_number == self.current_episode:
                return ep
        # Fallback conservatif : la roadmap existe mais n'est pas initialisée avec le bon numéro
        for ep in self.episodes_roadmap:
            if ep.status == "pending":
                return ep
        return None


# ── Serializer & Deserializer ─────────────────────────────────────────────────

def series_concept_to_dict(series: SeriesConcept) -> Dict[str, Any]:
    """Convertit un SeriesConcept en dictionnaire JSON-serializable."""
    data = dataclasses.asdict(series)
    return data


def series_concept_from_dict(data: Dict[str, Any]) -> SeriesConcept:
    """Reconstruit un SeriesConcept à partir d'un dictionnaire JSON."""
    data_copy = dict(data)
    
    # Reconstitution des CharacterSpec
    raw_chars = data_copy.pop("character_bible", [])
    character_bible = [
        c if isinstance(c, CharacterSpec) else CharacterSpec(**c)
        for c in raw_chars
    ]
    
    # Reconstitution des EpisodeOutline
    raw_eps = data_copy.pop("episodes_roadmap", [])
    episodes_roadmap = [
        e if isinstance(e, EpisodeOutline) else EpisodeOutline(**e)
        for e in raw_eps
    ]

    return SeriesConcept(
        character_bible=character_bible,
        episodes_roadmap=episodes_roadmap,
        **data_copy
    )


# ── SeriesStore (ABC) ─────────────────────────────────────────────────────────

class SeriesStore(ABC):
    """Interface abstraite de persistance des concepts de séries."""

    @abstractmethod
    def load_active_series(self, serie: Optional[str] = None, market: str = "FR") -> Optional[SeriesConcept]:
        """Retourne la série actuellement active pour un identifiant de série et un marché donnés."""
        ...

    @abstractmethod
    def save_series(self, series: SeriesConcept) -> None:
        """Sauvegarde ou met à jour l'état d'une série."""
        ...

    @abstractmethod
    def list_series(self, market: str = "FR") -> List[SeriesConcept]:
        """Retourne la liste de toutes les séries enregistrées."""
        ...

    @abstractmethod
    def load_series_by_id(self, series_id: str) -> Optional[SeriesConcept]:
        """Charge une série par son identifiant unique."""
        ...


class JsonSeriesStore(SeriesStore):
    """Persistance JSON locale sur disque."""

    def __init__(self, path: Any = DEFAULT_JSON_FALLBACK_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> List[SeriesConcept]:
        if not self._path.exists():
            return []
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return [series_concept_from_dict(item) for item in raw]
        except Exception as exc:
            logger.warning("Lecture de '%s' impossible (%s) — store vide.", self._path, exc)
            return []

    def _write_all(self, series_list: List[SeriesConcept]) -> None:
        payload = [series_concept_to_dict(s) for s in series_list]
        self._path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_active_series(self, serie: Optional[str] = None, market: str = "FR") -> Optional[SeriesConcept]:
        series_list = self._read_all()
        for s in series_list:
            if s.market.upper() == market.upper() and s.status == "active":
                if serie is None or s.serie.lower() == serie.lower():
                    return s
        return None

    def save_series(self, series: SeriesConcept) -> None:
        series_list = self._read_all()
        series.updated_at = datetime.now(timezone.utc).isoformat()
        
        updated = False
        for idx, item in enumerate(series_list):
            if item.series_id == series.series_id:
                series_list[idx] = series
                updated = True
                break
        
        if not updated:
            series_list.append(series)

        self._write_all(series_list)
        logger.info("Série '%s' (%s) sauvegardée dans %s.", series.title, series.series_id, self._path.name)

    def list_series(self, market: str = "FR") -> List[SeriesConcept]:
        return [s for s in self._read_all() if s.market.upper() == market.upper()]

    def load_series_by_id(self, series_id: str) -> Optional[SeriesConcept]:
        for s in self._read_all():
            if s.series_id == series_id:
                return s
        return None


class SupabaseSeriesStore(SeriesStore):
    """Persistance Supabase — table `series_concepts`."""

    def __init__(self, url: str, key: str, fallback_path: Any = DEFAULT_JSON_FALLBACK_PATH) -> None:
        self._url = url
        self._key = key
        self._fallback = JsonSeriesStore(fallback_path)
        self._client = None
        try:
            from supabase import create_client  # type: ignore[import-untyped]
            self._client = create_client(url, key)
        except Exception as exc:
            logger.warning("Connexion Supabase impossible pour SeriesStore (%s) — fallback JSON.", exc)

    def load_active_series(self, serie: Optional[str] = None, market: str = "FR") -> Optional[SeriesConcept]:
        if not self._client:
            return self._fallback.load_active_series(serie, market)
        try:
            query = self._client.table("series_concepts").select("*").eq("market", market.upper()).eq("status", "active")
            if serie:
                query = query.eq("serie", serie.lower())  # minuscules — cohérent avec le payload d'upsert
            res = query.order("created_at", desc=True).limit(1).execute()
            if res.data:
                row = res.data[0]
                concept_json = row.get("concept_json")
                if isinstance(concept_json, str):
                    concept_json = json.loads(concept_json)
                return series_concept_from_dict(concept_json)
        except Exception as exc:
            logger.warning("Erreur Supabase load_active_series (%s) — fallback JSON.", exc)
            return self._fallback.load_active_series(serie, market)
        return None

    def save_series(self, series: SeriesConcept) -> None:
        # updated_at est assigné par JsonSeriesStore.save_series — pas besoin de le dupliquer ici.
        self._fallback.save_series(series)   # <-- positionne series.updated_at
        if not self._client:
            return
        try:
            payload = {
                "series_id": series.series_id,
                "title": series.title,
                "serie": series.serie.lower(),   # normalisé en minuscules pour cohérence avec la recherche
                "brand_id": series.brand_id,
                "market": series.market.upper(),
                "season_number": series.season_number,
                "total_episodes": series.total_episodes,
                "current_episode": series.current_episode,
                "status": series.status,
                "concept_json": series_concept_to_dict(series),
                "updated_at": series.updated_at   # déjà mis à jour par le fallback ci-dessus
            }
            self._client.table("series_concepts").upsert(payload).execute()
            logger.info("Série '%s' synchronisée sur Supabase.", series.title)
        except Exception as exc:
            logger.warning("Échec upsert Supabase series_concepts (%s).", exc)

    def list_series(self, market: str = "FR") -> List[SeriesConcept]:
        if not self._client:
            return self._fallback.list_series(market)
        try:
            res = self._client.table("series_concepts").select("*").eq("market", market.upper()).execute()
            if res.data:
                result = []
                for row in res.data:
                    c_json = row.get("concept_json")
                    if isinstance(c_json, str):
                        c_json = json.loads(c_json)
                    result.append(series_concept_from_dict(c_json))
                return result
        except Exception as exc:
            logger.warning("Erreur Supabase list_series (%s) — fallback JSON.", exc)
        return self._fallback.list_series(market)

    def load_series_by_id(self, series_id: str) -> Optional[SeriesConcept]:
        if not self._client:
            return self._fallback.load_series_by_id(series_id)
        try:
            res = self._client.table("series_concepts").select("*").eq("series_id", series_id).limit(1).execute()
            if res.data:
                c_json = res.data[0].get("concept_json")
                if isinstance(c_json, str):
                    c_json = json.loads(c_json)
                return series_concept_from_dict(c_json)
        except Exception as exc:
            logger.warning("Erreur Supabase load_series_by_id (%s) — fallback JSON.", exc)
        return self._fallback.load_series_by_id(series_id)


def build_series_store(json_path: Any = DEFAULT_JSON_FALLBACK_PATH) -> SeriesStore:
    """Factory instantiant SupabaseSeriesStore ou JsonSeriesStore selon l'environnement."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        return SupabaseSeriesStore(url, key, json_path)
    return JsonSeriesStore(json_path)


# ── SeriesPlanner ─────────────────────────────────────────────────────────────

class SeriesPlanner:
    """
    Moteur de création de séries et de préparation du contexte d'épisode.
    """

    def __init__(self, llm_provider: Optional[Any] = None) -> None:
        self._llm = llm_provider

    def pitch_new_series(
        self,
        opportunity: Any,
        brand_profile: Any,
        niche_name: str,
        total_episodes: int = DEFAULT_EPISODES_PER_SEASON,
        season_number: int = 1,
    ) -> SeriesConcept:
        """
        Génère un concept complet de série (12 épisodes, bible personnages, arc narratif).
        Utilise le LLM Provider si présent, sinon retombe sur le générateur heuristique.

        `niche_name` doit être la même clé que celle utilisée pour retrouver la série
        active (`SeriesStore.load_active_series(serie=...)`) — typiquement `niche.name`,
        et non `opportunity.niche` (qui est en réalité le `primary_topic` LLM par vidéo,
        pas l'identifiant de niche stable).
        """
        if self._llm is not None:
            try:
                concept = self._pitch_with_llm(opportunity, brand_profile, niche_name, total_episodes, season_number)
                if concept:
                    return concept
            except Exception as exc:
                logger.warning("Génération LLM de la série a échoué (%s) — fallback heuristique.", exc)

        return self._pitch_heuristic(opportunity, brand_profile, niche_name, total_episodes, season_number)

    def _pitch_heuristic(
        self,
        opportunity: Any,
        brand_profile: Any,
        niche_name: str,
        total_episodes: int,
        season_number: int,
    ) -> SeriesConcept:
        """Générateur heuristique de secours (robuste, zéro dépendance API)."""
        niche_label = niche_name
        topic = getattr(opportunity, "topic", "Les Mystères du Futur")
        brand_id = getattr(brand_profile, "id", "brand_default")
        market = getattr(brand_profile, "market", "FR")

        series_id = f"series_{brand_id}_{niche_label.lower()}_s{season_number}_{int(datetime.now(timezone.utc).timestamp())}"
        title = f"{niche_label} : La Saga de {topic[:30]}"
        logline = f"Une immersion captivante en {total_episodes} épisodes pour percer les secrets de {topic}."

        main_story_arc = {
            "intro": f"Découverte des enjeux initiaux de {topic}.",
            "development": f"Montée en puissance des révélations et conflits centraux.",
            "climax": f"La grande confrontation et le tournant majeur de la saison.",
            "resolution": f"Résolution et ouverture vers les perspectives futures."
        }

        character_bible = [
            CharacterSpec(
                name="Alex Vance",
                role="Protagoniste & Enquêteur",
                visual_description="Homme de 32 ans, veste en cuir brun sombre, regard déterminé, cheveux courts châtains.",
                personality="Curieux, tenace, sceptique mais passionné.",
                voice_tone="Voix captivante, posée et assertive.",
                permanent_visual_prompt="32 year old man, dark brown leather jacket, short chestnut hair, determined intense eyes, Arcane character design, painterly stylized illustration, hand-painted textures, cinematic lighting"
            ),
            CharacterSpec(
                name="Dr. Elena Rostova",
                role="Experte & Mentore",
                visual_description="Femme de 45 ans, lunettes fines à monture métallique, manteau élégant gris, cheveux poivre et sel tirés en arrière.",
                personality="Brillante, méthodique, mystérieuse.",
                voice_tone="Voix calme, docte et rassurante.",
                permanent_visual_prompt="45 year old woman, thin metal frame glasses, elegant grey coat, salt and pepper hair tied back, intellectual aesthetic, soft studio lighting"
            )
        ]

        world_building = {
            "setting": f"Un univers immersif au cœur des coulisses de {niche_label}.",
            "visual_theme": "Ambiance cinématographique, tons sombres contrastés avec éclairages néon bleus et ambrés.",
            "mood": "Mystérieux, haletant et instructif."
        }

        episodes_roadmap: List[EpisodeOutline] = []
        for i in range(1, total_episodes + 1):
            episodes_roadmap.append(
                EpisodeOutline(
                    episode_number=i,
                    title=f"Épisode {i} : Les Origines de {topic[:20]}" if i == 1 else f"Épisode {i} : Le Secret N°{i}",
                    synopsis=f"Dans cet épisode {i}, la quête s'intensifie autour de {topic}.",
                    key_events=[f"Révélation majeure N°{i}", f"Confrontation clé de l'épisode {i}"],
                    cliffhanger=f"Comment faire face au rebondissement final de l'épisode {i} ?"
                )
            )

        return SeriesConcept(
            series_id=series_id,
            title=title,
            logline=logline,
            serie=niche_label,
            brand_id=brand_id,
            market=market,
            total_episodes=total_episodes,
            season_number=season_number,
            is_multi_season=season_number > 1,  # True uniquement si saison > 1 (multi-saison explicite)
            main_story_arc=main_story_arc,
            character_bible=character_bible,
            world_building=world_building,
            episodes_roadmap=episodes_roadmap,
            current_episode=1,
            status="active"
        )

    def _pitch_with_llm(
        self,
        opportunity: Any,
        brand_profile: Any,
        niche_name: str,
        total_episodes: int,
        season_number: int
    ) -> Optional[SeriesConcept]:
        """Utilise le LLM Provider pour concevoir une série hautement personnalisée."""

        from src.llm import LLMMessage

        niche_label = niche_name
        topic = getattr(opportunity, "topic", "Intelligence artificielle & Futur")
        brand_id = getattr(brand_profile, "id", "brand_default")
        brand_name = getattr(brand_profile, "name", brand_id)
        market = getattr(brand_profile, "market", "FR")

        system_prompt = """Tu es un Showrunner et Directeur de Création Senior pour YouTube.
Ton objectif est d'inventer une série vidéo feuilletonnante de haute qualité pour une chaîne YouTube.
Tu dois répondre STRICTEMENT en JSON respectant exactement la structure suivante :

{
  "title": "Titre captivant de la série",
  "logline": "Pitch en 2 phrases qui donne envie de tout regarder",
  "main_story_arc": {
    "intro": "Description du début de saison",
    "development": "Description du milieu de saison",
    "climax": "Description de la fin dramatique",
    "resolution": "Conclusion et ouverture"
  },
  "character_bible": [
    {
      "name": "Nom du personnage",
      "role": "Protagoniste / Rival / Mentor",
      "visual_description": "Description physique détaillée en français",
      "personality": "Traits de caractère",
      "voice_tone": "Ton de voix",
      "permanent_visual_prompt": "Description visuelle ultra précise en ANGLAIS pour Midjourney/Flux"
    }
  ],
  "world_building": {
    "setting": "Décor et univers",
    "visual_theme": "Style visuel et palette de couleurs",
    "mood": "Ambiance générale"
  },
  "episodes_roadmap": [
    {
      "episode_number": 1,
      "title": "Titre de l'épisode 1",
      "synopsis": "Résumé de l'intrigue de cet épisode",
      "key_events": ["Événement 1", "Événement 2"],
      "cliffhanger": "Accroche suspense pour l'épisode suivant"
    }
  ]
}
"""

        user_prompt = f"""Conçois une série complète de {total_episodes} épisodes (Saison {season_number}) pour la chaîne '{brand_name}' ({market}).
Niche source : {niche_label}
Sujet d'opportunité source : {topic}
Nombre d'épisodes requis dans la roadmap : EXACTEMENT {total_episodes}.
Génère entre 2 et 3 personnages récurrents forts dans la character_bible.
"""

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt)
        ]

        response = self._llm.generate(messages, json_mode=True, max_tokens=3500)
        content = response.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)

        series_id = f"series_{brand_id}_{niche_label.lower()}_s{season_number}_{int(datetime.now(timezone.utc).timestamp())}"

        # Filtrage défensif : on ignore les clés inconnues que le LLM pourrait ajouter
        _char_fields = {f.name for f in dataclasses.fields(CharacterSpec)}
        _ep_fields = {f.name for f in dataclasses.fields(EpisodeOutline)}

        char_bible = [
            CharacterSpec(**{k: v for k, v in c.items() if k in _char_fields})
            for c in parsed.get("character_bible", [])
        ]
        roadmap = [
            EpisodeOutline(**{k: v for k, v in e.items() if k in _ep_fields})
            for e in parsed.get("episodes_roadmap", [])
        ]
        # Renumérotation défensive 1..N dans l'ordre de retour du LLM — évite
        # toute désynchronisation entre episode_number (potentiellement
        # dupliqué/mal ordonné par le LLM) et la position réelle dans la
        # roadmap, sur laquelle s'appuie mark_episode_produced().
        for idx, ep in enumerate(roadmap, start=1):
            ep.episode_number = idx

        effective_total = len(roadmap) if roadmap else total_episodes
        if roadmap and effective_total != total_episodes:
            logger.warning(
                "Roadmap LLM : %d épisode(s) reçu(s) au lieu des %d demandés — total_episodes ajusté à %d.",
                effective_total, total_episodes, effective_total,
            )

        return SeriesConcept(
            series_id=series_id,
            title=parsed.get("title", f"Série {niche_label}"),
            logline=parsed.get("logline", "Série fascinante en plusieurs épisodes."),
            serie=niche_label,
            brand_id=brand_id,
            market=market,
            total_episodes=effective_total,
            season_number=season_number,
            is_multi_season=season_number > 1,  # True uniquement si saison > 1 (multi-saison explicite)
            main_story_arc=parsed.get("main_story_arc", {}),
            character_bible=char_bible,
            world_building=parsed.get("world_building", {}),
            episodes_roadmap=roadmap,
            current_episode=1,
            status="active"
        )

    def get_next_episode_context(self, series: SeriesConcept) -> Dict[str, Any]:
        """
        Construit le contexte complet nécessaire pour produire le script et les visuels
        de l'épisode courant d'une série.
        """
        current_ep_number = series.current_episode
        outline = series.current_episode_outline

        produced_episodes = [ep for ep in series.episodes_roadmap if ep.status == "produced"]
        previous_recaps = []
        for ep in produced_episodes:
            summary = ep.summary_produced or ep.synopsis
            previous_recaps.append(f"Épisode {ep.episode_number} ({ep.title}): {summary}")
        
        previous_episodes_recap = "\n".join(previous_recaps) if previous_recaps else "Aucun épisode précédent (début de la série)."

        return {
            "series_id": series.series_id,
            "series_title": series.title,
            "series_logline": series.logline,
            "season_number": series.season_number,
            "total_episodes": series.total_episodes,
            "current_episode": current_ep_number,
            "remaining_episodes": series.remaining_episodes_count,
            "is_season_finale": (current_ep_number == series.total_episodes),
            "episode_outline": dataclasses.asdict(outline) if outline else {},
            "previous_episodes_recap": previous_episodes_recap,
            "character_bible": [dataclasses.asdict(c) for c in series.character_bible],
            "world_building": series.world_building,
            "main_story_arc": series.main_story_arc
        }

    def mark_episode_produced(
        self,
        series: SeriesConcept,
        episode_number: int,
        script_summary: str
    ) -> SeriesConcept:
        """
        Marque un épisode comme produit, enregistre son résumé, fait avancer le compteur d'épisodes
        et passe la série en 'completed' si le dernier épisode a été produit.
        """
        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        found = False
        for ep in series.episodes_roadmap:
            if ep.episode_number == episode_number:
                ep.status = "produced"
                ep.summary_produced = script_summary
                ep.produced_date = today_iso
                found = True
                break

        if not found:
            logger.warning("Épisode %d non trouvé dans la roadmap de la série %s.", episode_number, series.series_id)

        if episode_number >= series.total_episodes:
            series.status = "completed"
            series.current_episode = series.total_episodes
            logger.info("Série '%s' (%s) marquée COMPLETE (tous les %d épisodes ont été produits).", series.title, series.series_id, series.total_episodes)
        else:
            series.current_episode = episode_number + 1

        series.updated_at = datetime.now(timezone.utc).isoformat()
        return series
