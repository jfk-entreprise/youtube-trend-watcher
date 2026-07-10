"""
Brand Engine v1 — Mémoire éditoriale des chaînes YouTube.

Représente l'identité éditoriale complète d'une chaîne.
Ne génère aucun contenu. Ne dépend d'aucun autre moteur.
Tous les moteurs créatifs en aval dépendront de lui.

Composants :
  - BrandProfile    : contrat officiel (identité de chaîne, immuable).
  - BrandStore      : interface abstraite de persistance.
  - JsonBrandStore  : implémentation JSON sur disque (V1).
  - BrandEngine     : orchestrateur avec API haut niveau.

Extensibilité Sprint 14+ :
  BrandStore
        │
        ├── JsonBrandStore        (V1, fichiers locaux)
        └── SupabaseBrandStore    (Sprint 14+, persistance cloud)

Contrat Sprint 14 (Script Engine) :
    Opportunity + CreativeBrief + BrandProfile → Script
    Le Script Engine ne lit jamais directement le Knowledge Engine.
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_LANGUAGES = {"fr", "en", "es", "pt", "de", "it", "ar", "zh", "ja", "ko"}
_VALID_VOICE_SPEEDS = {"Rapide", "Modéré", "Lent"}


# ── BrandProfile ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BrandProfile:
    """
    Identité éditoriale complète d'une chaîne YouTube.

    Contrat officiel entre le Brand Engine et tous les moteurs créatifs
    (Creative Engine, Script Engine, Visual Engine, Distribution Engine).

    Niveaux [0.0–1.0] :
      emotion_level     : intensité émotionnelle du contenu produit.
      humor_level       : place de l'humour dans la communication.
      authority_level   : posture d'expert vs posture de pair.
      curiosity_level   : approche questionnante vs affirmative.
      storytelling_level: place de la narration dans les vidéos.
    """

    id: str                            # identifiant unique (ex. "business_fr")
    name: str                          # nom affiché de la chaîne
    description: str                   # pitch de la chaîne (1-2 phrases)
    niche: str                         # thématique principale
    target_audience: str               # description précise du public cible
    primary_language: str              # code langue (fr, en, es…)

    tone: str                          # tonalité principale (Professionnel, Innovant…)
    personality: str                   # description longue de la personnalité
    writing_style: str                 # style rédactionnel

    emotion_level: float
    humor_level: float
    authority_level: float
    curiosity_level: float
    storytelling_level: float

    voice_speed: str                   # Rapide | Modéré | Lent
    preferred_video_duration: int      # durée cible en secondes

    preferred_formats: List[str]       # ex. ["Short", "Liste", "Tutorial"]
    preferred_hooks: List[str]         # templates d'accroches
    preferred_cta: List[str]           # variantes de call-to-action
    forbidden_words: List[str]         # mots à ne jamais utiliser

    visual_style: str                  # description du style visuel général
    color_palette: List[str]           # couleurs hex ou descriptives
    typography_style: str              # guidelines typographiques
    logo_description: str              # description du logo/identité visuelle
    thumbnail_style: str               # guide de création des miniatures

    metadata: Dict[str, Any]           # données extensibles (fréquence, piliers…)

    # Mots-clés (minuscules) utilisés pour faire correspondre une Niche
    # détectée (Niche.name) à cette chaîne — Sprint 28 (Studio de production).
    niche_keywords: List[str] = dataclass_field(default_factory=list)


# ── Sérialisation ─────────────────────────────────────────────────────────────

def _profile_from_dict(data: Dict[str, Any]) -> BrandProfile:
    return BrandProfile(
        id=str(data["id"]),
        name=str(data["name"]),
        description=str(data.get("description", "")),
        niche=str(data["niche"]),
        target_audience=str(data["target_audience"]),
        primary_language=str(data.get("primary_language", "fr")),
        tone=str(data["tone"]),
        personality=str(data.get("personality", "")),
        writing_style=str(data.get("writing_style", "")),
        emotion_level=float(data.get("emotion_level", 0.5)),
        humor_level=float(data.get("humor_level", 0.2)),
        authority_level=float(data.get("authority_level", 0.5)),
        curiosity_level=float(data.get("curiosity_level", 0.5)),
        storytelling_level=float(data.get("storytelling_level", 0.5)),
        voice_speed=str(data.get("voice_speed", "Modéré")),
        preferred_video_duration=int(data.get("preferred_video_duration", 600)),
        preferred_formats=list(data.get("preferred_formats", [])),
        preferred_hooks=list(data.get("preferred_hooks", [])),
        preferred_cta=list(data.get("preferred_cta", [])),
        forbidden_words=list(data.get("forbidden_words", [])),
        visual_style=str(data.get("visual_style", "")),
        color_palette=list(data.get("color_palette", [])),
        typography_style=str(data.get("typography_style", "")),
        logo_description=str(data.get("logo_description", "")),
        thumbnail_style=str(data.get("thumbnail_style", "")),
        metadata=dict(data.get("metadata", {})),
        niche_keywords=list(data.get("niche_keywords", [])),
    )


def _profile_to_dict(profile: BrandProfile) -> Dict[str, Any]:
    return {
        "id": profile.id,
        "name": profile.name,
        "description": profile.description,
        "niche": profile.niche,
        "target_audience": profile.target_audience,
        "primary_language": profile.primary_language,
        "tone": profile.tone,
        "personality": profile.personality,
        "writing_style": profile.writing_style,
        "emotion_level": profile.emotion_level,
        "humor_level": profile.humor_level,
        "authority_level": profile.authority_level,
        "curiosity_level": profile.curiosity_level,
        "storytelling_level": profile.storytelling_level,
        "voice_speed": profile.voice_speed,
        "preferred_video_duration": profile.preferred_video_duration,
        "preferred_formats": list(profile.preferred_formats),
        "preferred_hooks": list(profile.preferred_hooks),
        "preferred_cta": list(profile.preferred_cta),
        "forbidden_words": list(profile.forbidden_words),
        "visual_style": profile.visual_style,
        "color_palette": list(profile.color_palette),
        "typography_style": profile.typography_style,
        "logo_description": profile.logo_description,
        "thumbnail_style": profile.thumbnail_style,
        "metadata": dict(profile.metadata),
        "niche_keywords": list(profile.niche_keywords),
    }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_profile(profile: BrandProfile) -> List[str]:
    """
    Valide un BrandProfile. Retourne une liste de messages d'erreur.
    Liste vide = profil valide.
    """
    issues: List[str] = []

    for field in ("id", "name", "niche", "target_audience", "tone"):
        if not getattr(profile, field, "").strip():
            issues.append(f"Champ '{field}' manquant ou vide.")

    for field in ("emotion_level", "humor_level", "authority_level",
                  "curiosity_level", "storytelling_level"):
        val = getattr(profile, field, None)
        if val is None or not (0.0 <= val <= 1.0):
            issues.append(f"'{field}' doit être dans [0.0, 1.0] — valeur : {val}")

    if profile.primary_language not in _VALID_LANGUAGES:
        issues.append(
            f"'primary_language' non reconnu : '{profile.primary_language}'. "
            f"Valeurs acceptées : {sorted(_VALID_LANGUAGES)}"
        )

    if profile.voice_speed not in _VALID_VOICE_SPEEDS:
        issues.append(
            f"'voice_speed' non reconnu : '{profile.voice_speed}'. "
            f"Valeurs acceptées : {sorted(_VALID_VOICE_SPEEDS)}"
        )

    if profile.preferred_video_duration <= 0:
        issues.append("'preferred_video_duration' doit être > 0 secondes.")

    for field in ("preferred_formats", "preferred_hooks", "preferred_cta"):
        if not getattr(profile, field, []):
            issues.append(f"'{field}' ne peut pas être vide.")

    return issues


# ── BrandStore ────────────────────────────────────────────────────────────────

class BrandStore(ABC):
    """
    Interface abstraite de persistance des BrandProfile.

    Implémentations prévues :
      - JsonBrandStore      : fichiers JSON locaux (V1)
      - SupabaseBrandStore  : table Supabase (Sprint 14+)
    """

    @abstractmethod
    def save(self, profile: BrandProfile) -> None:
        """Persiste un profil (création ou mise à jour)."""
        ...

    @abstractmethod
    def load(self, brand_id: str) -> Optional[BrandProfile]:
        """Charge un profil par identifiant. Retourne None si introuvable."""
        ...

    @abstractmethod
    def list(self) -> List[BrandProfile]:
        """Retourne tous les profils disponibles."""
        ...

    @abstractmethod
    def delete(self, brand_id: str) -> bool:
        """Supprime un profil. Retourne True si supprimé, False si inexistant."""
        ...


# ── JsonBrandStore ────────────────────────────────────────────────────────────

class JsonBrandStore(BrandStore):
    """
    Persistance JSON sur disque local.

    Convention : un fichier <brand_id>.json par profil dans le répertoire cible.
    Chargement dynamique — un nouveau fichier est immédiatement disponible.

    Pour migrer vers Supabase : sous-classer BrandStore → SupabaseBrandStore
    et injecter dans BrandEngine(store=SupabaseBrandStore(...)).
    """

    def __init__(self, directory: Any) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, profile: BrandProfile) -> None:
        path = self._dir / f"{profile.id}.json"
        data = _profile_to_dict(profile)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Profil '%s' écrit → %s", profile.id, path)

    def load(self, brand_id: str) -> Optional[BrandProfile]:
        path = self._dir / f"{brand_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return _profile_from_dict(data)
        except Exception as exc:
            logger.error("Erreur lecture '%s' : %s", path, exc)
            return None

    def list(self) -> List[BrandProfile]:
        profiles: List[BrandProfile] = []
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profiles.append(_profile_from_dict(data))
            except Exception as exc:
                logger.warning("Fichier ignoré '%s' : %s", path.name, exc)
        return profiles

    def delete(self, brand_id: str) -> bool:
        path = self._dir / f"{brand_id}.json"
        if path.exists():
            path.unlink()
            logger.debug("Profil '%s' supprimé.", brand_id)
            return True
        return False


# ── BrandEngine ───────────────────────────────────────────────────────────────

class BrandEngine:
    """
    Orchestrateur du Brand Engine.

    Interface haut niveau pour charger, sauvegarder et valider des BrandProfile.

    Exemple minimal (JsonBrandStore automatique) :
        engine = BrandEngine()
        profiles = engine.list()
        profile = engine.load("business_fr")

    Avec un store personnalisé :
        engine = BrandEngine(store=SupabaseBrandStore(url, key))
        engine.save(my_profile)

    Avec un répertoire personnalisé :
        engine = BrandEngine(brands_dir="/path/to/brands")

    Le moteur ne connaît aucun autre moteur du système.
    Toute l'intelligence éditoriale est encapsulée dans le BrandProfile.
    """

    def __init__(
        self,
        store: Optional[BrandStore] = None,
        brands_dir: Optional[Any] = None,
    ) -> None:
        if store is not None:
            self._store = store
        else:
            directory = Path(brands_dir) if brands_dir else Path(__file__).parent.parent / "brands"
            self._store = JsonBrandStore(directory)

    # ── Interface publique ─────────────────────────────────────────────────────

    def load(self, brand_id: str) -> Optional[BrandProfile]:
        """Charge un profil par identifiant."""
        profile = self._store.load(brand_id)
        if profile is None:
            logger.warning("Profil '%s' introuvable.", brand_id)
        return profile

    def list(self) -> List[BrandProfile]:
        """Retourne tous les profils disponibles, triés par id."""
        profiles = self._store.list()
        logger.info("%d profil(s) de marque chargé(s).", len(profiles))
        return profiles

    def save(self, profile: BrandProfile) -> None:
        """Valide puis persiste un profil."""
        issues = validate_profile(profile)
        if issues:
            raise ValueError(
                f"BrandProfile '{profile.id}' invalide :\n"
                + "\n".join(f"  - {i}" for i in issues)
            )
        self._store.save(profile)
        logger.info("Profil '%s' sauvegardé.", profile.id)

    def delete(self, brand_id: str) -> bool:
        """Supprime un profil. Retourne True si suppression effective."""
        result = self._store.delete(brand_id)
        if result:
            logger.info("Profil '%s' supprimé.", brand_id)
        return result

    def validate(self, profile: BrandProfile) -> List[str]:
        """Retourne la liste des problèmes de validation (liste vide = valide)."""
        return validate_profile(profile)
