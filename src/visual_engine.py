"""
Visual Engine v1 — Plan visuel complet à partir d'un Script.

Transforme un Script (scènes textuelles + prompts) en un VisualPlan structuré,
prêt à être consommé par les futurs Image Engine, Animation Engine, Video Engine.

Architecture à responsabilité unique :
  - Script Engine    → produit le Script (contrat officiel)
  - Visual Engine    → produit le VisualPlan (plan visuel complet)
  - Image Engine     → futur : génère les images à partir du VisualPlan
  - Animation Engine → futur : anime les scènes
  - Video Engine     → futur : assemble la vidéo finale

Découplage strict :
  - Le Visual Engine ne connaît QUE Script et ScriptScene.
  - Il n'importe AUCUN autre module du projet.
  - Aucune dépendance vers Opportunity, Brand, Creative, Knowledge, Collector.

Contrat :
  - Entrée : Script (scènes avec visual_description, image_prompt, animation_notes)
  - Sortie  : VisualPlan (plan visuel complet prêt pour production)

Extensibilité (Sprint 19+) :
  VisualGenerator
        │
        ├── HeuristicVisualGenerator  (V1 — règles heuristiques)
        ├── LLMVisualGenerator         (Sprint 19)
        └── AIVisualGenerator          (Sprint 20+)

  VisualEngine orchestre sans connaître le générateur réel.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.script_engine import Script, ScriptScene

logger = logging.getLogger(__name__)

# ── Constantes visuelles ────────────────────────────────────────────────────

# Types de plans possibles (shot types)
_SHOT_TYPES = [
    "close_up",
    "medium",
    "wide",
    "extreme_close_up",
    "medium_close_up",
    "cowboy",
    "two_shot",
    "over_the_shoulder",
    "establishing",
    "aerial",
    "macro",
    "pov",
]

# Mouvements de caméra possibles
_CAMERA_MOTIONS = [
    "static",
    "pan_left",
    "pan_right",
    "tilt_up",
    "tilt_down",
    "dolly_in",
    "dolly_out",
    "tracking_left",
    "tracking_right",
    "crane_up",
    "crane_down",
    "handheld",
    "steadycam",
    "zoom_in",
    "zoom_out",
    "raking",
]

# Motifs visuels → types de plans
_SECTION_SHOT_MAP: Dict[str, str] = {
    "Hook":            "medium_close_up",
    "Introduction":    "medium",
    "Contexte":        "wide",
    "Développement":   "medium",
    "Problème/Défi":   "close_up",
    "Tentative":       "wide",
    "Tentative #1":    "wide",
    "Tentative #2":    "medium",
    "Tentative #3":    "close_up",
    "Rebondissement":  "extreme_close_up",
    "Point #1":        "medium_close_up",
    "Point #2":        "medium_close_up",
    "Point #3":        "close_up",
    "Point bonus":     "medium",
    "Erreur #1":       "close_up",
    "Erreur #2":       "medium",
    "Erreur #3":       "extreme_close_up",
    "La bonne approche": "medium",
    "Critère #1":      "medium",
    "Critère #2":      "medium_close_up",
    "Critère #3":      "close_up",
    "Verdict":         "medium",
    "Résolution":      "wide",
    "Résultat":        "close_up",
    "Leçon":           "medium",
    "Conclusion":      "medium",
    "CTA":             "close_up",
}

# Motifs visuels → mouvements caméra
_SECTION_MOTION_MAP: Dict[str, str] = {
    "Hook":            "dolly_in",
    "Introduction":    "static",
    "Contexte":        "pan_left",
    "Développement":   "handheld",
    "Problème/Défi":   "dolly_in",
    "Tentative":       "tracking_right",
    "Tentative #1":    "tracking_right",
    "Tentative #2":    "handheld",
    "Tentative #3":    "dolly_in",
    "Rebondissement":  "zoom_in",
    "Point #1":        "static",
    "Point #2":        "pan_right",
    "Point #3":        "dolly_in",
    "Point bonus":     "pan_left",
    "Erreur #1":       "zoom_in",
    "Erreur #2":       "handheld",
    "Erreur #3":       "dolly_in",
    "La bonne approche": "steadycam",
    "Critère #1":      "pan_right",
    "Critère #2":      "static",
    "Critère #3":      "dolly_in",
    "Verdict":         "static",
    "Résolution":      "dolly_out",
    "Résultat":        "dolly_in",
    "Leçon":           "crane_down",
    "Conclusion":      "dolly_out",
    "CTA":             "zoom_out",
}

# Motifs visuels → composition
_SECTION_COMPOSITION_MAP: Dict[str, str] = {
    "Hook":            "Centrage dynamique avec typographie imposante. Sujet au centre, espace négatif maîtrisé.",
    "Introduction":    "Règle des tiers. Sujet décalé à droite, espace à gauche pour overlay texte ou B-roll.",
    "Contexte":        "Plan large centré. Horizon au tiers supérieur. Profondeur de champ pour capter l'environnement.",
    "Développement":   "Règle des tiers. Face caméra ou légèrement décalé. Fond professionnel flouté.",
    "Problème/Défi":   "Gros plan resserré. Sujet centré. Arrière-plan sombre. Tension visuelle par symétrie.",
    "Tentative":       "Plan large. Sujet en mouvement dans le cadre. Ligne directrice qui guide le regard.",
    "Tentative #1":    "Plan large légèrement en contre-plongée. Sujet actif. Profondeur de champ.",
    "Tentative #2":    "Plan moyen. Sujet décalé à gauche. Chronomètre visible à droite. Action visible.",
    "Tentative #3":    "Gros plan. Sujet centré. Hors-choc suggestif. Tension par cadrage serré.",
    "Rebondissement":  "Extrême gros plan. Micro-détail. Arrière-plan totalement noir. Isolement total.",
    "Point #1":        "Cadrage typographique. Infographie centrée. Numéro large à gauche. Texte à droite.",
    "Point #2":        "Même structure que Point #1. Variation de couleur dominante.",
    "Point #3":        "Infographie plein écran. Donnée choc centrée. Échelle visuelle imposante.",
    "Point bonus":     "Cadrage aéré. Espace négatif généreux. Ambiance plus légère visuellement.",
    "Erreur #1":       "Split-screen vertical. Erreur à gauche (rouge), correction à droite (vert). Symétrie.",
    "Erreur #2":       "Cadrage liste. Éléments empilés verticalement. Aspect 'checklist' visuel.",
    "Erreur #3":       "Zoom progressif sur détail. Composition en entonnoir vers le point critique.",
    "La bonne approche": "Plan de travail. Surface plane cadrée du dessus (top-down). Outils disposés.",
    "Critère #1":      "Tableau comparatif. Deux colonnes A/B. En-têtes contrastés. Défilement vertical.",
    "Critère #2":      "Même structure que Critère #1. Nouvelles lignes ajoutées. Barres de score animées.",
    "Critère #3":      "Point culminant du tableau. Colonne gagnante mise en avant. Composition asymétrique.",
    "Verdict":         "Plein cadre. Logo ou nom du vainqueur centré. Effet de confirmation visuelle.",
    "Résolution":      "Plan large apaisant. Retour au calme. Horizon visible. Espace et lumière.",
    "Résultat":        "Donnée finale centrée. Plein écran typographique. Pause dans le mouvement général.",
    "Leçon":           "Face caméra centré. Fond neutre chaleureux. Lumière douce en clair-obscur.",
    "Conclusion":      "Résumé en grille 2×2 ou 3×1. Éléments ordonnés. Hiérarchie claire.",
    "CTA":             "Bouton d'abonnement centré. Logo de chaîne en haut à gauche. Liens organisés.",
}

# Palettes de couleurs par style / ambiance
_COLOR_PALETTES: Dict[str, List[str]] = {
    "default":            ["#1A1A2E", "#16213E", "#0F3460", "#E94560", "#FFFFFF"],
    "professionnel":      ["#2C3E50", "#34495E", "#95A5A6", "#ECF0F1", "#3498DB"],
    "innovant":           ["#0D0D0D", "#00D4FF", "#7B2FBE", "#FFFFFF", "#1A1A2E"],
    "ludique":            ["#FF6B6B", "#4ECDC4", "#FFE66D", "#1A1A2E", "#FFFFFF"],
    "sombre":             ["#0A0A0A", "#1A1A2E", "#2D2D44", "#E94560", "#AAAAAA"],
    "clair":              ["#F5F5F5", "#E0E0E0", "#9E9E9E", "#2196F3", "#FFFFFF"],
    "chaleureux":         ["#2D1B00", "#5C3A21", "#8B5E3C", "#D4A76A", "#FFE4B5"],
    "froid":              ["#0D1B2A", "#1B2838", "#415A77", "#778DA9", "#E0E1DD"],
    "nature":             ["#2D5016", "#4A7C3F", "#8CB26B", "#D4C9A8", "#F5F0E1"],
    "technologique":      ["#0F0F0F", "#00FF41", "#003B00", "#1A1A1A", "#CCFFCC"],
    "minimaliste":        ["#FFFFFF", "#F0F0F0", "#D0D0D0", "#333333", "#000000"],
    "vibrant":            ["#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF"],
}

# Lumières par type de plan
_LIGHTING_MAP: Dict[str, str] = {
    "close_up":          "Éclairage trois points. Key à 45°, fill à 60%, rim light latérale. Contraste modéré.",
    "medium":            "Lumière naturelle diffusée. Key à 30°, fill à 90%. Ambiance soft.",
    "wide":              "Lumière ambiante naturelle. Source principale large. Ombres douces.",
    "extreme_close_up":  "Éclairage dur. Key unique à 0°, ombres marquées. Rim light agressive.",
    "medium_close_up":   "Rembrandt. Key à 45°, triangle de lumière sur la joue. Fill minimal.",
    "cowboy":            "Lumière latérale dramatique. Contraste élevé. Ombres cinématographiques.",
    "two_shot":          "Double key à 45° de chaque côté. Fill central doux. Éclairage équilibré.",
    "over_the_shoulder": "Key côté épaule visible. Ombres portées au premier plan. Profondeur.",
    "establishing":      "Lumière naturelle d'ambiance. Heure dorée ou ciel couvert. Palette naturelle.",
    "aerial":            "Lumière zénithale diffuse. Ombres courtes. Contraste modéré.",
    "macro":             "Éclairage annulaire (ring light). Diffusion uniforme. Pas d'ombre portée.",
    "pov":               "Lumière directionnelle subjective. Source visible dans le cadre. Réalisme brut.",
}

# Transitions entre scènes
_TRANSITIONS = [
    "cut",
    "crossfade",
    "fade_to_black",
    "fade_from_black",
    "dissolve",
    "wipe_left",
    "wipe_right",
    "slide_up",
    "slide_down",
    "push_left",
    "push_right",
    "zoom_in",
    "zoom_out",
    "iris_in",
    "iris_out",
    "page_curl",
    "clock_wipe",
    "glitch_effect",
    "luma_fade",
    "radial_wipe",
]

_TRANSITION_MAP: Dict[str, str] = {
    "Hook":            "fade_from_black",
    "Introduction":    "crossfade",
    "Contexte":        "dissolve",
    "Développement":   "cut",
    "Problème/Défi":   "fade_to_black",
    "Tentative":       "cut",
    "Tentative #1":    "wipe_right",
    "Tentative #2":    "wipe_right",
    "Tentative #3":    "zoom_in",
    "Rebondissement":  "glitch_effect",
    "Point #1":        "slide_up",
    "Point #2":        "slide_up",
    "Point #3":        "zoom_in",
    "Point bonus":     "dissolve",
    "Erreur #1":       "wipe_left",
    "Erreur #2":       "wipe_left",
    "Erreur #3":       "zoom_in",
    "La bonne approche": "dissolve",
    "Critère #1":      "slide_up",
    "Critère #2":      "slide_up",
    "Critère #3":      "push_left",
    "Verdict":         "crossfade",
    "Résolution":      "fade_to_black",
    "Résultat":        "zoom_in",
    "Leçon":           "dissolve",
    "Conclusion":      "crossfade",
    "CTA":             "fade_to_black",
}

# Overlay text patterns par type de section
_OVERLAY_MAP: Dict[str, str] = {
    "Hook":            "{hook_text}",
    "Introduction":    "{title}",
    "Contexte":        "Le contexte",
    "Développement":   "{topic} expliqué",
    "Problème/Défi":   "Le vrai problème",
    "Tentative":       "On teste",
    "Tentative #1":    "Étape 1",
    "Tentative #2":    "Étape 2",
    "Tentative #3":    "Étape 3 — Le moment clé",
    "Rebondissement":  "MAIS...",
    "Point #1":        "1",
    "Point #2":        "2",
    "Point #3":        "3",
    "Point bonus":     "BONUS",
    "Erreur #1":       "ERREUR #1",
    "Erreur #2":       "ERREUR #2",
    "Erreur #3":       "ERREUR #3",
    "La bonne approche": "La solution",
    "Critère #1":      "Critère 1",
    "Critère #2":      "Critère 2",
    "Critère #3":      "Critère 3",
    "Verdict":         "VERDICT",
    "Résolution":      "En résumé",
    "Résultat":        "LE RÉSULTAT",
    "Leçon":           "À retenir",
    "Conclusion":      "Conclusion",
    "CTA":             "Abonne-toi !",
}


# ── VisualScene ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VisualScene:
    """
    Plan visuel complet pour une scène.

    Chaque VisualScene décrit précisément ce qui doit être VISUEL
    (pas audio, pas narration — ça appartient au Script).

    L'Image Engine lira : shot_type, visual_prompt, composition, lighting, color_palette
    L'Animation Engine lira : camera_motion, animation_notes, transition
    Le Video Engine lira : duration_seconds, overlay_text, transition
    Le Graphic Engine lira : overlay_text, color_palette

    Champs :
      scene_order       : index de la scène (1-based, correspond à ScriptScene.order)
      shot_type         : type de plan (close_up, medium, wide...)
      camera_motion     : mouvement de caméra (static, pan, dolly...)
      visual_prompt     : prompt détaillé pour générateur d'image (DALL-E, Midjourney, Stable Diffusion)
      composition       : description de la composition du cadre
      lighting          : schéma d'éclairage
      color_palette     : palette de couleurs dominantes (liste de hex)
      transition        : transition depuis la scène précédente
      overlay_text      : texte affiché en superposition
      animation_notes   : instructions d'animation pour cette scène spécifique
      duration_seconds  : durée en secondes (copiée depuis ScriptScene)
      metadata          : données extensibles
    """

    scene_order: int
    shot_type: str
    camera_motion: str
    visual_prompt: str
    composition: str
    lighting: str
    color_palette: List[str]
    transition: str
    overlay_text: str
    animation_notes: str
    duration_seconds: int
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── VisualPlan ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VisualPlan:
    """
    Plan visuel complet pour une vidéo entière.

    C'est le contrat officiel entre le Visual Engine et tous les
    moteurs en aval (Image Engine, Animation Engine, Video Engine).

    Contient :
      title         : titre de la vidéo (hérité du Script)
      style         : style visuel global (hérité du Script.style)
      aspect_ratio  : format d'image (9:16 par défaut — Shorts, Stories, Reels)
      scenes        : liste ordonnée de VisualScene
      color_palette : palette de couleurs globale de la vidéo
      metadata      : données extensibles

    Pour les prochains moteurs :
      Image Engine     → VisualPlan.scenes[].visual_prompt + color_palette + composition
      Animation Engine → VisualPlan.scenes[].animation_notes (= transition de la scène) + camera_motion
      Video Engine     → VisualPlan.scenes (montage + timing)
    """

    title: str
    style: str
    aspect_ratio: str = "9:16"
    scenes: List[VisualScene] = field(default_factory=list)
    color_palette: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── VisualGenerator ──────────────────────────────────────────────────────────

class VisualGenerator(ABC):
    """
    Interface abstraite pour tous les générateurs de VisualPlan.

    Pour intégrer un LLM (Sprint 19) :
      1. Sous-classer VisualGenerator
      2. Implémenter name et generate()
      3. Injecter dans VisualEngine(generator=MonGenerateur())

    Le système ne change pas — respect du principe ouvert/fermé.

    Le générateur ne reçoit qu'un Script — rien d'autre.
    Aucune dépendance vers Brand, Creative, Opportunity, Knowledge, Collector.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, script: Script) -> VisualPlan:
        """
        Transforme un Script complet en VisualPlan.

        Args:
            script: Script vidéo complet (scènes, narration, prompts visuels).

        Returns:
            VisualPlan complet prêt pour les moteurs en aval.
        """
        ...


# ── HeuristicVisualGenerator ────────────────────────────────────────────────

class HeuristicVisualGenerator(VisualGenerator):
    """
    Générateur heuristique de plans visuels — aucun appel LLM.

    Transforme chaque scène d'un Script en VisualScene en appliquant
    des règles heuristiques basées sur le titre de la scène et son contenu :

    Règles appliquées pour chaque scène :
      1. Type de plan  → mapping section → _SECTION_SHOT_MAP
      2. Mouvement     → mapping section → _SECTION_MOTION_MAP
      3. Composition   → mapping section → _SECTION_COMPOSITION_MAP
      4. Lumière       → mapping shot_type → _LIGHTING_MAP
      5. Transition    → mapping section → _TRANSITION_MAP
      6. Overlay text  → mapping section → _OVERLAY_MAP (paramétré)
      7. Palette       → mapping style → _COLOR_PALETTES
      8. Visual prompt → enrichi depuis ScriptScene.image_prompt
      9. Animation     → enrichi depuis ScriptScene.transition (Sprint 31.1)

    Aucune IA — uniquement des règles déterministes.
    """

    @property
    def name(self) -> str:
        return "heuristic_visual_v1"

    def generate(self, script: Script) -> VisualPlan:
        """
        Génère un VisualPlan complet à partir d'un Script.

        Pipeline :
          1. Déterminer le style visuel global
          2. Choisir la palette de couleurs globale
          3. Pour chaque scène du Script → construire une VisualScene
          4. Assembler le VisualPlan

        Args:
            script: Script vidéo complet.

        Returns:
            VisualPlan prêt pour les moteurs en aval.
        """
        style = script.style or "default"
        global_palette = _COLOR_PALETTES.get(style.lower(), _COLOR_PALETTES["default"])

        visual_scenes: List[VisualScene] = []
        for scene in script.scenes:
            visual_scenes.append(self._build_visual_scene(scene, script, style))

        # Durée totale estimée
        total_duration = sum(s.duration_seconds for s in visual_scenes)

        plan = VisualPlan(
            title=script.title,
            style=style,
            aspect_ratio="9:16",
            scenes=visual_scenes,
            color_palette=list(global_palette),
            metadata={
                "generator": self.name,
                "scene_count": len(visual_scenes),
                "total_duration_seconds": total_duration,
                "script_style": script.style,
                "script_language": script.language,
                "script_title": script.title,
            },
        )

        logger.info(
            "VisualPlan '%s' généré : %d scènes visuelles, %d s (générateur: %s)",
            script.title[:50],
            len(visual_scenes),
            total_duration,
            self.name,
        )
        return plan

    # ── Construction d'une VisualScene ────────────────────────────────────────

    def _build_visual_scene(
        self,
        scene: ScriptScene,
        script: Script,
        style: str,
    ) -> VisualScene:
        """
        Construit une VisualScene à partir d'une ScriptScene.

        Sprint 31.1 : les scènes n'ont plus de titre nommé (Hook/Point #1/...)
        — le mapping heuristique se base désormais sur la POSITION de la
        scène (première = hook, dernière = CTA, rotation pour les scènes
        intermédiaires) plutôt que sur un texte de titre. Ce plan heuristique
        reste une base : VisualDirector (LLM) le redirige ensuite scène par
        scène pour la production réelle (shot_type/composition/lighting/
        color_palette) — voir scripts/run_daily_pipeline.py.

        Applique les règles heuristiques :
          - Type de plan basé sur la position de la scène
          - Mouvement basé sur la position
          - Composition basée sur la position
          - Lumière basée sur le type de plan
          - Transition basée sur la position
          - Overlay basé sur la position (paramétré avec les données du script)
          - Palette basée sur le style global
          - Prompt visuel basé sur la description riche de la scène
        """
        position_key = self._resolve_position_key(scene.order, len(script.scenes))

        shot_type = _SECTION_SHOT_MAP.get(position_key, "medium")
        camera_motion = _SECTION_MOTION_MAP.get(position_key, "static")
        composition = _SECTION_COMPOSITION_MAP.get(
            position_key,
            "Composition standard. Règle des tiers. Fond neutre.",
        )
        lighting = _LIGHTING_MAP.get(shot_type, _LIGHTING_MAP["medium"])
        transition = _TRANSITION_MAP.get(position_key, "cut")
        palette = _COLOR_PALETTES.get(style.lower(), _COLOR_PALETTES["default"])

        # Overlay text paramétré
        overlay_template = _OVERLAY_MAP.get(position_key, "")
        overlay_text = self._render_overlay(overlay_template, scene, script)

        # Prompt visuel enrichi
        visual_prompt = self._build_visual_prompt(scene, shot_type, camera_motion, style, palette)

        scene_metadata = {
            "script_scene_order": scene.order,
            "position_key_resolved": position_key,
            "style_resolved": style,
            "palette_source": style.lower() if style.lower() in _COLOR_PALETTES else "default",
        }

        visual_scene = VisualScene(
            scene_order=scene.order,
            shot_type=shot_type,
            camera_motion=camera_motion,
            visual_prompt=visual_prompt,
            composition=composition,
            lighting=lighting,
            color_palette=list(palette),
            transition=transition,
            overlay_text=overlay_text,
            animation_notes=scene.transition,
            duration_seconds=scene.duration_seconds,
            metadata=scene_metadata,
        )

        return visual_scene

    # ── Méthodes auxiliaires ──────────────────────────────────────────────────

    # Rotation de clés génériques pour les scènes intermédiaires — assure une
    # variété de plans/mouvements/compositions sans dépendre d'un titre nommé.
    _MIDDLE_ROTATION = [
        "Contexte", "Développement", "Point #1", "Point #2", "Point #3", "Rebondissement",
    ]

    @classmethod
    def _resolve_position_key(cls, order: int, total: int) -> str:
        """
        Résout la position d'une scène (1-based) en clé de mapping.

        La première scène joue le rôle du hook, la dernière celui du CTA —
        les scènes intermédiaires tournent sur un petit jeu de clés
        génériques pour varier plan/mouvement/composition (Sprint 31.1).
        """
        if order <= 1:
            return "Hook"
        if order >= total:
            return "CTA"
        idx = (order - 2) % len(cls._MIDDLE_ROTATION)
        return cls._MIDDLE_ROTATION[idx]

    @staticmethod
    def _render_overlay(template: str, scene: ScriptScene, script: Script) -> str:
        """
        Remplit le template d'overlay avec les données réelles.

        Variables disponibles :
          {hook_text}    → script.hook (tronqué)
          {title}        → script.title
          {topic}        → dérivé du script.title ou metadata.niche
          {scene_title}  → début de la description riche de la scène
        """
        topic = (
            script.metadata.get("niche", "")
            or script.metadata.get("opportunity_id", "")
            or script.title
        )
        hook_short = script.hook[:60] if len(script.hook) > 60 else script.hook

        text = template.format(
            hook_text=hook_short,
            title=script.title[:50],
            topic=topic[:40],
            scene_title=scene.scene.description.setting[:40],
        )
        return text

    @staticmethod
    def _build_visual_prompt(
        scene: ScriptScene,
        shot_type: str,
        camera_motion: str,
        style: str,
        palette: List[str],
    ) -> str:
        """
        Construit un prompt visuel enrichi pour générateur d'image.

        Combine :
          1. La description riche de la scène (ScriptScene.scene.description.setting/composition)
          2. Le type de plan
          3. Le mouvement de caméra
          4. Le style
          5. La palette de couleurs
          6. La durée

        Format : "[description scène] -- [type plan] -- [mouvement] -- style: [style] -- palette: [couleurs]"
        """
        desc = scene.scene.description
        base_prompt = f"{desc.setting.strip()} {desc.composition.strip()}".strip().rstrip(".")
        colors_str = ", ".join(palette[:3])

        enhanced = (
            f"{base_prompt}. "
            f"Shot type: {shot_type.replace('_', ' ')}, "
            f"camera motion: {camera_motion.replace('_', ' ')}, "
            f"style: {style}, "
            f"color palette: {colors_str}, "
            f"duration: {scene.duration_seconds}s"
        )
        return enhanced


# ── VisualEngine ─────────────────────────────────────────────────────────────

class VisualEngine:
    """
    Orchestrateur du Visual Engine.

    Transforme un ou plusieurs Scripts en VisualPlans complets,
    prêts à être consommés par les futurs Image Engine, Animation Engine,
    et Video Engine.

    Exemple minimal (HeuristicVisualGenerator automatique) :
        engine = VisualEngine()
        plan = engine.generate(script)

    Avec un générateur LLM (Sprint 19) :
        engine = VisualEngine(generator=LLMVisualGenerator())
        plan = engine.generate(script)

    Génération multiple :
        plans = engine.generate_all([script1, script2])

    Le moteur ne connaît aucun autre moteur du système.
    Il ne manipule que : Script → VisualPlan.
    """

    def __init__(self, generator: Optional[VisualGenerator] = None) -> None:
        self._generator = generator or HeuristicVisualGenerator()

    @property
    def generator_name(self) -> str:
        return self._generator.name

    # ── Interface publique ─────────────────────────────────────────────────────

    def generate(self, script: Script) -> VisualPlan:
        """
        Génère un VisualPlan pour un Script.

        Args:
            script: Script vidéo complet (scènes avec visuels).

        Returns:
            VisualPlan complet, prêt pour production.

        Raises:
            ValueError: si le script n'a pas de scènes.
        """
        if not script.scenes:
            raise ValueError(
                f"Impossible de générer un VisualPlan : "
                f"le script '{script.title[:50]}' n'a aucune scène."
            )

        return self._generator.generate(script)

    def generate_all(self, scripts: List[Script]) -> List[VisualPlan]:
        """
        Génère des VisualPlans pour une liste de Scripts.

        Args:
            scripts: Liste de Scripts vidéo.

        Returns:
            Liste de VisualPlans (même ordre que les scripts).
        """
        plans: List[VisualPlan] = []
        errors = 0

        for script in scripts:
            try:
                plans.append(self.generate(script))
            except Exception as exc:
                logger.warning(
                    "Échec VisualPlan pour '%s' : %s",
                    script.title[:50], exc,
                )
                errors += 1

        if errors:
            logger.warning(
                "%d VisualPlan(s) sur %d échoué(s).",
                errors, len(scripts),
            )
        else:
            logger.info(
                "%d VisualPlan(s) généré(s) (générateur: %s).",
                len(plans), self.generator_name,
            )

        return plans
