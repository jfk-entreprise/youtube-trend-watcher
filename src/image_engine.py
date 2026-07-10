"""
Image Engine v1 — Transforme un VisualPlan en GeneratedImage.

À partir d'un VisualPlan (prompt visuel, composition, palette, éclairage...),
produit une GeneratedImage contenant tout ce qu'il faut pour générer une image :
  - prompt final (prêt pour API)
  - negative_prompt (ce qu'il faut éviter)
  - dimensions (largeur, hauteur)
  - aspect ratio (hérité ou dérivé)
  - seed (pour reproductibilité)
  - provider (identité du générateur)

Architecture à responsabilité unique :
  - Visual Engine    → produit le VisualPlan (plan visuel complet)
  - Image Engine     → produit la GeneratedImage (prête pour génération)
  - (futur) FluxProvider     → génère l'image via Flux
  - (futur) SdProvider       → génère l'image via Stable Diffusion
  - (futur) FalProvider      → génère l'image via Fal.ai
  - (futur) OpenAIProvider   → génère l'image via DALL-E
  - (futur) GeminiProvider   → génère l'image via Imagen

Découplage strict :
  - Le Image Engine ne connaît QUE VisualPlan et VisualScene.
  - Il n'importe AUCUN autre module du projet (sauf utilitaires standards).
  - Aucune dépendance vers Script, Brand, Creative, Knowledge, Collector, LLM.

Extensibilité :
  ImageGenerator
        │
        ├── HeuristicImageGenerator  (V1 — règles heuristiques, fallback)
        ├── FluxImageGenerator        (Sprint 20)
        ├── SdImageGenerator          (Sprint 20)
        ├── FalImageGenerator         (Sprint 20)
        ├── OpenAiImageGenerator      (Sprint 20)
        └── GeminiImageGenerator      (Sprint 20)

  ImageEngine orchestre sans connaître le générateur réel.
  Aucun des générateurs futurs ne modifie l'ImageEngine.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.visual_engine import VisualPlan, VisualScene

logger = logging.getLogger(__name__)


# ── Constantes ───────────────────────────────────────────────────────────────

# Ratios d'aspect standard avec leurs dimensions
_ASPECT_RATIOS: Dict[str, tuple[int, int]] = {
    "1:1":      (1024, 1024),
    "4:5":      (896,  1120),
    "3:4":      (896,  1152),
    "2:3":      (768,  1152),
    "9:16":     (720,  1280),   # Shorts, Reels, Stories
    "16:9":     (1280, 720),    # YouTube Standard
    "21:9":     (1400, 600),    # Cinémascope
    "3:2":      (1152, 768),    # Photo standard
    "4:3":      (1152, 864),    # Présentation
}

# Résolutions maximales par aspect ratio (pour les providers qui supportent le full HD)
_ASPECT_RATIOS_FULL: Dict[str, tuple[int, int]] = {
    "1:1":      (1536, 1536),
    "4:5":      (1344, 1680),
    "3:4":      (1344, 1728),
    "2:3":      (1152, 1728),
    "9:16":     (1080, 1920),   # Full HD Portrait
    "16:9":     (1920, 1080),   # Full HD Landscape
    "21:9":     (2100, 900),
    "3:2":      (1728, 1152),
    "4:3":      (1728, 1296),
}

# Styles → mots-clés de style pour le prompt final
_STYLE_KEYWORDS: Dict[str, str] = {
    "default":        "cinematic, professional photography, high detail",
    "professionnel":  "corporate, clean, professional lighting, sharp focus, 8k",
    "innovant":       "futuristic, cyberpunk aesthetic, neon accents, holographic",
    "ludique":        "playful, vibrant colors, cartoon style, energetic composition",
    "sombre":         "dark moody, dramatic shadows, noir aesthetic, atmospheric",
    "clair":          "bright, airy, high key lighting, soft pastels, minimalist",
    "chaleureux":     "warm tones, golden hour, cozy atmosphere, soft lighting",
    "froid":          "cold tones, blue tint, sterile environment, clinical lighting",
    "nature":         "natural lighting, organic textures, earthy tones, biophilic",
    "technologique":  "tech, digital aesthetic, circuit patterns, neon glow, data viz",
    "minimaliste":    "minimalist, clean lines, negative space, simple geometry",
    "vibrant":        "bold colors, high saturation, dynamic contrast, pop art",
}

# Mots-clés négatifs par style (ce qu'il faut éviter)
_NEGATIVE_STYLE_KEYWORDS: Dict[str, str] = {
    "default":        "blurry, low quality, distorted, ugly, deformed, bad anatomy",
    "professionnel":  "casual, messy, informal, noisy, grainy, unprofessional",
    "innovant":       "dated, retro, old fashioned, boring, dull, generic",
    "ludique":        "dark, scary, violent, serious, gloomy, depressing",
    "sombre":         "bright, cheerful, colorful, pastel, overexposed, sunny",
    "clair":          "dark, shadowy, moody, underexposed, heavy shadows, grim",
    "chaleureux":     "cold, sterile, clinical, blue, icy, uninviting",
    "froid":          "warm, cozy, golden, amber, tropical, sunlit",
    "nature":         "artificial, plastic, synthetic, urban, industrial, concrete",
    "technologique":  "organic, natural, hand drawn, vintage, rustic, primitive",
    "minimaliste":    "cluttered, busy, chaotic, ornate, complex, crowded",
    "vibrant":        "muted, dull, gray, washed out, faded, desaturated",
}

# Qualités par défaut
_DEFAULT_QUALITY = "standard"  # ou "hd"
_DEFAULT_STEPS = 20


# ── GeneratedImage ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GeneratedImage:
    """
    Image prête à être générée — contrat officiel entre Image Engine
    et tous les providers d'image (Flux, Stable Diffusion, DALL-E, Imagen...).

    Contient TOUT ce qu'un provider a besoin pour générer une image,
    sans que le provider ait à connaître le VisualPlan.

    Champs :
      scene_order     : index de la scène source (1-based)
      prompt          : prompt final prêt pour API (string complet)
      negative_prompt : ce qu'il faut éviter dans l'image
      width           : largeur en pixels
      height          : hauteur en pixels
      aspect_ratio    : format d'image (ex: "9:16", "16:9")
      seed            : seed pour reproductibilité (déterministe basé sur le prompt)
      quality         : qualité ("standard" ou "hd")
      steps           : nombre d'étapes de génération (pour diffusion models)
      style           : style visuel (hérité)
      color_palette   : palette de couleurs dominantes
      provider        : identité du générateur (ex: "heuristic", "flux", "sd")
      metadata        : données extensibles (debug, contexte)

    Pour les providers futurs :
      FluxImageGenerator      → lit prompt, negative_prompt, width, height, seed
      SdImageGenerator        → lit prompt, negative_prompt, width, height, seed, steps
      OpenAiImageGenerator    → lit prompt, width, height, quality
      GeminiImageGenerator    → lit prompt, aspect_ratio
      FalImageGenerator       → lit prompt, negative_prompt, width, height, seed
    """

    scene_order: int
    prompt: str
    negative_prompt: str
    width: int
    height: int
    aspect_ratio: str
    seed: int
    quality: str = "standard"
    steps: int = 20
    style: str = "default"
    color_palette: List[str] = field(default_factory=list)
    provider: str = "heuristic"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── ImageGenerator ──────────────────────────────────────────────────────────

class ImageGenerator(ABC):
    """
    Interface abstraite pour tous les générateurs d'image.

    Pour intégrer un nouveau provider (Flux, SD, DALL-E, Imagen...) :
      1. Sous-classer ImageGenerator
      2. Implémenter name et generate()
      3. Injecter dans ImageEngine(generator=MonGenerateur())

    Le système ne change pas — principe ouvert/fermé.
    Aucun import de provider n'est nécessaire dans ImageEngine.

    Le générateur ne reçoit qu'une VisualScene.
    Une scène = une image.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate(self, scene: VisualScene, plan: VisualPlan) -> GeneratedImage:
        """
        Transforme une VisualScene (issue d'un VisualPlan) en GeneratedImage.

        Args:
            scene: La scène visuelle à transformer en image.
            plan:  Le plan visuel complet (pour contexte global : style, aspect_ratio, metadata).

        Returns:
            GeneratedImage prête à être envoyée à un provider.
        """
        ...


# ── HeuristicImageGenerator ─────────────────────────────────────────────────

class HeuristicImageGenerator(ImageGenerator):
    """
    Générateur heuristique d'image — aucun appel API, aucune IA.

    Construit une GeneratedImage uniquement à partir des données structurées
    du VisualPlan et du VisualScene.

    Règles :
      1. Prompt final : composition + éclairage + style + palette + format
      2. Negative prompt : basé sur le style (contre-exemples)
      3. Dimensions : basées sur l'aspect ratio du plan
      4. Seed : hash déterministe du prompt (reproductible)
      5. Provider : "heuristic" (fallback)

    Aucune IA — uniquement des règles déterministes.
    """

    @property
    def name(self) -> str:
        return "heuristic_image_v1"

    def generate(self, scene: VisualScene, plan: VisualPlan) -> GeneratedImage:
        """
        Génère une GeneratedImage à partir d'une VisualScene.

        Pipeline :
          1. Résoudre l'aspect ratio (plan → ratio → dimensions)
          2. Construire le prompt final (composition + éclairage + style + palette)
          3. Construire le negative prompt (basé sur le style)
          4. Calculer le seed (hash déterministe du prompt)
          5. Assembler la GeneratedImage

        Args:
            scene: Scène visuelle à transformer.
            plan:  Plan visuel complet (contexte global).

        Returns:
            GeneratedImage prête pour tout provider d'image.
        """
        # 1. Aspect ratio et dimensions
        aspect_ratio = self._resolve_aspect_ratio(plan.aspect_ratio, scene)
        width, height = self._resolve_dimensions(aspect_ratio, scene)

        # 2. Prompt final
        style_key = plan.style.lower() if plan.style.lower() in _STYLE_KEYWORDS else "default"
        style_kw = _STYLE_KEYWORDS.get(style_key, _STYLE_KEYWORDS["default"])
        palette_str = ", ".join(scene.color_palette[:4])
        prompt = self._build_prompt(scene, style_kw, palette_str, aspect_ratio, width, height)

        # 3. Negative prompt
        negative = _NEGATIVE_STYLE_KEYWORDS.get(style_key, _NEGATIVE_STYLE_KEYWORDS["default"])

        # 4. Seed déterministe
        seed = self._compute_seed(prompt, scene.scene_order)

        # Qualité et steps
        quality = self._resolve_quality(plan, scene)
        steps = self._resolve_steps(plan, scene)

        generated = GeneratedImage(
            scene_order=scene.scene_order,
            prompt=prompt,
            negative_prompt=negative,
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            seed=seed,
            quality=quality,
            steps=steps,
            style=plan.style,
            color_palette=list(scene.color_palette),
            provider=self.name,
            metadata={
                "generator": self.name,
                "scene_shot_type": scene.shot_type,
                "scene_camera_motion": scene.camera_motion,
                "scene_transition": scene.transition,
                "style_keyword": style_kw,
                "aspect_ratio": aspect_ratio,
                "width": width,
                "height": height,
                "seed": seed,
                "quality": quality,
                "steps": steps,
                "plan_style": plan.style,
                "plan_aspect_ratio": plan.aspect_ratio,
            },
        )

        logger.debug(
            "GeneratedImage scene #%d : %dx%d, seed=%d, ratio=%s (provider: %s)",
            scene.scene_order, width, height, seed, aspect_ratio, self.name,
        )
        return generated

    # ── Méthodes auxiliaires ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_aspect_ratio(plan_ratio: str, scene: VisualScene) -> str:
        """
        Résout l'aspect ratio final.

        Priorité :
          1. Si le plan a un ratio valide dans _ASPECT_RATIOS → on le garde
          2. Sinon → on le déduit du shot_type (heuristic)
          3. Fallback → "9:16" (Shorts)

        Règles heuristiques shot_type → aspect_ratio :
          - wide, establishing, aerial → "16:9" (paysage)
          - close_up, extreme_close_up, macro → "4:5" (portrait modéré)
          - medium, medium_close_up → "4:5" ou "9:16"
          - tout autre → "9:16" (par défaut Shorts)
        """
        if plan_ratio in _ASPECT_RATIOS:
            return plan_ratio

        # Heuristique basée sur le type de plan
        landscape_shots = {"wide", "establishing", "aerial", "cowboy", "two_shot"}
        portrait_shots = {"close_up", "extreme_close_up", "macro", "pov"}
        square_shots = {"over_the_shoulder"}

        if scene.shot_type in landscape_shots:
            return "16:9"
        elif scene.shot_type in portrait_shots:
            return "4:5"
        elif scene.shot_type in square_shots:
            return "4:3"
        else:
            return "9:16"

    @staticmethod
    def _resolve_dimensions(aspect_ratio: str, scene: VisualScene) -> tuple[int, int]:
        """
        Résout les dimensions en pixels.

        Priorité :
          1. Si le ratio est dans _ASPECT_RATIOS → on prend les dimensions standard
          2. Sinon → on calcule depuis le ratio

        Pour les macros et extremely close ups, on utilise la résolution max.
        """
        # Vérifie si on doit utiliser la haute résolution
        use_hd = scene.shot_type in {"macro", "extreme_close_up", "aerial"}

        if use_hd and aspect_ratio in _ASPECT_RATIOS_FULL:
            return _ASPECT_RATIOS_FULL[aspect_ratio]

        if aspect_ratio in _ASPECT_RATIOS:
            return _ASPECT_RATIOS[aspect_ratio]

        # Fallback : calcul depuis le ratio
        if ":" in aspect_ratio:
            parts = aspect_ratio.split(":")
            try:
                w_ratio = float(parts[0])
                h_ratio = float(parts[1])
                # Base : 1024px sur le côté le plus long
                if w_ratio >= h_ratio:
                    width = 1024
                    height = int(1024 * h_ratio / w_ratio)
                else:
                    height = 1024
                    width = int(1024 * w_ratio / h_ratio)
                # Arrondir au multiple de 64 (requis par la plupart des modèles)
                width = (width // 64) * 64
                height = (height // 64) * 64
                return (max(256, width), max(256, height))
            except (ValueError, ZeroDivisionError):
                pass

        return (720, 1280)  # Fallback 9:16

    @staticmethod
    def _build_prompt(
        scene: VisualScene,
        style_keywords: str,
        palette_str: str,
        aspect_ratio: str,
        width: int,
        height: int,
    ) -> str:
        """
        Construit le prompt final optimisé pour génération d'image.

        Format :
          [composition]. [éclairage]. Style: [style_keywords].
          Couleurs: [palette]. Ratio: [ratio] ([largeur]x[hauteur]).

        Le prompt intègre aussi overlay_text si présent.
        """
        parts = [
            scene.composition.strip().rstrip("."),
            scene.lighting.strip().rstrip("."),
        ]

        if style_keywords:
            parts.append(f"Style: {style_keywords}")

        if palette_str:
            parts.append(f"Color palette: {palette_str}")

        # Overlay text si présent et pertinent
        if scene.overlay_text and scene.overlay_text not in {"", " ", "-"}:
            parts.append(f"Text overlay: \"{scene.overlay_text}\"")

        parts.append(f"Aspect ratio: {aspect_ratio} ({width}x{height})")

        return ". ".join(parts) + "."

    @staticmethod
    def _compute_seed(prompt: str, scene_order: int) -> int:
        """
        Calcule un seed déterministe à partir du prompt.

        Garantit :
          - Même prompt + même order → même seed (reproductible)
          - Prompts différents → seeds très différents
          - Seed dans [0, 2^32 - 1] (compatible tous les générateurs)
        """
        hash_input = f"{prompt}_{scene_order}"
        hash_bytes = hashlib.md5(hash_input.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big")
        return seed % (2**32)

    @staticmethod
    def _resolve_quality(plan: VisualPlan, scene: VisualScene) -> str:
        """
        Résout la qualité de l'image.

        Règles :
          - Scènes clés (Hook, CTA, Rebondissement) → "hd"
          - Plans macro ou aériens → "hd"
          - Sinon → "standard"
        """
        hd_scenes = {"Hook", "CTA", "Rebondissement", "Verdict", "Point #3"}
        hd_shots = {"macro", "aerial", "extreme_close_up"}

        scene_title = scene.metadata.get("script_scene_title", "")
        if scene_title in hd_scenes or scene.shot_type in hd_shots:
            return "hd"

        return "standard"

    @staticmethod
    def _resolve_steps(plan: VisualPlan, scene: VisualScene) -> int:
        """
        Résout le nombre d'étapes de génération.

        Règles :
          - Qualité "hd" → 30 steps
          - Scènes clés → 25 steps
          - Standard → 20 steps
        """
        quality = HeuristicImageGenerator._resolve_quality(plan, scene)
        if quality == "hd":
            return 30

        key_scenes = {"Hook", "Conclusion", "CTA"}
        scene_title = scene.metadata.get("script_scene_title", "")
        if scene_title in key_scenes:
            return 25

        return 20


# ── ImageEngine ─────────────────────────────────────────────────────────────

class ImageEngine:
    """
    Orchestrateur du Image Engine.

    Transforme un VisualPlan en liste de GeneratedImage (une par scène),
    prêtes à être envoyées à n'importe quel provider d'image.

    Exemple minimal (HeuristicImageGenerator automatique) :
        engine = ImageEngine()
        images = engine.generate(visual_plan)
        for img in images:
            print(f"Scène #{img.scene_order}: {img.width}x{img.height} seed={img.seed}")

    Avec un provider Flux (Sprint 20) :
        engine = ImageEngine(generator=FluxImageGenerator())
        images = engine.generate(visual_plan)

    Avec un provider DALL-E :
        engine = ImageEngine(generator=OpenAiImageGenerator())
        images = engine.generate(visual_plan)

    Génération multiple :
        images_list = engine.generate_all([plan1, plan2])

    Le moteur ne connaît aucun autre moteur du système.
    Il ne manipule que : VisualPlan → List[GeneratedImage].
    """

    def __init__(self, generator: Optional[ImageGenerator] = None) -> None:
        self._generator = generator or HeuristicImageGenerator()

    @property
    def generator_name(self) -> str:
        return self._generator.name

    # ── Interface publique ─────────────────────────────────────────────────────

    def generate(self, plan: VisualPlan) -> List[GeneratedImage]:
        """
        Génère une liste de GeneratedImage pour chaque scène d'un VisualPlan.

        Args:
            plan: Plan visuel complet (scènes avec prompts, composition, palette...).

        Returns:
            Liste ordonnée de GeneratedImage (une par scène).

        Raises:
            ValueError: si le plan n'a pas de scènes.
        """
        if not plan.scenes:
            raise ValueError(
                f"Impossible de générer des images : "
                f"le plan '{plan.title[:50]}' n'a aucune scène."
            )

        images: List[GeneratedImage] = []
        for scene in plan.scenes:
            try:
                img = self._generator.generate(scene, plan)
                images.append(img)
            except Exception as exc:
                logger.warning(
                    "Échec generation image scene #%d du plan '%s' : %s",
                    scene.scene_order, plan.title[:50], exc,
                )
                # On continue avec les autres scènes

        logger.info(
            "%d image(s) générée(s) pour le plan '%s' (générateur: %s).",
            len(images), plan.title[:50], self.generator_name,
        )
        return images

    def generate_all(self, plans: List[VisualPlan]) -> List[List[GeneratedImage]]:
        """
        Génère des images pour une liste de VisualPlans.

        Args:
            plans: Liste de plans visuels.

        Returns:
            Liste de listes de GeneratedImage (même ordre que les plans).
        """
        results: List[List[GeneratedImage]] = []
        total_images = 0
        errors = 0

        for plan in plans:
            try:
                images = self.generate(plan)
                results.append(images)
                total_images += len(images)
            except Exception as exc:
                logger.warning(
                    "Échec generation images pour le plan '%s' : %s",
                    plan.title[:50], exc,
                )
                results.append([])
                errors += 1

        if errors:
            logger.warning(
                "%d plan(s) sur %d en échec. %d image(s) totale(s).",
                errors, len(plans), total_images,
            )
        else:
            logger.info(
                "%d plan(s) → %d image(s) (générateur: %s).",
                len(plans), total_images, self.generator_name,
            )

        return results
