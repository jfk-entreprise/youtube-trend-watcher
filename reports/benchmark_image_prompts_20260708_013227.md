# Benchmark — Prompt Heuristique vs Prompt LLM (Sprint 23)

Comparaison de la richesse descriptive des prompts d'image sur 8 dimensions (composition, caméra, lumière, ambiance, couleurs, profondeur, style cinématographique, niveau de détail). Détection par mots-clés — indicateur de richesse descriptive, pas une note de qualité d'image réelle (aucun rendu).

## Détail par scène

| Scène | Heuristique /8 | LLM /8 | Longueur (H) | Longueur (LLM) | Negative prompt (LLM) |
|-------|---------------:|-------:|-------------:|----------------:|:----------------------:|
| Hook | 3 | 8 | 384 | 706 | Oui |
| Contexte | 2 | 8 | 341 | 525 | Oui |
| Demonstration | 4 | 6 | 266 | 540 | Oui |
| Impact | 4 | 7 | 266 | 678 | Oui |
| Conclusion | 3 | 6 | 309 | 567 | Oui |

## Moyennes

- **Heuristique** : 3.2/8 dimensions couvertes en moyenne
- **LLM (deepseek)** : 7.0/8 dimensions couvertes en moyenne

## Détail des prompts

### Hook

**Heuristique** (3/8) :
> Centrage dynamique avec typographie imposante. Sujet au centre, espace négatif maîtrisé. Rembrandt. Key à 45°, triangle de lumière sur la joue. Fill minimal. Style: futuristic, cyberpunk aesthetic, neon accents, holographic. Color palette: #0D0D0D, #00D4FF, #7B2FBE, #FFFFFF. Text overlay: "Et si l'IA remplacait deja votre monteur video sans que vous". Aspect ratio: 9:16 (720x1280).

**LLM** (8/8) :
> A medium close-up shot of a video editor's hands on a keyboard, with a glowing holographic interface overlay displaying AI editing tools. The scene is centered with dynamic composition, negative space on top for typography. Cinematic Rembrandt lighting from a 45-degree key light, creating a triangle of light on the cheek, with minimal fill. Deep black background with neon blue and violet animated light lines. The atmosphere is innovative and futuristic, with a color palette of deep black (#0D0D0D), cyan (#00D4FF), violet (#7B2FBE), white (#FFFFFF), and dark blue (#1A1A2E). Shallow depth of field, sharp focus on the hands and interface. Ultra-detailed, 8k resolution, photorealistic cinematic style.

**Negative prompt (LLM)** : Blurry, deformed hands, distorted text, low resolution, inconsistent lighting, overly bright or dull colors, messy composition, cluttered background, unrealistic neon effects, cartoonish style, watermark, logo.

### Contexte

**Heuristique** (2/8) :
> Plan large centré. Horizon au tiers supérieur. Profondeur de champ pour capter l'environnement. Lumière ambiante naturelle. Source principale large. Ombres douces. Style: futuristic, cyberpunk aesthetic, neon accents, holographic. Color palette: #0D0D0D, #00D4FF, #7B2FBE, #FFFFFF. Text overlay: "Le contexte". Aspect ratio: 9:16 (720x1280).

**LLM** (8/8) :
> A wide shot of a modern film production studio, cinematic composition, horizon at upper third, deep depth of field capturing the entire environment. The scene is illuminated with ambient natural light, soft shadows, and a neon palette of deep black #0D0D0D, electric blue #00D4FF, violet #7B2FBE, white #FFFFFF, and dark navy #1A1A2E. The camera performs a slow pan left, revealing the studio space with smooth motion. Futuristic and innovative atmosphere, realistic style, 8k ultra-detailed, film grain, volumetric lighting.

**Negative prompt (LLM)** : blurry, deformed, distorted, watermark, text, logo, oversaturated, low contrast, cartoon, anime, 3d render, low quality, jpeg artifacts, dark areas without detail, lens flare, chromatic aberration.

### Demonstration

**Heuristique** (4/8) :
> Composition standard. Règle des tiers. Fond neutre. Lumière naturelle diffusée. Key à 30°, fill à 90%. Ambiance soft. Style: futuristic, cyberpunk aesthetic, neon accents, holographic. Color palette: #0D0D0D, #00D4FF, #7B2FBE, #FFFFFF. Aspect ratio: 9:16 (720x1280).

**LLM** (6/8) :
> A medium shot, static camera view of a futuristic interface on a dark screen, generating a complete video montage in seconds. The interface features glowing neon blue (#00D4FF) and purple (#7B2FBE) lines and elements, with a deep black (#0D0D0D) background. Soft diffused lighting from 30 degrees key light, 90% fill. Composition follows the rule of thirds, with the interface as the main focal point. Cinematic, photorealistic style, ultra-detailed, 8k resolution. Ambient glow, smooth animation lines, high contrast, vibrant neon accents.

**Negative prompt (LLM)** : blurry, distorted, low resolution, text or watermarks, noisy, unrealistic colors, flat lighting, cluttered background, incoherent style, cartoonish, oversaturated, underexposed, camera shake, motion blur, grainy, artifacts, deformed elements, inconsistent neon glow, dull, messy composition.

### Impact

**Heuristique** (4/8) :
> Composition standard. Règle des tiers. Fond neutre. Lumière naturelle diffusée. Key à 30°, fill à 90%. Ambiance soft. Style: futuristic, cyberpunk aesthetic, neon accents, holographic. Color palette: #0D0D0D, #00D4FF, #7B2FBE, #FFFFFF. Aspect ratio: 9:16 (720x1280).

**LLM** (7/8) :
> A professional video editor in a medium shot, static camera, standing in a dark studio with neon blue and violet glowing lines. The editor is focused on a holographic interface, hands gesturing over floating controls. Composition follows the rule of thirds, editor on the left, interface on the right. Soft diffused key light at 30 degrees, fill at 90%, creating a soft ambient glow. Deep black background (#0D0D0D) with neon cyan (#00D4FF) and violet (#7B2FBE) accents, white (#FFFFFF) highlights. Cinematic lighting, shallow depth of field with the editor in sharp focus, background blurred. Photorealistic, ultra-detailed, 8k resolution, futuristic and accessible atmosphere.

**Negative prompt (LLM)** : Blurry, distorted faces, unnatural skin tones, text or logos, cluttered background, warm colors, orange or red tones, low contrast, cartoon style, anime, oversaturated, lens flare, grain, noise, deformed hands, extra limbs, incoherent style.

### Conclusion

**Heuristique** (3/8) :
> Résumé en grille 2×2 ou 3×1. Éléments ordonnés. Hiérarchie claire. Lumière naturelle diffusée. Key à 30°, fill à 90%. Ambiance soft. Style: futuristic, cyberpunk aesthetic, neon accents, holographic. Color palette: #0D0D0D, #00D4FF, #7B2FBE, #FFFFFF. Text overlay: "Conclusion". Aspect ratio: 9:16 (720x1280).

**LLM** (6/8) :
> A futuristic 2x2 grid composition summarizing video editing innovation. Each cell shows a distinct phase: raw footage on top left, timeline editing on top right, effects application on bottom left, final polished video on bottom right. The grid is overlaid on a deep black background (#0D0D0D) with glowing neon cyan (#00D4FF) and violet (#7B2FBE) lines connecting the cells. Soft diffused key light at 30 degrees, fill at 90%. Medium shot, dolly-out camera movement revealing the entire grid. Cinematic, photorealistic, ultra-detailed, 8K. Moody, high-tech ambiance.

**Negative prompt (LLM)** : blurry, distorted, text, watermarks, low contrast, flat lighting, warm colors, messy composition, incoherent style, overexposed, underexposed, artifacts, noise, grainy

## Conclusion

✅ Le LLM produit des prompts plus riches en moyenne (+3.8 dimension(s)).
