-- ============================================================
-- Sprint 34 — Ajout de la colonne market à active_niches
-- À exécuter dans l'éditeur SQL de Supabase (une seule fois)
-- ============================================================
--
-- Une même niche (ex. "IA") peut désormais être active simultanément sur
-- plusieurs marchés (US en anglais, FR en français) — la clé naturelle
-- devient (niche_name, market) au lieu de niche_name seul.
-- Voir src/niche_selector.py — NicheSelector.select_daily_niches(market=...).

ALTER TABLE active_niches ADD COLUMN IF NOT EXISTS market TEXT NOT NULL DEFAULT 'FR';

-- Retire l'ancienne contrainte d'unicité sur niche_name seul (le nom de la
-- contrainte suit la convention par défaut Postgres pour une colonne UNIQUE).
ALTER TABLE active_niches DROP CONSTRAINT IF EXISTS active_niches_niche_name_key;

ALTER TABLE active_niches ADD CONSTRAINT active_niches_niche_name_market_key
    UNIQUE (niche_name, market);

CREATE INDEX IF NOT EXISTS idx_active_niches_market
    ON active_niches (market);
