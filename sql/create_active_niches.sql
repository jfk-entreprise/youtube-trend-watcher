-- ============================================================
-- Sprint 28 — Initialisation de la table active_niches
-- À exécuter dans l'éditeur SQL de Supabase (une seule fois)
-- ============================================================
--
-- Persiste l'état des niches actives entre les runs quotidiens du pipeline
-- (scripts/run_daily_pipeline.py) : on ne change de niche que lorsqu'une
-- nouvelle niche présente un potentiel significativement supérieur
-- (voir src/niche_selector.py — NicheSelector).

CREATE TABLE IF NOT EXISTS active_niches (
    id                    BIGSERIAL    PRIMARY KEY,

    -- Identité de la niche (correspond à Niche.name — NicheAnalyzer)
    niche_name            TEXT         NOT NULL UNIQUE,
    niche_score           DOUBLE PRECISION NOT NULL,

    -- Historique de sélection
    first_selected_date   DATE         NOT NULL,
    last_confirmed_date   DATE         NOT NULL,

    metadata              JSONB        DEFAULT '{}'::jsonb
);

-- Index sur niche_name (lookup rapide lors de la mise à jour quotidienne)
CREATE INDEX IF NOT EXISTS idx_active_niches_name
    ON active_niches (niche_name);
