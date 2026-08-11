-- Table `series_concepts` (Sprint 39 — Gestion des séries épisodiques et bibles de personnages)
-- Exécuter dans le Dashboard Supabase > SQL Editor

CREATE TABLE IF NOT EXISTS public.series_concepts (
    series_id       TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    serie           TEXT NOT NULL,          -- identifiant de la série (ex: "ia", "histoire")
    brand_id        TEXT NOT NULL,
    market          TEXT NOT NULL DEFAULT 'FR',
    season_number   INTEGER NOT NULL DEFAULT 1,
    total_episodes  INTEGER NOT NULL DEFAULT 12,
    current_episode INTEGER NOT NULL DEFAULT 1,
    status          TEXT NOT NULL DEFAULT 'active', -- 'active' | 'completed' | 'archived'
    concept_json    JSONB NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index pour recherches rapides par série, marché et statut
CREATE INDEX IF NOT EXISTS idx_series_concepts_market_status ON public.series_concepts(market, status);
CREATE INDEX IF NOT EXISTS idx_series_concepts_serie_market ON public.series_concepts(serie, market);

COMMENT ON TABLE public.series_concepts IS 'Stocke le concept complet des séries vidéo, la bible de personnages et l''état d''avancement des épisodes (Sprint 39).';
