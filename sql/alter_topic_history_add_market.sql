-- ============================================================
-- Sprint 34 — Ajout de la colonne market à topic_history
-- À exécuter dans l'éditeur SQL de Supabase (une seule fois)
-- ============================================================
--
-- Une niche peut être active simultanément sur plusieurs marchés (US en
-- anglais, FR en français) sans que ce soit un doublon — TopicHistoryFilter
-- doit comparer un sujet uniquement aux sujets déjà produits sur LE MÊME
-- marché. Voir src/topic_history.py — TopicRecord.market, classify(market=...).

ALTER TABLE topic_history ADD COLUMN IF NOT EXISTS market TEXT NOT NULL DEFAULT 'FR';

-- Remplace l'index (niche, produced_date) par un index incluant market,
-- cohérent avec le nouveau filtre de TopicHistoryFilter.classify().
DROP INDEX IF EXISTS idx_topic_history_niche_date;
CREATE INDEX IF NOT EXISTS idx_topic_history_niche_market_date
    ON topic_history (niche, market, produced_date);
