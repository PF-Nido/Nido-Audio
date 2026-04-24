-- =============================================
-- ÍNDICES ESPACIALES (GIST)
-- Críticos para queries geoespaciales con PostGIS
-- =============================================

-- Distribución de especies (el más importante)
CREATE INDEX IF NOT EXISTS idx_species_distribution
    ON species USING GIST (distribution);

-- Ubicación de grabaciones
CREATE INDEX IF NOT EXISTS idx_recordings_location
    ON recordings USING GIST (location);


-- =============================================
-- ÍNDICES DE BÚSQUEDA FRECUENTE
-- =============================================

-- Buscar grabaciones por especie
CREATE INDEX IF NOT EXISTS idx_recordings_species_id
    ON recordings (species_id);

-- Filtrar grabaciones por split (train/val/test)
CREATE INDEX IF NOT EXISTS idx_recordings_split
    ON recordings (split);

-- Filtrar grabaciones del golden dataset
CREATE INDEX IF NOT EXISTS idx_recordings_golden
    ON recordings (is_golden) WHERE is_golden = TRUE;

-- Buscar especies por familia
CREATE INDEX IF NOT EXISTS idx_species_family_id
    ON species (family_id);

-- Buscar por código de eBird
CREATE INDEX IF NOT EXISTS idx_species_ebird_code
    ON species (ebird_species_code);

-- Predicciones por modelo (para monitoreo de drift)
CREATE INDEX IF NOT EXISTS idx_predictions_model_version
    ON predictions_log (model_version);

-- Predicciones por fecha (para dashboards de uso)
CREATE INDEX IF NOT EXISTS idx_predictions_predicted_at
    ON predictions_log (predicted_at DESC);



-- Modelos en producción
CREATE INDEX IF NOT EXISTS idx_model_versions_status
    ON model_versions (status)
    WHERE status = 'production';

-- Cache de embeddings por hash
CREATE INDEX IF NOT EXISTS idx_embeddings_cache_hash
    ON embeddings_cache (audio_hash);


-- =============================================
-- ÍNDICE DE TEXTO para búsqueda de especies
-- =============================================
CREATE INDEX IF NOT EXISTS idx_species_scientific_name_text
    ON species USING GIN (to_tsvector('spanish', scientific_name));

CREATE INDEX IF NOT EXISTS idx_species_common_name_es_text
    ON species USING GIN (to_tsvector('spanish', unaccent(common_name_es)));