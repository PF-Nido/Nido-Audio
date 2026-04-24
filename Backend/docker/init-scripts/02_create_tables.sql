-- =============================================
-- TABLA: orders
-- Nivel taxonómico más alto (Passeriformes, etc)
-- =============================================
CREATE TABLE IF NOT EXISTS orders (
    id              SERIAL PRIMARY KEY,
    scientific_name VARCHAR(200) NOT NULL UNIQUE,
    created_at      TIMESTAMP DEFAULT NOW()
);


-- =============================================
-- TABLA: families
-- Familias de aves (Tyrannidae, Thraupidae, etc)
-- =============================================
CREATE TABLE IF NOT EXISTS families (
    id              SERIAL PRIMARY KEY,
    scientific_name VARCHAR(200) NOT NULL UNIQUE,
    order_id        INT NOT NULL REFERENCES orders(id),
    created_at      TIMESTAMP DEFAULT NOW()
);


-- =============================================
-- TABLA: species
-- Especie individual con su distribución geográfica
-- =============================================
CREATE TABLE IF NOT EXISTS species (
    id                  SERIAL PRIMARY KEY,
    scientific_name     VARCHAR(200) NOT NULL UNIQUE,
    common_name_es      VARCHAR(200),
    common_name_en      VARCHAR(200),
    genus               VARCHAR(100) NOT NULL,
    family_id           INT NOT NULL REFERENCES families(id),

    -- Rango altitudinal en metros
    elevation_min       INT DEFAULT 0,
    elevation_max       INT DEFAULT 5500,

    -- Polígono de distribución geográfica en Colombia
    -- MULTIPOLYGON porque algunas especies tienen rangos discontinuos
    distribution        GEOMETRY(MULTIPOLYGON, 4326),

    -- Información de conservación
    iucn_status         VARCHAR(10),   -- LC, NT, VU, EN, CR, EW, EX
    is_endemic          BOOLEAN DEFAULT FALSE,
    is_migratory        BOOLEAN DEFAULT FALSE,

    -- Metadatos
    ebird_species_code  VARCHAR(20),   -- código único de eBird
    created_at          TIMESTAMP DEFAULT NOW(),
    updated_at          TIMESTAMP DEFAULT NOW()
);


-- =============================================
-- TABLA: recordings
-- Grabaciones de audio individuales
-- =============================================
CREATE TABLE IF NOT EXISTS recordings (
    id                  SERIAL PRIMARY KEY,
    xeno_canto_id       VARCHAR(20) UNIQUE,   -- ej: XC123456
    species_id          INT NOT NULL REFERENCES species(id),

    -- Ubicación geográfica
    location            GEOMETRY(POINT, 4326),
    elevation           INT,

    -- Fecha y hora de grabación
    recorded_at         TIMESTAMP,

    -- Metadatos del audio
    duration_seconds    FLOAT,
    sample_rate         INT,
    file_format         VARCHAR(10),   -- mp3, wav, ogg, flac

    -- Calidad
    snr_db              FLOAT,         -- señal a ruido estimado
    quality_rating      CHAR(1),       -- rating de Xeno-Canto: A, B, C, D, E
    has_background_species BOOLEAN DEFAULT FALSE,

    -- Rutas de archivos
    audio_path          VARCHAR(500),  -- ruta en Azure Blob Storage
    embedding_path      VARCHAR(500),  -- ruta al .npy del embedding

    -- ML
    split               VARCHAR(10)    -- 'train', 'val', 'test'
        CHECK (split IN ('train', 'val', 'test')),
    is_golden           BOOLEAN DEFAULT FALSE,

    -- Metadatos
    source              VARCHAR(50) DEFAULT 'xeno-canto',
    created_at          TIMESTAMP DEFAULT NOW()
);


-- =============================================
-- TABLA: embeddings_cache
-- Cache de embeddings ya procesados
-- Evita reprocesar el mismo audio dos veces
-- =============================================
CREATE TABLE IF NOT EXISTS embeddings_cache (
    id              SERIAL PRIMARY KEY,
    recording_id    INT NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,

    -- Hash del audio para detectar duplicados
    audio_hash      VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256

    -- Estadísticas del embedding (mean pooling de los segmentos)
    embedding_mean  FLOAT[],   -- vector 1024-d
    embedding_max   FLOAT[],   -- vector 1024-d
    embedding_std   FLOAT[],   -- vector 1024-d

    -- Info del procesamiento
    n_segments      INT,       -- cuántos segmentos de 3s se procesaron
    birdnet_version VARCHAR(20),

    created_at      TIMESTAMP DEFAULT NOW()
);


-- =============================================
-- TABLA: predictions_log
-- Registro de cada predicción hecha por la API
-- =============================================
CREATE TABLE IF NOT EXISTS predictions_log (
    id                  SERIAL PRIMARY KEY,

    -- Puede ser NULL si el audio viene de un usuario externo
    recording_id        INT REFERENCES recordings(id),

    -- Resultado
    predicted_species_id    INT REFERENCES species(id),
    predicted_family_id     INT REFERENCES families(id),
    confidence              FLOAT NOT NULL,
    audio_score             FLOAT,
    context_score           FLOAT,
    top5_predictions        JSONB,  -- guarda el top 5 completo

    -- Contexto de la predicción
    input_lat           FLOAT,
    input_lon           FLOAT,
    input_elevation     INT,
    input_datetime      TIMESTAMP,
    snr_estimated       FLOAT,

    -- Info técnica
    model_version       VARCHAR(50) NOT NULL,
    processing_time_ms  INT,
    predicted_at        TIMESTAMP DEFAULT NOW(),

    -- Trazabilidad
    request_id          UUID DEFAULT gen_random_uuid()
);



-- =============================================
-- TABLA: model_versions
-- Registro de modelos entrenados y desplegados
-- =============================================
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    version         VARCHAR(50) NOT NULL UNIQUE,  -- ej: v2.1.3
    model_type      VARCHAR(50) NOT NULL,          -- 'filter', 'family', 'species_tyrannidae', etc

    -- Métricas en golden dataset
    top1_accuracy   FLOAT,
    top5_accuracy   FLOAT,
    macro_f1        FLOAT,
    latency_p95_ms  INT,

    -- Estado
    status          VARCHAR(20) DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'staging', 'production', 'archived')),

    -- Rutas
    model_path      VARCHAR(500),  -- ruta en Azure Blob Storage

    -- Metadatos
    trained_at      TIMESTAMP,
    deployed_at     TIMESTAMP,
    archived_at     TIMESTAMP,
    training_notes  TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);


-- =============================================
-- TRIGGER: actualizar updated_at en species
-- =============================================
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER species_updated_at
    BEFORE UPDATE ON species
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();