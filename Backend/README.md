# Nido Audio Processing

# NIDO Backend

Backend inteligente para proyecto NIDO Sistema Inteligente Multimodal de Identificación Bioacústica y Predicción Espaciotemporal de Aves en Colombia.

---

## Descripción

NIDO es un sistema desarrollado para la clasificación automática de especies de aves a partir de grabaciones de campo. El proyecto combina modelos de análisis acústico con modelos geoespaciales y una estrategia propia de fusión multimodal mediante filtro de umbral.

El sistema recibe audios de aves junto con información contextual (ubicación, fecha, elevación), extrae características acústicas, genera embeddings y produce predicciones de especies usando modelos de machine learning especializados.

Está especialmente orientado a especies de aves de Colombia.

---

# Características principales

- API REST desarrollada con FastAPI
- Procesamiento de audio y extracción de características acústicas
- Generación de embeddings acústicos
- Modelos de clasificación con LightGBM
- Fusión multimodal audio + geolocalización
- Sistema de caché con Redis
- Persistencia de datos con SQLAlchemy y PostgreSQL
- Validación geográfica de coordenadas de Colombia
- Registro de predicciones y métricas de inferencia

---

# Stack tecnológico

| Tecnología | Uso |
|---|---|
| Python | Backend principal |
| FastAPI | API REST |
| Uvicorn | Servidor ASGI |
| Librosa | Procesamiento de audio |
| LightGBM | Modelos de Machine Learning |
| SQLAlchemy | ORM |
| AsyncPG | Conexión asíncrona PostgreSQL |
| Redis | Caché |
| BirdNET | Clasificación acústica |
| Perch-Hoplite | Embeddings acústicos |

---

# Arquitectura general

```text
Audio de campo
       ↓
Procesamiento acústico
       ↓
Extracción de embeddings
       ↓
Modelo acústico 
       ↓
Modelo geográfico/contextual
       ↓
Fusión multimodal (Filtro de umbral)
       ↓
Predicción final de especies
```

---

# Endpoint principal

## `POST /predict`

Permite enviar una grabación de audio junto con información contextual para obtener predicciones de especies de aves.

### Parámetros

| Campo | Tipo | Descripción |
|---|---|---|
| audio | file | Archivo de audio |
| latitude | float | Latitud de la grabación |
| longitude | float | Longitud de la grabación |
| recorded_at | datetime | Fecha y hora de grabación |
| elevation | int (opcional) | Elevación en metros |

---

## Formatos soportados

- MP3
- WAV
- OGG
- FLAC

Tamaño máximo permitido: **50 MB**

---

# Ejemplo de solicitud

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "audio=@bird.wav" \
  -F "latitude=10.9878" \
  -F "longitude=-74.7889" \
  -F "recorded_at=2026-05-25T10:00:00"
```

---

# Ejemplo de respuesta

```json
{
  "predictions": [
    {
      "scientific_name": "Tangara arthus",
      "confidence": 0.91,
      "audio_score": 0.95,
      "context_score": 0.72
    }
  ],
  "audio_quality": {
    "snr_db": 18.2,
    "n_segments": 4,
    "duration_seconds": 12.5
  },
  "model_version": "nido-v1.0",
  "processing_time_ms": 1342,
  "request_id": "0b5d0c55-7e7b-4f5b-8f67-d5b18ef2d0d"
}
```

---

# Flujo de inferencia

El pipeline de inferencia realiza las siguientes etapas:

1. Validación del archivo de audio
2. Validación geográfica de coordenadas dentro de Colombia
3. Verificación de caché mediante hash SHA-256
4. Inferencia del modelo acústico
5. Inferencia del modelo geográfico/contextual
6. Fusión multimodal mediante filtro de umbral
7. Generación de respuesta
8. Almacenamiento en Redis y PostgreSQL

---

# Instalación

```bash

cd backend

pip install -r .
```

---

# Ejecución

```bash
cd src
uvicorn app.main:app --reload
```

# Ejecución con Docker

## Levantar servicios

```bash
docker compose up --build
```

Esto iniciará:

- API backend NIDO
- PostgreSQL + PostGIS
- Redis

---

## Servicios incluidos

| Servicio | Puerto |
|---|---|
| Backend API | 8000 |
| PostgreSQL/PostGIS | 5432 |
| Redis | 6379 |

---

## Detener contenedores

```bash
docker compose down
```

---


# Objetivo del proyecto

El objetivo de NIDO es Desarrollar un sistema multimodal que adquiera, integre y fusione señales acústicas y variables espaciotemporales para potenciar la identificación automatizada de especies de aves en Colombia y reducir errores derivados del contexto geográfico y temporal.

---

# Estado del proyecto

Proyecto experimental y de investigación enfocado en clasificación multimodal de aves colombianas mediante inteligencia artificial.