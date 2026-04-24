import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from nido.serving.db.database import get_db
from nido.serving.db.models import PredictionLog
from nido.serving.db.redis_client import cache_prediction, get_cached_prediction
from nido.serving.schemas.prediction import (
    AudioQuality,
    PredictionResponse,
    SpeciesPrediction,
)

router = APIRouter()

ALLOWED_FORMATS = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"}
MAX_FILE_SIZE_MB = 50


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    audio: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    recorded_at: datetime = Form(...),
    elevation: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    start_time = time.time()

    # Validar formato
    if audio.content_type not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: {audio.content_type}.",
        )

    # Leer audio y validar tamaño
    audio_bytes = await audio.read()
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail="Archivo demasiado grande:",
        )

    # Validar coordenadas dentro de Colombia

    # Hash del audio para cache
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()

    # Buscar en cache de Redis
    cached = await get_cached_prediction(audio_hash)
    if cached:
        cached["request_id"] = str(uuid.uuid4())
        cached["from_cache"] = True
        return PredictionResponse(**cached)

    # pipeline real
    # Por ahora respuesta mock
    processing_time = int((time.time() - start_time) * 1000)
    request_id = str(uuid.uuid4())

    response = PredictionResponse(
        predictions=[
            SpeciesPrediction(
                scientific_name="Zonotrichia capensis",
                common_name_es="Copetón",
                family="Passerellidae",
                confidence=0.87,
                audio_score=0.82,
                context_score=0.94,
            )
        ],
        audio_quality=AudioQuality(
            snr_db=None,
            n_segments=0,
            duration_seconds=round(size_mb, 2),
        ),
        model_version="dev-mock",
        processing_time_ms=processing_time,
        request_id=request_id,
    )

    # Guardar en cache de Redis
    await cache_prediction(audio_hash, response.model_dump())

    # Guardar log en PostgreSQL
    log = PredictionLog(
        confidence=response.predictions[0].confidence,
        audio_score=response.predictions[0].audio_score,
        context_score=response.predictions[0].context_score,
        top5_predictions=json.dumps([p.model_dump() for p in response.predictions]),
        input_lat=latitude,
        input_lon=longitude,
        input_elevation=elevation,
        input_datetime=recorded_at,
        model_version=response.model_version,
        processing_time_ms=processing_time,
        request_id=uuid.UUID(request_id),
    )
    db.add(log)
    await db.commit()

    return response
