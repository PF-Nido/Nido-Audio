import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from nido.ml.birdnet import analyze_audio
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

    # Leer y validar tamaño
    audio_bytes = await audio.read()
    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"Archivo demasiado grande: {size_mb:.1f}MB.",
        )

    # Validar coordenadas Colombia
    if not (-4.5 <= latitude <= 13.0 and -82.0 <= longitude <= -66.0):
        raise HTTPException(
            status_code=400,
            detail="Las coordenadas están fuera de Colombia.",
        )

    # Buscar en cache
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    cached = await get_cached_prediction(audio_hash)
    if cached:
        cached["request_id"] = str(uuid.uuid4())
        return PredictionResponse(**cached)
    print(">>> ANTES DE BIRDNET")
    # Analizar con BirdNET
    detections = analyze_audio(
        audio_bytes=audio_bytes,
        latitude=latitude,
        longitude=longitude,
        recorded_at=recorded_at,
        elevation=elevation,
    )

    if not detections:
        raise HTTPException(
            status_code=422,
            detail="No se detectaron aves en el audio.",
        )
    print(">>> DESPUÉS DE BIRDNET")
    # Construir respuesta
    predictions = [
        SpeciesPrediction(
            scientific_name=d["scientific_name"],
            common_name_en=d.get("common_name_en"),
            confidence=d["confidence"],
            audio_score=d["confidence"],
            context_score=d["confidence"],
        )
        for d in detections
    ]

    processing_time = int((time.time() - start_time) * 1000)
    request_id = str(uuid.uuid4())

    response = PredictionResponse(
        predictions=predictions,
        audio_quality=AudioQuality(
            snr_db=None,
            n_segments=len(detections),
            duration_seconds=round(size_mb, 2),
        ),
        model_version="birdnet-2.4",
        processing_time_ms=processing_time,
        request_id=request_id,
    )

    # Guardar en cache
    await cache_prediction(audio_hash, response.model_dump())

    # Guardar en DB
    log = PredictionLog(
        confidence=predictions[0].confidence,
        audio_score=predictions[0].audio_score,
        context_score=predictions[0].context_score,
        top5_predictions=json.dumps([p.model_dump() for p in predictions]),
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
    print("Guardando log de predicción request_id en la base de datos...")

    return response
