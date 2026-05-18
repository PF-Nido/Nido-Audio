import hashlib
import json
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from nido.ml.audio_model import predict_audio
from nido.ml.fusion import fusion_filtro_umbral
from nido.ml.geo_model import predict_geo
from nido.serving.db.database import get_db
from nido.serving.db.models import PredictionLog
from nido.serving.db.redis_client import cache_prediction
from nido.serving.schemas.prediction import (
    AudioQuality,
    PredictionResponse,
    SpeciesPrediction,
)

router = APIRouter()

ALLOWED_FORMATS = {"audio/mpeg", "audio/wav", "audio/ogg", "audio/flac"}
MAX_FILE_SIZE_MB = 50


@router.post("/predict")
async def predict(
    audio: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    recorded_at: datetime = Form(...),
    elevation: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    print("ENTRÓ")
    return {"ok": True}


@router.post("/predict2", response_model=PredictionResponse)
async def predict2(
    audio: UploadFile = File(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    recorded_at: datetime = Form(...),
    elevation: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    start_time = time.time()
    print("Recibida solicitud de predicción")
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
    # cached = await get_cached_prediction(audio_hash)
    # if cached:
    #    cached["request_id"] = str(uuid.uuid4())
    #    print(">>> RESPONDIENDO DESDE CACHE")
    #    return PredictionResponse(**cached)
    # print(">>> ANTES DE modelos de audio y geo")
    # Modelo audio
    audio_probs = predict_audio(audio_bytes)
    if not audio_probs:
        raise HTTPException(
            status_code=422,
            detail="No se detectaron aves en el audio.",
        )
    print(">>> DESPUÉS DE modelo de audio")
    print(
        f"Audio probs: {list(audio_probs.items())[:5]}"
    )  # Mostrar solo un fragmento para no saturar el log
    # Modelo geo

    # Modelo geo
    geo_probs = predict_geo(
        latitude=latitude,
        longitude=longitude,
        month=recorded_at.month,
        day_of_year=recorded_at.timetuple().tm_yday,
        elevation=float(elevation) if elevation else None,
    )
    print(">>> DESPUÉS DE modelo geo")

    # Fusión
    detections, filtro_aplicado = fusion_filtro_umbral(
        audio_probs=audio_probs,
        geo_probs=geo_probs,
    )
    print(f">>> DESPUÉS DE fusión (filtro aplicado: {filtro_aplicado})")
    # Construir respuesta
    print(detections)
    predictions = [
        SpeciesPrediction(
            scientific_name=d["scientific_name"],
            confidence=d["confidence"],
            audio_score=audio_probs.get(d["scientific_name"], 0.0),
            context_score=geo_probs.get(d["scientific_name"], 0.0),
        )
        for d in detections
    ]
    processing_time = int((time.time() - start_time) * 1000)
    request_id = str(uuid.uuid4())
    response = PredictionResponse(
        predictions=predictions,
        audio_quality=AudioQuality(
            snr_db=None,
            n_segments=len(audio_probs),
            duration_seconds=round(size_mb, 2),
        ),
        model_version="nido-v1.0",
        processing_time_ms=processing_time,
        request_id=request_id,
    )
    return response
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
