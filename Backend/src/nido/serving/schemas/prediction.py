from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    recorded_at: datetime
    elevation: Optional[int] = Field(None, ge=0, le=5500)


class SpeciesPrediction(BaseModel):
    scientific_name: str
    common_name_es: Optional[str] = None
    common_name_en: Optional[str] = None
    family: Optional[str] = None
    confidence: float
    audio_score: float
    context_score: float


class AudioQuality(BaseModel):
    snr_db: Optional[float]
    n_segments: int
    duration_seconds: float


class PredictionResponse(BaseModel):
    predictions: list[SpeciesPrediction]
    audio_quality: AudioQuality
    model_version: str
    processing_time_ms: int
    request_id: str
