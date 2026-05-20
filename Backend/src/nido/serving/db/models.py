import uuid

from geoalchemy2 import Geometry
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from nido.serving.db.database import Base


class Species(Base):
    __tablename__ = "species"

    id = Column(Integer, primary_key=True)
    scientific_name = Column(String(200), unique=True, nullable=False)
    common_name_es = Column(String(200))
    common_name_en = Column(String(200))
    genus = Column(String(100), nullable=False)
    family_id = Column(Integer, ForeignKey("families.id"), nullable=False)
    elevation_min = Column(Integer, default=0)
    elevation_max = Column(Integer, default=5500)
    distribution = Column(Geometry("MULTIPOLYGON", srid=4326))
    iucn_status = Column(String(10))
    is_endemic = Column(Boolean, default=False)
    is_migratory = Column(Boolean, default=False)
    ebird_species_code = Column(String(20))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(Integer, primary_key=True)
    recording_id = Column(Integer, nullable=True)  # sin ForeignKey por ahora
    predicted_species_id = Column(Integer, nullable=True)
    predicted_family_id = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=False)
    audio_score = Column(Float)
    context_score = Column(Float)
    top5_predictions = Column(JSONB)
    input_lat = Column(Float)
    input_lon = Column(Float)
    input_elevation = Column(Integer)
    input_datetime = Column(DateTime)
    snr_estimated = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Integer)
    predicted_at = Column(DateTime, server_default=func.now())
    request_id = Column(UUID(as_uuid=True), default=uuid.uuid4)
