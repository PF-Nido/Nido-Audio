from datetime import datetime

from fastapi import APIRouter

from nido.serving.db.database import check_db_connection
from nido.serving.db.redis_client import check_redis_connection

router = APIRouter()


@router.get("/health")
async def health_check():
    db_ok, db_error = await check_db_connection()
    redis_ok, redis_error = await check_redis_connection()

    status = "ok" if db_ok and redis_ok else "degraded"

    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_version": "dev",
        "services": {
            "database": "ok" if db_ok else f"error: {db_error}",
            "redis": "ok" if redis_ok else f"error: {redis_error}",
        },
    }


@router.get("/")
async def root():
    return {
        "project": "NIDO",
        "description": "Sistema de identificación de aves de Colombia",
        "docs": "/docs",
    }
