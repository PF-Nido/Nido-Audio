import json
import os
from typing import Optional

import redis.asyncio as aioredis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Cliente global, se inicializa al arrancar la API
redis_client: Optional[aioredis.Redis] = None


async def init_redis() -> None:
    """Inicializa la conexión a Redis."""
    global redis_client
    redis_client = aioredis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )


async def close_redis() -> None:
    """Cierra la conexión a Redis."""
    global redis_client
    if redis_client:
        await redis_client.close()


# redis_client.py
async def check_redis_connection() -> tuple[bool, str]:
    try:
        if redis_client:
            await redis_client.ping()  # type: ignore[misc]
            return True, ""
        return False, "Redis client not initialized"
    except Exception as e:
        return False, str(e)


async def get_cached_prediction(audio_hash: str) -> Optional[dict]:
    """
    Busca una predicción cacheada por hash del audio.
    Evita reprocesar el mismo audio dos veces.
    """
    try:
        if redis_client:
            cached = await redis_client.get(f"prediction:{audio_hash}")
            if cached:
                return json.loads(cached)
        return None
    except Exception:
        return None


async def cache_prediction(
    audio_hash: str, prediction: dict, ttl_seconds: int = 86400  # 24 horas
) -> None:
    """Guarda una predicción en cache."""
    try:
        if redis_client:
            await redis_client.setex(
                f"prediction:{audio_hash}",
                ttl_seconds,
                json.dumps(prediction),
            )
    except Exception:
        pass  # Si falla el cache no es crítico, la API sigue funcionandos
