from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from nido.serving.db.redis_client import close_redis, init_redis
from nido.serving.routes import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al arrancar la API
    await init_redis()
    yield
    # Al apagar la API
    await close_redis()


app = FastAPI(
    title="NIDO API",
    description="Sistema inteligente de identificación de aves de Colombia",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — ajustar cuando haya frontend real
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas
app.include_router(health.router, tags=["Sistema"])
app.include_router(predict.router, tags=["Predicción"])
