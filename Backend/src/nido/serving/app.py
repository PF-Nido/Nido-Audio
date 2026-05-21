from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nido.serving.db.redis_client import close_redis, init_redis
from nido.serving.routes import health, predict, species


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    yield
    await close_redis()


app = FastAPI(
    title="NIDO API",
    description="Sistema inteligente de identificación de aves de Colombia",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Sistema"])
app.include_router(predict.router, tags=["Predicción"])
app.include_router(species.router, tags=["Especies"])