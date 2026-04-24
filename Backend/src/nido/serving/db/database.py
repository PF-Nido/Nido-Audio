import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://nido_user:password@localhost:5432/nido"
)

# SQLAlchemy usa postgresql+asyncpg para conexiones async
# Si la URL viene con postgresql:// la convertimos
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # True para ver las queries en los logs (útil en dev)
    pool_size=5,  # conexiones simultáneas
    max_overflow=10,  # conexiones extra en picos de tráfico
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """
    Dependency de FastAPI.
    Abre una sesión de DB por request y la cierra al terminar.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> tuple[bool, str]:
    try:
        async with engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return True, ""
    except Exception as e:
        return False, str(e) + DATABASE_URL
