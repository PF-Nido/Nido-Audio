import subprocess
import sys

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from nido.serving.db.database import get_db

router = APIRouter(prefix="/training", tags=["Reentrenamiento"])

MIN_SAMPLES_TO_RETRAIN = 10
RETRAIN_EVERY_N_SAMPLES = 1000

# Estado interno del pipeline (en memoria, simple para MVP)
_pipeline_running = False


# ══════════════════════════════════════════════════════════════════
# GET /training/status
# Datos actuales disponibles para reentrenamiento
# ══════════════════════════════════════════════════════════════════


@router.get("/status")
async def get_training_status(db: AsyncSession = Depends(get_db)):
    """
    Retorna los datos disponibles actualmente para reentrenamiento
    y el estado del sistema.
    """

    # Total de grabaciones y por split
    totals_query = await db.execute(
        text(
            """
        SELECT
            COUNT(*)                                          AS total,
            COUNT(*) FILTER (WHERE split = 'train')          AS train,
            COUNT(*) FILTER (WHERE split = 'val')            AS val,
            COUNT(*) FILTER (WHERE split = 'test')           AS test,
            COUNT(*) FILTER (WHERE embedding_path IS NULL
                             AND split = 'train')            AS pending_embeddings,
            COUNT(*) FILTER (WHERE is_golden = TRUE)         AS golden
        FROM recordings
    """
        )
    )
    row = totals_query.fetchone()

    total_recordings = int(row.total or 0)  # type: ignore
    train_count = int(row.train or 0)  # type: ignore
    val_count = int(row.val or 0)  # type: ignore
    test_count = int(row.test or 0)  # type: ignore
    pending_embeddings = int(row.pending_embeddings or 0)  # type: ignore
    golden_count = int(row.golden or 0)  # type: ignore

    # Modelo actual en producción
    model_query = await db.execute(
        text(
            """
        SELECT version, top5_accuracy, top1_accuracy,
               macro_f1, trained_at, deployed_at
        FROM model_versions
        WHERE status = 'production'
        ORDER BY deployed_at DESC
        LIMIT 1
    """
        )
    )
    current_model = model_query.fetchone()

    # Progreso hacia el siguiente reentrenamiento
    samples_since_last = train_count % RETRAIN_EVERY_N_SAMPLES
    samples_until_next = RETRAIN_EVERY_N_SAMPLES - samples_since_last
    progress_pct = round(samples_since_last / RETRAIN_EVERY_N_SAMPLES * 100, 1)
    can_retrain = train_count >= MIN_SAMPLES_TO_RETRAIN

    return {
        "system": {
            "status": "running" if _pipeline_running else "idle",
            "can_retrain": can_retrain,
            "pipeline_active": _pipeline_running,
        },
        "data": {
            "total_recordings": total_recordings,
            "validated": train_count + val_count + test_count,
            "by_split": {
                "train": train_count,
                "val": val_count,
                "test": test_count,
            },
            "pending_embeddings": pending_embeddings,
            "golden_dataset": golden_count,
        },
        "retraining": {
            "min_samples_required": MIN_SAMPLES_TO_RETRAIN,
            "retrain_every_n_samples": RETRAIN_EVERY_N_SAMPLES,
            "samples_since_last_retrain": samples_since_last,
            "samples_until_next_retrain": samples_until_next,
            "progress_pct": progress_pct,
        },
        "current_model": {
            "version": current_model.version if current_model else None,
            "top5_accuracy": (
                float(current_model.top5_accuracy)
                if current_model and current_model.top5_accuracy
                else None
            ),
            "top1_accuracy": (
                float(current_model.top1_accuracy)
                if current_model and current_model.top1_accuracy
                else None
            ),
            "macro_f1": (
                float(current_model.macro_f1)
                if current_model and current_model.macro_f1
                else None
            ),
            "trained_at": (
                current_model.trained_at.isoformat()
                if current_model and current_model.trained_at
                else None
            ),
            "deployed_at": (
                current_model.deployed_at.isoformat()
                if current_model and current_model.deployed_at
                else None
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════
# GET /training/history
# Historial de versiones del modelo
# ══════════════════════════════════════════════════════════════════


@router.get("/history")
async def get_training_history(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """
    Retorna el historial de versiones entrenadas con sus métricas.
    Alimenta el componente de historial del dashboard.
    """
    query = await db.execute(
        text(
            """
        SELECT
            version,
            model_type,
            top1_accuracy,
            top5_accuracy,
            macro_f1,
            latency_p95_ms,
            status,
            trained_at,
            deployed_at,
            training_notes
        FROM model_versions
        ORDER BY trained_at DESC
        LIMIT :limit
    """
        ),
        {"limit": limit},
    )

    rows = query.fetchall()

    # Calcular delta de precisión respecto a la versión anterior
    versions = []
    for i, row in enumerate(rows):
        current_acc = float(row.top5_accuracy) if row.top5_accuracy else None
        previous_acc = (
            float(rows[i + 1].top5_accuracy)
            if (i + 1 < len(rows) and rows[i + 1].top5_accuracy)
            else None
        )

        delta = None
        if current_acc is not None and previous_acc is not None:
            delta = round(current_acc - previous_acc, 4)

        versions.append(
            {
                "version": row.version,
                "model_type": row.model_type,
                "top1_accuracy": (
                    float(row.top1_accuracy) if row.top1_accuracy else None
                ),
                "top5_accuracy": current_acc,
                "macro_f1": float(row.macro_f1) if row.macro_f1 else None,
                "latency_p95_ms": (
                    int(row.latency_p95_ms) if row.latency_p95_ms else None
                ),
                "status": row.status,
                "trained_at": row.trained_at.isoformat() if row.trained_at else None,
                "deployed_at": row.deployed_at.isoformat() if row.deployed_at else None,
                "training_notes": row.training_notes,
                "delta_accuracy": delta,
            }
        )

    return {
        "total_versions": len(versions),
        "versions": versions,
    }


# ══════════════════════════════════════════════════════════════════
# POST /training/trigger
# Activa el pipeline de reentrenamiento manualmente
# ══════════════════════════════════════════════════════════════════


def _run_pipeline_background(max_recordings: int):
    """Corre el pipeline en un proceso separado para no bloquear la API."""
    global _pipeline_running
    _pipeline_running = True
    try:
        subprocess.run(
            [sys.executable, "-m", "nido.pipeline.flow"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error en pipeline: {e.stderr.decode()}")
    finally:
        _pipeline_running = False


@router.post("/trigger")
async def trigger_retraining(
    background_tasks: BackgroundTasks,
    max_recordings: int = 1000,
    db: AsyncSession = Depends(get_db),
):
    """
    Activa el pipeline de reentrenamiento en segundo plano.
    Retorna inmediatamente sin esperar a que termine.
    """
    global _pipeline_running

    if _pipeline_running:
        raise HTTPException(
            status_code=409,
            detail="Ya hay un pipeline de reentrenamiento en ejecución.",
        )

    # Verificar que hay datos suficientes
    count_query = await db.execute(
        text(
            """
        SELECT COUNT(*) FROM recordings WHERE split = 'train'
    """
        )
    )
    train_count = count_query.scalar() or 0

    if train_count < MIN_SAMPLES_TO_RETRAIN:
        raise HTTPException(
            status_code=422,
            detail=f"Datos insuficientes: {train_count} muestras. "
            f"Se requieren {MIN_SAMPLES_TO_RETRAIN}.",
        )

    background_tasks.add_task(_run_pipeline_background, max_recordings)

    return {
        "status": "triggered",
        "message": "Pipeline de reentrenamiento iniciado en segundo plano.",
        "max_recordings": max_recordings,
    }


# ══════════════════════════════════════════════════════════════════
# GET /training/schedule
# Estado del scheduler automático
# ══════════════════════════════════════════════════════════════════


@router.get("/schedule")
async def get_schedule():
    """
    Retorna la configuración del scheduler automático.
    """
    from nido.pipeline.scheduler import get_schedule_status

    return get_schedule_status()


@router.post("/schedule")
async def update_schedule(
    enabled: bool = True,
    frequency: str = "weekly",  # daily, weekly, monthly
    min_samples: int = 500,
):
    """
    Actualiza la configuración del scheduler automático.
    """
    if frequency not in ("daily", "weekly", "monthly"):
        raise HTTPException(
            status_code=400,
            detail="frequency debe ser daily, weekly o monthly.",
        )

    from nido.pipeline.scheduler import update_schedule_config

    return update_schedule_config(enabled, frequency, min_samples)
