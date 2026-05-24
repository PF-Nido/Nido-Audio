import os
import shutil
from datetime import datetime
from pathlib import Path

import psycopg2
from prefect import task

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://nido_user:password@localhost:5432/nido"
)

BASE_DIR = Path(__file__).resolve().parents[4]
AUDIO_MODEL_DIR = BASE_DIR / "models" / "audio"
GEO_MODEL_DIR = BASE_DIR / "models" / "geo"

MIN_TOP5_ACCURACY_AUDIO = 0.75
MIN_TOP1_ACCURACY_GEO = 0.02  # el modelo geo tiene métricas más bajas por diseño


@task(name="evaluate-and-register")
def evaluate_and_register(
    audio_result: dict,
    geo_result: dict,
) -> dict:
    """
    Evalúa los modelos reentrenados y los promueve a producción
    si superan los umbrales mínimos.

    La promoción es atómica: si un modelo no pasa, ninguno se promueve,
    para mantener consistencia entre el Modelo A y el Modelo B.
    """
    audio_ok = False
    geo_ok = False

    # Evaluar Modelo A
    if audio_result.get("status") == "success":
        avg_acc = audio_result.get("avg_accuracy", 0.0)
        audio_ok = avg_acc >= MIN_TOP5_ACCURACY_AUDIO
        print(
            f"Modelo A — accuracy promedio: {avg_acc:.4f} "
            f"({'✓' if audio_ok else '✗'} umbral: {MIN_TOP5_ACCURACY_AUDIO})"
        )
    else:
        print(f"Modelo A saltado: {audio_result.get('reason', 'sin datos')}")

    # Evaluar Modelo B
    if geo_result.get("status") == "success":
        top1_geo = geo_result.get("top1_accuracy", 0.0)
        geo_ok = top1_geo >= MIN_TOP1_ACCURACY_GEO
        print(
            f"Modelo B — Top-1: {top1_geo:.4f} "
            f"({'✓' if geo_ok else '✗'} umbral: {MIN_TOP1_ACCURACY_GEO})"
        )
    else:
        print(f"Modelo B saltado: {geo_result.get('reason', 'sin datos')}")

    # Ambos deben pasar para promover
    if not (audio_ok and geo_ok):
        print("Modelos no promovidos a producción.")
        return {
            "promoted": False,
            "audio_ok": audio_ok,
            "geo_ok": geo_ok,
        }

    # Promover: reemplazar archivos existentes con los nuevos
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        # Promover Modelo A — reemplazar cada fold
        for fold in range(1, 6):
            new_path = AUDIO_MODEL_DIR / f"modelo_embeddings_fold{fold}_new.txt"
            old_path = AUDIO_MODEL_DIR / f"modelo_embeddings_fold{fold}.txt"
            if new_path.exists():
                shutil.move(str(new_path), str(old_path))

        # Promover Modelo B
        new_geo = GEO_MODEL_DIR / "model_b_new.txt"
        old_geo = GEO_MODEL_DIR / "model_b.txt"
        if new_geo.exists():
            shutil.move(str(new_geo), str(old_geo))

        print(f"Modelos promovidos a producción: {version}")

        # Registrar en DB
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_versions (
                version, model_type,
                top1_accuracy, top5_accuracy,
                status, trained_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
            (
                version,
                "audio+geo",
                audio_result.get("avg_accuracy"),
                geo_result.get("top5_accuracy"),
                "production",
                datetime.now(),
            ),
        )

        # Archivar versión anterior
        cursor.execute(
            """
            UPDATE model_versions
            SET status = 'archived', archived_at = NOW()
            WHERE status = 'production'
            AND version != %s
        """,
            (version,),
        )

        conn.commit()
        cursor.close()
        conn.close()

        return {
            "promoted": True,
            "version": version,
            "audio_ok": audio_ok,
            "geo_ok": geo_ok,
        }

    except Exception as e:
        print(f"Error promoviendo modelos: {e}")
        return {
            "promoted": False,
            "reason": str(e),
        }
