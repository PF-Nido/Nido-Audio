import os
from datetime import datetime

import psycopg2
from prefect import task

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://nido_user:password@localhost:5432/nido"
)

# Umbral mínimo para promover un modelo a producción
MIN_TOP5_ACCURACY = 0.75


@task(name="evaluate-and-register")
def evaluate_and_register(model_result: dict) -> dict:
    """
    Evalúa el modelo entrenado contra el golden dataset
    y lo registra en model_versions si supera el umbral.
    """
    if model_result.get("status") == "skipped":
        print("Entrenamiento fue saltado, nada que evaluar")
        return {"promoted": False, "reason": "training_skipped"}

    top5_accuracy = model_result.get("top5_accuracy", 0.0)
    print(f"Top-5 accuracy del modelo: {top5_accuracy:.2%}")

    # Verificar si supera el umbral mínimo
    if top5_accuracy < MIN_TOP5_ACCURACY:
        print(
            f"Modelo no supera el umbral mínimo "
            f"({MIN_TOP5_ACCURACY:.0%}), descartando"
        )
        return {
            "promoted": False,
            "reason": "below_threshold",
            "top5_accuracy": top5_accuracy,
        }

    # Registrar en la DB
    version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO model_versions (
                version,
                model_type,
                top5_accuracy,
                top1_accuracy,
                macro_f1,
                status,
                model_path,
                trained_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
            (
                version,
                "species_classifier",
                model_result.get("top5_accuracy"),
                model_result.get("top1_accuracy"),
                model_result.get("macro_f1"),
                "candidate",
                model_result.get("model_path"),
                datetime.now(),
            ),
        )

        conn.commit()
        cursor.close()
        conn.close()

        print(f"Modelo registrado como versión {version}")
        return {
            "promoted": True,
            "version": version,
            "top5_accuracy": top5_accuracy,
        }

    except Exception as e:
        print(f"Error registrando modelo en DB: {e}")
        return {
            "promoted": False,
            "reason": f"db_error: {e}",
        }
