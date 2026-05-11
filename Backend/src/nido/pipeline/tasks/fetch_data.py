import os

import psycopg2
from prefect import task

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://nido_user:password@localhost:5432/nido"
)


@task(name="fetch-validated-recordings")
def fetch_validated_recordings(limit: int = 100) -> list[dict]:
    """
    Obtiene grabaciones validadas de la DB que aún
    no tienen embedding extraído y están en split train.
    """
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            id,
            xeno_canto_id,
            species_id,
            audio_path,
            elevation
        FROM recordings
        WHERE split = 'train'
        AND embedding_path IS NULL
        LIMIT %s
    """,
        (limit,),
    )

    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]  # type: ignore

    cursor.close()
    conn.close()

    recordings = [dict(zip(columns, row)) for row in rows]
    print(f"Encontradas {len(recordings)} grabaciones para procesar")
    return recordings
