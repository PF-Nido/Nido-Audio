import os

import numpy as np
from prefect import task

from nido.ml.birdnet import get_analyzer


@task(name="extract-embeddings", retries=2)
def extract_embeddings(recordings: list[dict]) -> list[dict]:
    """
    Extrae embeddings de BirdNET para cada grabación.
    Retorna las grabaciones con su embedding adjunto.
    """
    if not recordings:
        print("No hay grabaciones para procesar")
        return []

    analyzer = get_analyzer()
    results = []

    for recording in recordings:
        audio_path = recording.get("audio_path")

        if not audio_path or not os.path.exists(audio_path):
            print(f"Audio no encontrado: {audio_path}, saltando...")
            continue

        try:
            from birdnetlib import Recording as BirdNetRecording

            rec = BirdNetRecording(
                analyzer,
                audio_path,
                min_conf=0.1,
            )
            rec.analyze()

            if rec.detections:
                # Usar la confianza promedio como embedding simplificado
                # Cuando llegue el modelo real esto será el vector 1024-d
                embedding = np.array([d["confidence"] for d in rec.detections])

                results.append(
                    {
                        **recording,
                        "embedding": embedding.tolist(),
                        "n_detections": len(rec.detections),
                    }
                )
                print(f"Embedding extraído para recording {recording['id']}")
            else:
                print(f"Sin detecciones para recording {recording['id']}")

        except Exception as e:
            print(f"Error procesando recording {recording['id']}: {e}")
            continue

    print(f"Embeddings extraídos: {len(results)}/{len(recordings)}")
    return results
