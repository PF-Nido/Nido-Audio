from prefect import flow

from nido.pipeline.tasks.evaluate_model import evaluate_and_register
from nido.pipeline.tasks.extract_embeddings import extract_embeddings
from nido.pipeline.tasks.fetch_data import fetch_validated_recordings
from nido.pipeline.tasks.train_model import train_audio_model, train_geo_model


@flow(
    name="nido-retraining-pipeline",
    description="Reentrenamiento incremental del Modelo A y Modelo B de NIDO",
)
def retraining_pipeline(max_recordings: int = 100) -> dict:
    print("Iniciando pipeline de reentrenamiento NIDO")

    # Step 1: Obtener grabaciones validadas
    recordings = fetch_validated_recordings(limit=max_recordings)

    # Step 2: Extraer embeddings BirdNET
    recordings_with_embeddings = extract_embeddings(recordings)

    # Step 3: Reentrenar ambos modelos en paralelo
    audio_result = train_audio_model(recordings_with_embeddings)
    geo_result = train_geo_model(recordings_with_embeddings)

    # Step 4: Evaluar y promover si ambos pasan
    evaluation = evaluate_and_register(audio_result, geo_result)

    print(f"Pipeline finalizado: {evaluation}")
    return evaluation


if __name__ == "__main__":
    retraining_pipeline()
