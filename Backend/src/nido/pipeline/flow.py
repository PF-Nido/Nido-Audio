from prefect import flow

from nido.pipeline.tasks.evaluate_model import evaluate_and_register
from nido.pipeline.tasks.extract_embeddings import extract_embeddings
from nido.pipeline.tasks.fetch_data import fetch_validated_recordings
from nido.pipeline.tasks.train_model import train_model


@flow(
    name="nido-retraining-pipeline",
    description="Pipeline de reentrenamiento del modelo de clasificación de aves",
)
def retraining_pipeline(max_recordings: int = 100) -> dict:
    """
    Pipeline completo de reentrenamiento:
    1. Obtiene grabaciones validadas de la DB
    2. Extrae embeddings con BirdNET
    3. Entrena el modelo (placeholder por ahora)
    4. Evalúa y registra si supera el umbral
    """
    print("Iniciando pipeline de reentrenamiento NIDO")

    # Step 1: Obtener datos
    recordings = fetch_validated_recordings(limit=max_recordings)

    # Step 2: Extraer embeddings
    recordings_with_embeddings = extract_embeddings(recordings)

    # Step 3: Entrenar modelo
    model_result = train_model(recordings_with_embeddings)

    # Step 4: Evaluar y registrar
    evaluation = evaluate_and_register(model_result)

    print(f"Pipeline finalizado: {evaluation}")
    return evaluation


if __name__ == "__main__":
    # Correr manualmente
    retraining_pipeline()
