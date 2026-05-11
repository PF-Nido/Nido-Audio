from prefect import task


@task(name="train-model")
def train_model(recordings_with_embeddings: list[dict]) -> dict:
    """
    Placeholder del entrenamiento del modelo LightGBM.
    Cuando la Persona A entregue el modelo real, este task
    se reemplaza con el entrenamiento real.
    """
    if not recordings_with_embeddings:
        print("Sin datos suficientes para entrenar")
        return {"status": "skipped", "reason": "no_data"}

    print(f"Entrenando con {len(recordings_with_embeddings)} grabaciones...")

    # TODO: reemplazar con entrenamiento real de LightGBM
    # model = lgb.LGBMClassifier(...)
    # model.fit(X_train, y_train)
    # model_path = save_model(model)
    pass

    print("Entrenamiento completado (placeholder)")

    return {
        "status": "success",
        "n_samples": len(recordings_with_embeddings),
        "model_path": "models/placeholder/model.pkl",
        "top1_accuracy": 0.0,  # placeholder
        "top5_accuracy": 0.0,  # placeholder
        "macro_f1": 0.0,  # placeholder
    }
