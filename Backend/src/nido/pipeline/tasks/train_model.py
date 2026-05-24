import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from prefect import task
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parents[4]
AUDIO_MODEL_DIR = BASE_DIR / "models" / "audio"
GEO_MODEL_DIR = BASE_DIR / "models" / "geo"

N_FOLDS = 5


# ══════════════════════════════════════════════════════════════════
# MODELO A — Acústico (LightGBM K-Folds sobre embeddings BirdNET)
# ══════════════════════════════════════════════════════════════════


@task(name="train-audio-model")
def train_audio_model(recordings_with_embeddings: list[dict]) -> dict:
    """
    Reentrena incrementalmente el Modelo A (K-Folds LightGBM)
    sobre embeddings BirdNET.

    Incremental: carga los modelos existentes como init_model
    para continuar el entrenamiento desde donde quedaron,
    en lugar de empezar desde cero.
    """
    if not recordings_with_embeddings:
        print("Sin datos para reentrenar Modelo A")
        return {"status": "skipped", "model": "audio"}

    # Cargar label encoder existente
    le_path = AUDIO_MODEL_DIR / "label_encoder.json"
    if not le_path.exists():
        print("label_encoder.json no encontrado, no se puede reentrenar")
        return {"status": "skipped", "model": "audio", "reason": "no_label_encoder"}

    with open(le_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        clases = np.array(data["clases"])

    # Preparar datos
    X = np.array([r["embedding"] for r in recordings_with_embeddings])
    y_raw = [r["species"] for r in recordings_with_embeddings]

    # Filtrar solo especies que el modelo ya conoce
    mask = [sp in clases for sp in y_raw]
    X = X[mask]
    y_raw = [sp for sp, m in zip(y_raw, mask) if m]

    if len(X) < N_FOLDS:
        print(f"Datos insuficientes para {N_FOLDS} folds: {len(X)} muestras")
        return {"status": "skipped", "model": "audio", "reason": "insufficient_data"}

    # Codificar etiquetas con el encoder existente
    label_to_idx = {cls: i for i, cls in enumerate(clases)}
    y = np.array([label_to_idx[sp] for sp in y_raw])

    params = {
        "objective": "multiclass",
        "num_class": len(clases),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 5,
        "verbose": -1,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    nuevos_modelos = []
    val_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Cargar modelo existente del fold para entrenamiento incremental
        fold_path = AUDIO_MODEL_DIR / f"modelo_embeddings_fold{fold}.txt"
        init_model = str(fold_path) if fold_path.exists() else None

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=100,
            valid_sets=[dval],
            init_model=init_model,  # continúa desde el modelo existente
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
        )

        # Evaluar en validación
        preds = model.predict(X_val)
        top1_pred = np.argmax(preds, axis=1)
        acc = float(np.mean(top1_pred == y_val))
        val_accuracies.append(acc)

        # Guardar modelo actualizado
        new_path = AUDIO_MODEL_DIR / f"modelo_embeddings_fold{fold}_new.txt"
        model.save_model(str(new_path))
        nuevos_modelos.append(str(new_path))

        print(f"Fold {fold}/{N_FOLDS} — val accuracy: {acc:.4f}")

    avg_accuracy = float(np.mean(val_accuracies))
    print(f"Modelo A — accuracy promedio K-Folds: {avg_accuracy:.4f}")

    return {
        "status": "success",
        "model": "audio",
        "n_samples": len(X),
        "n_folds": N_FOLDS,
        "avg_accuracy": avg_accuracy,
        "fold_accuracies": val_accuracies,
        "new_model_paths": nuevos_modelos,
    }


# ══════════════════════════════════════════════════════════════════
# MODELO B — Geoespacial (LightGBM sobre features espaciotemporales)
# ══════════════════════════════════════════════════════════════════


@task(name="train-geo-model")
def train_geo_model(recordings_with_embeddings: list[dict]) -> dict:
    """
    Reentrena incrementalmente el Modelo B (LightGBM geoespacial).

    El GeoTemporalFeaturizer NO se reentrena, se reutiliza el existente
    para mantener consistencia en las transformaciones.

    Incremental: usa el model_b.txt existente como init_model.
    """
    if not recordings_with_embeddings:
        print("Sin datos para reentrenar Modelo B")
        return {"status": "skipped", "model": "geo"}

    # Cargar featurizer y label encoder existentes
    featurizer_path = GEO_MODEL_DIR / "featurizer.joblib"
    label_encoder_path = GEO_MODEL_DIR / "label_encoder.joblib"
    model_path = GEO_MODEL_DIR / "model_b.txt"

    if not featurizer_path.exists() or not label_encoder_path.exists():
        print("Artefactos del Modelo B no encontrados")
        return {"status": "skipped", "model": "geo", "reason": "no_artifacts"}

    featurizer = joblib.load(featurizer_path)
    label_encoder = joblib.load(label_encoder_path)

    # Filtrar grabaciones que tienen las variables geoespaciales necesarias
    valid = [
        r
        for r in recordings_with_embeddings
        if all(k in r for k in ["latitude", "longitude", "recorded_at", "elevation"])
    ]

    if len(valid) < 10:
        print(f"Datos geoespaciales insuficientes: {len(valid)} muestras")
        return {"status": "skipped", "model": "geo", "reason": "insufficient_data"}

    # Preparar DataFrame de entrada
    rows = []
    for r in valid:
        dt = r["recorded_at"]
        rows.append(
            {
                "lat": r["latitude"],
                "lon": r["longitude"],
                "month": dt.month,
                "day_of_year": dt.timetuple().tm_yday,
                "elevation": r.get("elevation"),
                "species": r["species"],
            }
        )

    df = pd.DataFrame(rows)

    # Filtrar solo especies que el encoder ya conoce
    clases_conocidas = set(label_encoder.classes_)
    df = df[df["species"].isin(clases_conocidas)].reset_index(drop=True)

    if len(df) < 10:
        print("Insuficientes muestras de especies conocidas para Modelo B")
        return {
            "status": "skipped",
            "model": "geo",
            "reason": "insufficient_known_species",
        }

    # Aplicar featurizer existente (sin reentrenarlo)
    X = featurizer.transform(df).astype("float32")
    y = label_encoder.transform(df["species"])

    # Split simple 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        "objective": "multiclass",
        "num_class": len(label_encoder.classes_),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 3,
        "verbose": -1,
    }

    # Entrenamiento incremental desde model_b.txt existente
    init_model = str(model_path) if model_path.exists() else None

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=100,
        valid_sets=[dval],
        init_model=init_model,
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
    )

    # Evaluar
    preds = model.predict(X_val)
    top1_pred = np.argmax(preds, axis=1)
    top1_acc = float(np.mean(top1_pred == y_val))

    top5_correct = sum(y_val[i] in np.argsort(preds[i])[-5:] for i in range(len(y_val)))
    top5_acc = top5_correct / len(y_val)

    # Guardar modelo actualizado
    new_path = GEO_MODEL_DIR / "model_b_new.txt"
    model.save_model(str(new_path))

    print(f"Modelo B — Top-1: {top1_acc:.4f}, Top-5: {top5_acc:.4f}")

    return {
        "status": "success",
        "model": "geo",
        "n_samples": len(df),
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "new_model_path": str(new_path),
    }
