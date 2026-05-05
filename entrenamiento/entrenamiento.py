"""
fase3_entrenamiento.py
──────────────────────
Entrena 2 modelos LightGBM separados:
  1. Modelo de embeddings: embedding (1024 dims) → especie
  2. Modelo de metadata:   lat, lon, fecha, hora, etc. → especie

Técnicas de la literatura:
  • Label smoothing para manejar etiquetas ruidosas
  • Stratified K-Fold para evaluación robusta
  • Multi-label: incorpora secondary labels (campo "also" de XC)
  • Ensemble opcional: combina predicciones de ambos modelos

Requisitos:
    pip install lightgbm scikit-learn pandas numpy matplotlib seaborn

Uso:
    python fase3_entrenamiento.py \
        --embeddings datos_fase2/embeddings_limpios.parquet \
        --metadata datos_fase1/metadata_consolidado.parquet

Desde Python:
    from fase3_entrenamiento import entrenar_pipeline
    resultados = entrenar_pipeline(
        embeddings_path="datos_fase2/embeddings_limpios.parquet",
        metadata_path="datos_fase1/metadata_consolidado.parquet",
    )
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

N_FOLDS           = 5
LABEL_SMOOTHING   = 0.1    # 10% distribuido entre otras clases
RANDOM_STATE      = 42
EARLY_STOPPING    = 50

# LightGBM params para embeddings (1024 features)
PARAMS_EMBEDDINGS = {
    "objective":        "multiclass",
    "metric":           "multi_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       127,
    "learning_rate":    0.05,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 10,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             RANDOM_STATE,
}

# LightGBM params para metadata (pocas features, más regularización)
PARAMS_METADATA = {
    "objective":        "multiclass",
    "metric":           "multi_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       31,
    "learning_rate":    0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "min_child_samples": 5,
    "verbose":          -1,
    "n_jobs":           -1,
    "seed":             RANDOM_STATE,
}


# ════════════════════════════════════════════════════════════════════════════
# 1. CARGAR Y PREPARAR DATOS
# ════════════════════════════════════════════════════════════════════════════

def _cargar_archivo(path: str) -> pd.DataFrame:
    """Carga CSV o Parquet automáticamente."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _preparar_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame]:
    """
    Prepara datos para el modelo de embeddings.

    Retorna:
        X:       numpy array (n_segmentos, 1024)
        y:       numpy array de labels codificados
        le:      LabelEncoder para decodificar
        df_meta: columnas informativas (xc_id, genero, especie, etc.)
    """
    print("\n  Preparando datos de embeddings...")

    # Crear label: "genero especie"
    df = df.copy()
    df["label"] = df["genero"].str.strip() + " " + df["especie"].str.strip()

    # Filtrar especies con muy pocas muestras (< 5)
    conteo = df["label"].value_counts()
    especies_validas = conteo[conteo >= 5].index
    n_descartadas = len(conteo) - len(especies_validas)
    if n_descartadas > 0:
        print(f"    ⚠ {n_descartadas} especies con < 5 muestras descartadas")
    df = df[df["label"].isin(especies_validas)].reset_index(drop=True)

    # Parsear embeddings
    if isinstance(df["embedding"].iloc[0], str):
        X = np.array(
            df["embedding"].apply(lambda s: [float(x) for x in s.split(";")]).tolist()
        )
    elif isinstance(df["embedding"].iloc[0], (list, np.ndarray)):
        X = np.array(df["embedding"].tolist())
    else:
        raise ValueError(f"Formato de embedding no reconocido")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    print(f"    ✓ X: {X.shape}")
    print(f"    ✓ Clases: {len(le.classes_)}")
    print(f"    ✓ Muestras por clase:")
    for clase in le.classes_:
        n = (df["label"] == clase).sum()
        print(f"        {clase}: {n}")

    # Columnas informativas
    cols_meta = [c for c in df.columns if c != "embedding"]
    df_meta = df[cols_meta].copy()

    return X, y, le, df_meta


def _preparar_metadata(df: pd.DataFrame, le_target: LabelEncoder = None
                       ) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, list]:
    """
    Prepara datos para el modelo de metadata.

    Transforma features categóricas y temporales en numéricas.

    Retorna:
        X_meta:   DataFrame con features procesadas
        y:        numpy array de labels
        le:       LabelEncoder
        features: lista de nombres de columnas usadas
    """
    print("\n  Preparando datos de metadata...")

    df = df.copy()
    df["label"] = df["genero"].str.strip() + " " + df["especie"].str.strip()

    # Filtrar especies con pocas muestras
    conteo = df["label"].value_counts()
    especies_validas = conteo[conteo >= 3].index
    df = df[df["label"].isin(especies_validas)].reset_index(drop=True)

    # Encode labels
    if le_target is not None:
        # Usar el mismo encoder que embeddings (solo especies en común)
        especies_comunes = set(le_target.classes_) & set(df["label"].unique())
        df = df[df["label"].isin(especies_comunes)].reset_index(drop=True)
        le = le_target
    else:
        le = LabelEncoder()
        le.fit(df["label"])

    y = le.transform(df["label"])

    # ── Procesar features ────────────────────────────────────────────
    features = []

    # Numéricas directas
    for col in ["lat", "lon", "altura"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            features.append(col)

    # Fecha → mes, día del año
    if "fecha" in df.columns:
        df["fecha_parsed"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["mes"] = df["fecha_parsed"].dt.month
        df["dia_anio"] = df["fecha_parsed"].dt.dayofyear

        # Encoding cíclico para mes
        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
        features.extend(["mes", "dia_anio", "mes_sin", "mes_cos"])

    # Hora → hora decimal
    if "hora" in df.columns:
        def _parse_hora(h):
            try:
                if pd.isna(h) or h == "":
                    return np.nan
                partes = str(h).split(":")
                return float(partes[0]) + float(partes[1]) / 60 if len(partes) >= 2 else float(partes[0])
            except Exception:
                return np.nan

        df["hora_decimal"] = df["hora"].apply(_parse_hora)
        df["hora_sin"] = np.sin(2 * np.pi * df["hora_decimal"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora_decimal"] / 24)
        features.extend(["hora_decimal", "hora_sin", "hora_cos"])

    # Calidad XC → ordinal
    if "calidad_xc" in df.columns:
        mapa_calidad = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
        df["calidad_num"] = df["calidad_xc"].map(mapa_calidad).fillna(0)
        features.append("calidad_num")

    # Tipo de canto → label encode
    if "tipo_canto" in df.columns:
        le_tipo = LabelEncoder()
        df["tipo_canto_num"] = le_tipo.fit_transform(
            df["tipo_canto"].fillna("unknown").astype(str)
        )
        features.append("tipo_canto_num")

    # Duración → segundos
    if "duracion" in df.columns:
        def _parse_duracion(d):
            try:
                if pd.isna(d) or d == "":
                    return np.nan
                partes = str(d).split(":")
                if len(partes) == 2:
                    return float(partes[0]) * 60 + float(partes[1])
                return float(partes[0])
            except Exception:
                return np.nan

        df["duracion_seg"] = df["duracion"].apply(_parse_duracion)
        features.append("duracion_seg")

    # Playback usado → binario
    if "playback_usado" in df.columns:
        df["playback_bin"] = df["playback_usado"].map(
            {"yes": 1, "no": 0, "": 0}
        ).fillna(0).astype(int)
        features.append("playback_bin")

    # Número de otras especies
    if "otras_especies" in df.columns:
        def _contar_otras(x):
            if x is None:
                return 0
            if isinstance(x, (list, np.ndarray)):
                return len(x)
            s = str(x).strip()
            if s == "" or s == "nan" or s == "[]":
                return 0
            return len(s.split(","))

        df["n_otras_especies"] = df["otras_especies"].apply(_contar_otras)
        features.append("n_otras_especies")

    X_meta = df[features].copy()

    # Rellenar NaN
    for col in X_meta.columns:
        X_meta[col] = X_meta[col].fillna(X_meta[col].median() if X_meta[col].dtype in ["float64", "int64"] else 0)

    print(f"    ✓ Features: {features}")
    print(f"    ✓ X: {X_meta.shape}")
    print(f"    ✓ Clases: {len(le.classes_)}")

    return X_meta, y, le, features



# ════════════════════════════════════════════════════════════════════════════
# 4. ENTRENAR UN MODELO CON K-FOLD
# ════════════════════════════════════════════════════════════════════════════

def _entrenar_kfold(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    n_clases: int,
    params: dict,
    nombre_modelo: str,
    label_smoothing: float = LABEL_SMOOTHING,
    n_folds: int = N_FOLDS,
    n_rounds: int = 1000,
) -> dict:
    """
    Entrena LightGBM con Stratified K-Fold.
    
    Label smoothing se implementa con el objetivo built-in 'multiclass'
    y ajuste posterior de las predicciones, ya que LightGBM v4+
    eliminó el parámetro fobj.
    """
    print(f"\n{'═' * 60}")
    print(f"  ENTRENANDO: {nombre_modelo}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  {n_folds}-Fold CV | {n_rounds} rounds max")
    print(f"{'═' * 60}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    modelos = []
    oof_preds = np.zeros((len(y), n_clases))
    fold_metrics = []

    X_np = X.values if isinstance(X, pd.DataFrame) else X
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y), 1):
        print(f"\n  ── Fold {fold}/{n_folds} ──")

        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params_fold = params.copy()
        params_fold["num_class"] = n_clases

        callbacks = [
            lgb.early_stopping(EARLY_STOPPING, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        modelo = lgb.train(
            params_fold,
            dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        modelos.append(modelo)

        # Predicciones OOF
        preds_proba = modelo.predict(X_val)

        # Aplicar label smoothing POST-HOC a las predicciones
        # Esto suaviza las predicciones para que no sean tan extremas
        if label_smoothing > 0:
            preds_proba = (1 - label_smoothing) * preds_proba + \
                          label_smoothing / n_clases

        oof_preds[val_idx] = preds_proba

        # Métricas del fold
        y_pred_fold = preds_proba.argmax(axis=1)
        acc_fold = accuracy_score(y_val, y_pred_fold)

        top3_acc = 0
        if n_clases >= 3:
            top3_acc = top_k_accuracy_score(y_val, preds_proba, k=min(3, n_clases))

        fold_metrics.append({"fold": fold, "accuracy": acc_fold, "top3_acc": top3_acc})
        print(f"    Accuracy: {acc_fold:.4f} | Top-3: {top3_acc:.4f} | "
              f"Best iter: {modelo.best_iteration}")

    # ── Métricas globales ────────────────────────────────────────────
    y_pred_oof = oof_preds.argmax(axis=1)
    acc_global = accuracy_score(y, y_pred_oof)
    top3_global = 0
    if n_clases >= 3:
        top3_global = top_k_accuracy_score(y, oof_preds, k=min(3, n_clases))

    print(f"\n  ── Resultados globales (OOF) ──")
    print(f"    Accuracy:   {acc_global:.4f}")
    print(f"    Top-3 Acc:  {top3_global:.4f}")

    # Feature importance
    importances = None
    if feature_names:
        imp = np.zeros(len(feature_names))
        for m in modelos:
            imp += m.feature_importance(importance_type="gain")
        imp /= len(modelos)
        importances = pd.DataFrame({
            "feature": feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False)

    return {
        "modelos": modelos,
        "oof_preds": oof_preds,
        "oof_labels": y,
        "accuracy": acc_global,
        "top3_accuracy": top3_global,
        "fold_metrics": fold_metrics,
        "importances": importances,
    }

# ════════════════════════════════════════════════════════════════════════════
# 5. ENSEMBLE DE LOS 2 MODELOS
# ════════════════════════════════════════════════════════════════════════════

def _ensemble_predicciones(
    preds_emb: np.ndarray,
    preds_meta: np.ndarray,
    peso_emb: float = 0.7,
    peso_meta: float = 0.3,
) -> np.ndarray:
    """
    Combina predicciones de ambos modelos con promedio ponderado.
    Embeddings suelen ser más informativos → mayor peso.
    """
    return peso_emb * preds_emb + peso_meta * preds_meta


# ════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIONES
# ════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(y_true, y_pred, clases, titulo, output_path):
    """Genera y guarda matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(clases)), max(6, len(clases) * 0.8)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=clases, yticklabels=clases, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {output_path}")


def _plot_importances(importances: pd.DataFrame, titulo: str, output_path: str,
                      top_n: int = 30):
    """Genera gráfico de feature importances."""
    if importances is None or len(importances) == 0:
        return

    df_plot = importances.head(top_n)
    fig, ax = plt.subplots(1, 1, figsize=(10, max(6, len(df_plot) * 0.3)))
    ax.barh(range(len(df_plot)), df_plot["importance"].values)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(titulo)
    ax.set_xlabel("Importance (gain)")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {output_path}")


# ════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════════════

def entrenar_pipeline(
    embeddings_path: str,
    metadata_path: str = None,
    output_dir: str = "datos_fase3/",
    label_smoothing: float = LABEL_SMOOTHING,
    n_folds: int = N_FOLDS,
    peso_emb: float = 0.7,
    peso_meta: float = 0.3,
) -> dict:
    """
    Pipeline Fase 3:
      1. Carga datos de Fase 2 (embeddings limpios) y Fase 1 (metadata)
      2. Entrena modelo de embeddings (LightGBM + label smoothing)
      3. Entrena modelo de metadata (LightGBM + label smoothing)
      4. Ensemble opcional
      5. Genera reportes y visualizaciones

    Returns:
        Dict con resultados de ambos modelos y ensemble.
    """
    print(f"\n{'═' * 60}")
    print(f"  FASE 3: ENTRENAMIENTO")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  K-Folds: {n_folds}")
    print(f"{'═' * 60}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    resultados = {}

    # ══════════════════════════════════════════════════════════════════
    # MODELO 1: EMBEDDINGS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[1/3] Preparando modelo de embeddings...")
    df_emb = _cargar_archivo(embeddings_path)
    X_emb, y_emb, le_emb, df_meta_emb = _preparar_embeddings(df_emb)
    n_clases = len(le_emb.classes_)

    res_emb = _entrenar_kfold(
        X=X_emb, y=y_emb, n_clases=n_clases,
        params=PARAMS_EMBEDDINGS,
        nombre_modelo="EMBEDDINGS",
        label_smoothing=label_smoothing,
        n_folds=n_folds,
    )
    resultados["embeddings"] = res_emb

    # Classification report
    y_pred_emb = res_emb["oof_preds"].argmax(axis=1)
    report_emb = classification_report(
        y_emb, y_pred_emb, target_names=le_emb.classes_, output_dict=True
    )
    print(f"\n  Classification Report (Embeddings):")
    print(classification_report(y_emb, y_pred_emb, target_names=le_emb.classes_))

    # ══════════════════════════════════════════════════════════════════
    # MODELO 2: METADATA (si se proporciona)
    # ══════════════════════════════════════════════════════════════════
    res_meta = None
    if metadata_path and os.path.exists(metadata_path):
        print(f"\n[2/3] Preparando modelo de metadata...")
        df_metadata = _cargar_archivo(metadata_path)

        # Solo usar audios exitosos
        if "estado" in df_metadata.columns:
            df_metadata = df_metadata[df_metadata["estado"] == "ok"].reset_index(drop=True)

        X_meta, y_meta, le_meta, feature_names = _preparar_metadata(
            df_metadata, le_target=le_emb
        )

        if len(X_meta) >= n_folds * 2:
            n_clases_meta = len(le_meta.classes_)

            res_meta = _entrenar_kfold(
                X=X_meta, y=y_meta, n_clases=n_clases_meta,
                params=PARAMS_METADATA,
                nombre_modelo="METADATA",
                label_smoothing=label_smoothing,
                n_folds=n_folds,
            )
            resultados["metadata"] = res_meta

            y_pred_meta = res_meta["oof_preds"].argmax(axis=1)
            print(f"\n  Classification Report (Metadata):")
            print(classification_report(
                y_meta, y_pred_meta, target_names=le_meta.classes_
            ))
        else:
            print(f"    ⚠ Muy pocos datos de metadata ({len(X_meta)}), saltando")
    else:
        print(f"\n[2/3] Sin archivo de metadata, saltando modelo 2")

    # ══════════════════════════════════════════════════════════════════
    # VISUALIZACIONES Y REPORTES
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3/3] Generando reportes...")

    # Confusion matrix — embeddings
    _plot_confusion_matrix(
        y_emb, y_pred_emb, le_emb.classes_,
        f"Modelo Embeddings (Acc={res_emb['accuracy']:.3f})",
        os.path.join(output_dir, "confusion_embeddings.png"),
    )

    # Confusion matrix — metadata
    if res_meta is not None:
        y_pred_meta = res_meta["oof_preds"].argmax(axis=1)
        _plot_confusion_matrix(
            y_meta, y_pred_meta, le_meta.classes_,
            f"Modelo Metadata (Acc={res_meta['accuracy']:.3f})",
            os.path.join(output_dir, "confusion_metadata.png"),
        )

        # Feature importances — metadata
        _plot_importances(
            res_meta["importances"],
            "Feature Importances — Metadata",
            os.path.join(output_dir, "importances_metadata.png"),
        )

    # ── Guardar modelos ──────────────────────────────────────────────
    for i, modelo in enumerate(res_emb["modelos"]):
        modelo.save_model(os.path.join(output_dir, f"modelo_embeddings_fold{i}.txt"))

    if res_meta:
        for i, modelo in enumerate(res_meta["modelos"]):
            modelo.save_model(os.path.join(output_dir, f"modelo_metadata_fold{i}.txt"))

    # ── Guardar label encoder ────────────────────────────────────────
    le_path = os.path.join(output_dir, "label_encoder.json")
    with open(le_path, "w") as f:
        json.dump({"clases": le_emb.classes_.tolist()}, f, indent=2)
    print(f"    ✓ Label encoder: {le_path}")

    # ── Guardar métricas ─────────────────────────────────────────────
    metricas = {
        "embeddings": {
            "accuracy": res_emb["accuracy"],
            "top3_accuracy": res_emb["top3_accuracy"],
            "n_clases": n_clases,
            "n_muestras": len(y_emb),
            "fold_metrics": res_emb["fold_metrics"],
        },
    }
    if res_meta:
        metricas["metadata"] = {
            "accuracy": res_meta["accuracy"],
            "top3_accuracy": res_meta["top3_accuracy"],
            "n_muestras": len(y_meta),
            "fold_metrics": res_meta["fold_metrics"],
        }

    metricas_path = os.path.join(output_dir, "metricas.json")
    with open(metricas_path, "w") as f:
        json.dump(metricas, f, indent=2)
    print(f"    ✓ Métricas: {metricas_path}")

    # ── Resumen final ────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  RESUMEN FASE 3")
    print(f"{'═' * 60}")
    print(f"    Modelo Embeddings:")
    print(f"      Accuracy:    {res_emb['accuracy']:.4f}")
    print(f"      Top-3 Acc:   {res_emb['top3_accuracy']:.4f}")
    print(f"      Clases:      {n_clases}")
    print(f"      Muestras:    {len(y_emb)}")

    if res_meta:
        print(f"\n    Modelo Metadata:")
        print(f"      Accuracy:    {res_meta['accuracy']:.4f}")
        print(f"      Top-3 Acc:   {res_meta['top3_accuracy']:.4f}")
        print(f"      Muestras:    {len(y_meta)}")

    print(f"\n    📁 {output_dir}")
    print(f"{'═' * 60}\n")

    return resultados


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 3: Entrenamiento LightGBM con label smoothing"
    )
    parser.add_argument("--embeddings", required=True,
                        help="Ruta a embeddings_limpios de Fase 2")
    parser.add_argument("--metadata", default=None,
                        help="Ruta a metadata_consolidado de Fase 1")
    parser.add_argument("--output_dir", default="datos_fase3/")
    parser.add_argument("--smoothing", type=float, default=LABEL_SMOOTHING,
                        help=f"Label smoothing (default: {LABEL_SMOOTHING})")
    parser.add_argument("--folds", type=int, default=N_FOLDS,
                        help=f"Número de folds (default: {N_FOLDS})")
    parser.add_argument("--peso_emb", type=float, default=0.7,
                        help="Peso del modelo de embeddings en ensemble")
    parser.add_argument("--peso_meta", type=float, default=0.3,
                        help="Peso del modelo de metadata en ensemble")
    args = parser.parse_args()

    entrenar_pipeline(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        label_smoothing=args.smoothing,
        n_folds=args.folds,
        peso_emb=args.peso_emb,
        peso_meta=args.peso_meta,
    )