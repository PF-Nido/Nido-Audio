"""
fase3_entrenamiento.py
──────────────────────
Entrena modelos LightGBM para reconocimiento de canto de aves:

  1. Modelo de embeddings:
       embedding 1024 dims → especie

  2. Modelo de metadata:
       lat, lon, fecha, hora, calidad, etc. → especie

Incluye:
  • LightGBM CPU/GPU
  • Selección explícita de GPU OpenCL:
        --gpu_platform_id 0 --gpu_device_id 0
  • Autodetección de GPU NVIDIA con pyopencl
  • StratifiedGroupKFold por xc_id para evitar fuga entre segmentos
  • top_k_accuracy_score robusto
  • Manejo robusto de NaN en metadata
  • Guardado de modelos, métricas, classification reports y gráficas

Requisitos:
    pip install lightgbm scikit-learn pandas numpy matplotlib seaborn pyarrow pyopencl

Ejemplo Windows + NVIDIA:
    python fase3_entrenamiento.py ^
      --embeddings datos_fase2/embeddings_limpios.parquet ^
      --metadata datos_fase1/metadata_consolidado.parquet ^
      --output_dir datos_fase3_gpu/ ^
      --folds 5 ^
      --rounds 300 ^
      --device gpu ^
      --gpu_platform_id 0 ^
      --gpu_device_id 0
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

N_FOLDS = 5
LABEL_SMOOTHING = 0.1
RANDOM_STATE = 42
EARLY_STOPPING = 50

MAX_ANNOTATED_CM_CLASSES = 80


PARAMS_EMBEDDINGS = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",

    # Modelo
    "num_leaves": 127,
    "learning_rate": 0.05,

    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,

    "min_child_samples": 10,

    # Regularización
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,

    "verbose": 1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
}


PARAMS_METADATA = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",

    # Metadata tiene pocas features
    "num_leaves": 31,
    "learning_rate": 0.05,

    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,

    "min_child_samples": 5,

    # Regularización
    "lambda_l1": 0.0,
    "lambda_l2": 2.0,

    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
}


# ════════════════════════════════════════════════════════════════════════════
# GPU / OPENCL
# ════════════════════════════════════════════════════════════════════════════

def _listar_opencl_devices():
    """
    Lista plataformas y dispositivos OpenCL disponibles.

    Requiere:
        pip install pyopencl

    Retorna lista de dicts con:
        platform_id, device_id, platform_name, platform_vendor,
        device_name, device_vendor, is_gpu
    """
    try:
        import pyopencl as cl
    except ImportError:
        print("    ⚠ pyopencl no está instalado. No se puede autodetectar GPU.")
        print("      Instala con: pip install pyopencl")
        return []

    dispositivos = []

    try:
        platforms = cl.get_platforms()
    except Exception as e:
        print(f"    ⚠ No se pudieron listar plataformas OpenCL: {e}")
        return []

    print("\n  Dispositivos OpenCL detectados:")

    for platform_id, platform in enumerate(platforms):
        print(f"\n    Platform ID {platform_id}")
        print(f"      Name:   {platform.name}")
        print(f"      Vendor: {platform.vendor}")

        try:
            devices = platform.get_devices()
        except Exception as e:
            print(f"      ⚠ No se pudieron listar devices: {e}")
            continue

        for device_id, device in enumerate(devices):
            is_gpu = bool(device.type & cl.device_type.GPU)
            is_cpu = bool(device.type & cl.device_type.CPU)

            tipo = []
            if is_gpu:
                tipo.append("GPU")
            if is_cpu:
                tipo.append("CPU")

            tipo_str = "/".join(tipo) if tipo else str(device.type)

            print(f"\n      Device ID {device_id}")
            print(f"        Name:              {device.name}")
            print(f"        Vendor:            {device.vendor}")
            print(f"        Type:              {tipo_str}")
            print(f"        Global memory GB:  {device.global_mem_size / 1024**3:.2f}")
            print(f"        Max compute units: {device.max_compute_units}")

            dispositivos.append(
                {
                    "platform_id": platform_id,
                    "device_id": device_id,
                    "platform_name": platform.name,
                    "platform_vendor": platform.vendor,
                    "device_name": device.name,
                    "device_vendor": device.vendor,
                    "is_gpu": is_gpu,
                }
            )

    print("")
    return dispositivos


def _seleccionar_opencl_gpu(
    prefer_vendor: str = "NVIDIA",
    gpu_platform_id: int | None = None,
    gpu_device_id: int | None = None,
):
    """
    Selecciona GPU OpenCL para LightGBM.

    Si gpu_platform_id y gpu_device_id se pasan manualmente, los usa.

    Si no, intenta encontrar una GPU cuyo vendor/name contenga prefer_vendor.
    """
    if gpu_platform_id is not None and gpu_device_id is not None:
        print("\n  GPU OpenCL seleccionada manualmente:")
        print(f"    gpu_platform_id: {gpu_platform_id}")
        print(f"    gpu_device_id:   {gpu_device_id}")
        return int(gpu_platform_id), int(gpu_device_id)

    dispositivos = _listar_opencl_devices()

    if len(dispositivos) == 0:
        print("    ⚠ No se detectaron dispositivos OpenCL mediante pyopencl.")
        print("      LightGBM usará el GPU default del sistema.")
        return None, None

    prefer_vendor_l = prefer_vendor.lower().strip()

    # 1. Buscar GPU preferida, por ejemplo NVIDIA
    for d in dispositivos:
        texto = (
            d["platform_name"] + " "
            + d["platform_vendor"] + " "
            + d["device_name"] + " "
            + d["device_vendor"]
        ).lower()

        if d["is_gpu"] and prefer_vendor_l in texto:
            print("\n  GPU OpenCL seleccionada automáticamente:")
            print(f"    Prefer vendor:   {prefer_vendor}")
            print(f"    gpu_platform_id: {d['platform_id']}")
            print(f"    gpu_device_id:   {d['device_id']}")
            print(f"    Device name:     {d['device_name']}")
            print(f"    Device vendor:   {d['device_vendor']}")
            return int(d["platform_id"]), int(d["device_id"])

    # 2. Si no encontró la preferida, usar primera GPU
    for d in dispositivos:
        if d["is_gpu"]:
            print("\n  ⚠ No se encontró GPU del vendor preferido.")
            print("    Se usará la primera GPU OpenCL disponible:")
            print(f"    gpu_platform_id: {d['platform_id']}")
            print(f"    gpu_device_id:   {d['device_id']}")
            print(f"    Device name:     {d['device_name']}")
            print(f"    Device vendor:   {d['device_vendor']}")
            return int(d["platform_id"]), int(d["device_id"])

    print("    ⚠ No se encontró ninguna GPU OpenCL.")
    print("      LightGBM usará el GPU default del sistema.")
    return None, None


def _aplicar_device_lightgbm(
    params: dict,
    device: str = "cpu",
    gpu_platform_id: int | None = None,
    gpu_device_id: int | None = None,
    prefer_gpu_vendor: str = "NVIDIA",
) -> dict:
    """
    Aplica configuración CPU/GPU/CUDA a LightGBM.

    Para Windows + NVIDIA:
        device='gpu'
        gpu_platform_id=0
        gpu_device_id=0

    Tu caso detectado:
        NVIDIA GeForce RTX 3050 Laptop GPU
        gpu_platform_id=0
        gpu_device_id=0
    """
    params = params.copy()

    device = device.lower().strip()

    if device not in ["cpu", "gpu", "cuda"]:
        raise ValueError("device debe ser 'cpu', 'gpu' o 'cuda'")

    params["device_type"] = device

    if device == "gpu":
        # OpenCL GPU
        params["max_bin"] = 63
        params["gpu_use_dp"] = False

        selected_platform_id, selected_device_id = _seleccionar_opencl_gpu(
            prefer_vendor=prefer_gpu_vendor,
            gpu_platform_id=gpu_platform_id,
            gpu_device_id=gpu_device_id,
        )

        if selected_platform_id is not None and selected_device_id is not None:
            params["gpu_platform_id"] = int(selected_platform_id)
            params["gpu_device_id"] = int(selected_device_id)

    elif device == "cuda":
        # Solo si tu build de LightGBM fue compilado con soporte CUDA.
        # En Windows normalmente debes usar device='gpu', no 'cuda'.
        params["max_bin"] = 63
        params.pop("gpu_use_dp", None)

        if gpu_device_id is not None:
            params["gpu_device_id"] = int(gpu_device_id)

    else:
        # CPU
        params.pop("gpu_use_dp", None)
        params.pop("gpu_platform_id", None)
        params.pop("gpu_device_id", None)
        params.pop("max_bin", None)

    return params


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ════════════════════════════════════════════════════════════════════════════

def _cargar_archivo(path: str) -> pd.DataFrame:
    """Carga CSV o Parquet automáticamente."""
    if path is None:
        raise ValueError("path no puede ser None")

    path = str(path)

    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    if path.endswith(".csv"):
        return pd.read_csv(path)

    raise ValueError(f"Formato no soportado: {path}. Usa .parquet o .csv")


def _crear_label(df: pd.DataFrame) -> pd.Series:
    """Crea label 'genero especie' de forma robusta."""
    if "genero" not in df.columns or "especie" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'genero' y 'especie'.")

    genero = df["genero"].fillna("").astype(str).str.strip()
    especie = df["especie"].fillna("").astype(str).str.strip()

    label = genero + " " + especie
    label = label.str.strip()

    return label


def _parse_embedding_value(x):
    """
    Convierte un embedding a lista de floats.

    Soporta:
      • list
      • np.ndarray
      • string separado por ;
      • string JSON tipo "[0.1, 0.2, ...]"
    """
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()

    if isinstance(x, list):
        return [float(v) for v in x]

    if isinstance(x, str):
        s = x.strip()

        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                return [float(v) for v in arr]
            except Exception:
                pass

        if ";" in s:
            return [float(v) for v in s.split(";") if v != ""]

        if "," in s:
            s2 = s.strip("[]")
            return [float(v) for v in s2.split(",") if v.strip() != ""]

    raise ValueError(f"Formato de embedding no reconocido: {type(x)}")


def _filtrar_por_minimo(
    df: pd.DataFrame,
    label_col: str,
    min_muestras: int,
    group_col: str | None = None,
    nombre: str = "datos",
) -> pd.DataFrame:
    """
    Filtra clases con pocas muestras o pocos grupos.

    Si group_col existe, filtra por número de grupos únicos por clase.
    Esto es importante cuando se usa StratifiedGroupKFold.
    """
    df = df.copy()

    if group_col is not None and group_col in df.columns:
        conteo = df.groupby(label_col)[group_col].nunique()
        criterio = "grupos únicos"
    else:
        conteo = df[label_col].value_counts()
        criterio = "muestras"

    clases_validas = conteo[conteo >= min_muestras].index
    n_descartadas = len(conteo) - len(clases_validas)

    if n_descartadas > 0:
        print(
            f"    ⚠ {n_descartadas} clases descartadas en {nombre} "
            f"por tener < {min_muestras} {criterio}"
        )

    df = df[df[label_col].isin(clases_validas)].reset_index(drop=True)

    return df


def _rellenar_nan_robusto(X: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena NaN de forma robusta.

    Si una columna numérica está completamente vacía, su mediana es NaN.
    En ese caso se rellena con 0.
    """
    X = X.copy()

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            mediana = X[col].median()

            if pd.isna(mediana):
                mediana = 0

            X[col] = X[col].fillna(mediana)
        else:
            X[col] = X[col].fillna(0)

    X = X.replace([np.inf, -np.inf], 0)

    return X


def _safe_top_k_accuracy(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_clases: int,
    k: int = 3,
) -> float:
    """
    top_k_accuracy_score robusto para folds donde no aparecen todas las clases.
    """
    if n_clases < 2:
        return 0.0

    k = min(k, n_clases)

    return top_k_accuracy_score(
        y_true,
        y_score,
        k=k,
        labels=np.arange(n_clases),
    )


def _safe_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clases: np.ndarray,
    titulo: str,
) -> dict:
    """
    Classification report robusto con labels explícitos.
    """
    labels = np.arange(len(clases))

    print(f"\n  Classification Report ({titulo}):")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=clases,
            zero_division=0,
        )
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=clases,
        output_dict=True,
        zero_division=0,
    )

    return report


# ════════════════════════════════════════════════════════════════════════════
# PREPARAR EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

def _preparar_embeddings(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    usar_group_kfold: bool = True,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame, np.ndarray | None]:
    """
    Prepara datos para modelo de embeddings.

    Retorna:
        X:       array, shape (n_segmentos, n_dims)
        y:       labels codificados
        le:      LabelEncoder
        df_meta: DataFrame sin columna embedding
        groups:  xc_id si existe y se usará StratifiedGroupKFold
    """
    print("\n  Preparando datos de embeddings...")

    df = df.copy()
    df["label"] = _crear_label(df)
    df = df[df["label"] != ""].reset_index(drop=True)

    group_col = None

    if usar_group_kfold and "xc_id" in df.columns:
        group_col = "xc_id"
        print("    ✓ Se usará xc_id para StratifiedGroupKFold en embeddings")
    else:
        print("    ⚠ No se encontró xc_id o group_kfold desactivado; se usará StratifiedKFold")

    df = _filtrar_por_minimo(
        df=df,
        label_col="label",
        min_muestras=n_folds,
        group_col=group_col,
        nombre="embeddings",
    )

    if len(df) == 0:
        raise ValueError("No quedan datos de embeddings después del filtrado.")

    if "embedding" not in df.columns:
        raise ValueError("El DataFrame de embeddings debe tener columna 'embedding'.")

    embeddings_list = df["embedding"].apply(_parse_embedding_value).tolist()
    X = np.asarray(embeddings_list, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X debería ser 2D, pero tiene shape {X.shape}")

    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    groups = None

    if group_col is not None:
        groups = df[group_col].astype(str).values

    print(f"    ✓ X: {X.shape}")
    print(f"    ✓ Clases: {len(le.classes_)}")
    print(f"    ✓ Muestras: {len(y)}")

    if groups is not None:
        print(f"    ✓ Grupos únicos: {len(np.unique(groups))}")

    print("    ✓ Muestras por clase:")
    conteo = df["label"].value_counts().sort_index()

    for clase, n in conteo.items():
        if group_col is not None:
            ng = df.loc[df["label"] == clase, group_col].nunique()
            print(f"        {clase}: {n} segmentos | {ng} audios")
        else:
            print(f"        {clase}: {n}")

    cols_meta = [c for c in df.columns if c != "embedding"]
    df_meta = df[cols_meta].copy()

    return X, y, le, df_meta, groups


# ════════════════════════════════════════════════════════════════════════════
# PREPARAR METADATA
# ════════════════════════════════════════════════════════════════════════════

def _preparar_metadata(
    df: pd.DataFrame,
    le_target: LabelEncoder | None = None,
    min_muestras: int = N_FOLDS,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder, list[str]]:
    """
    Prepara datos para modelo de metadata.

    Si le_target se pasa, usa exactamente el mismo LabelEncoder que embeddings.
    Esto permite que ambos modelos tengan el mismo espacio de clases.
    """
    print("\n  Preparando datos de metadata...")

    df = df.copy()
    df["label"] = _crear_label(df)
    df = df[df["label"] != ""].reset_index(drop=True)

    if le_target is not None:
        clases_target = set(le_target.classes_)
        antes = len(df)
        df = df[df["label"].isin(clases_target)].reset_index(drop=True)
        despues = len(df)

        print(f"    ✓ Metadata filtrada a clases existentes en embeddings: {antes} → {despues}")

    df = _filtrar_por_minimo(
        df=df,
        label_col="label",
        min_muestras=min_muestras,
        group_col=None,
        nombre="metadata",
    )

    if len(df) == 0:
        raise ValueError("No quedan datos de metadata después del filtrado.")

    if le_target is not None:
        le = le_target
    else:
        le = LabelEncoder()
        le.fit(df["label"])

    y = le.transform(df["label"])

    features = []

    # Numéricas directas
    for col in ["lat", "lon", "altura"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            features.append(col)

    # Fecha
    if "fecha" in df.columns:
        df["fecha_parsed"] = pd.to_datetime(df["fecha"], errors="coerce")

        df["mes"] = df["fecha_parsed"].dt.month
        df["dia_anio"] = df["fecha_parsed"].dt.dayofyear

        df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
        df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

        features.extend(["mes", "dia_anio", "mes_sin", "mes_cos"])

    # Hora
    if "hora" in df.columns:
        def _parse_hora(h):
            try:
                if pd.isna(h) or str(h).strip() == "":
                    return np.nan

                partes = str(h).strip().split(":")

                if len(partes) >= 2:
                    return float(partes[0]) + float(partes[1]) / 60.0

                return float(partes[0])
            except Exception:
                return np.nan

        df["hora_decimal"] = df["hora"].apply(_parse_hora)
        df["hora_sin"] = np.sin(2 * np.pi * df["hora_decimal"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora_decimal"] / 24)

        features.extend(["hora_decimal", "hora_sin", "hora_cos"])

    # Calidad XC
    if "calidad_xc" in df.columns:
        mapa_calidad = {
            "A": 5,
            "B": 4,
            "C": 3,
            "D": 2,
            "E": 1,
        }

        df["calidad_num"] = (
            df["calidad_xc"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .map(mapa_calidad)
            .fillna(0)
        )

        features.append("calidad_num")

    # Tipo de canto
    if "tipo_canto" in df.columns:
        le_tipo = LabelEncoder()

        df["tipo_canto_num"] = le_tipo.fit_transform(
            df["tipo_canto"]
            .fillna("unknown")
            .astype(str)
            .str.strip()
        )

        features.append("tipo_canto_num")

    # Duración
    if "duracion" in df.columns:
        def _parse_duracion(d):
            try:
                if pd.isna(d) or str(d).strip() == "":
                    return np.nan

                s = str(d).strip()
                partes = s.split(":")

                # mm:ss
                if len(partes) == 2:
                    return float(partes[0]) * 60.0 + float(partes[1])

                # hh:mm:ss
                if len(partes) == 3:
                    return (
                        float(partes[0]) * 3600.0
                        + float(partes[1]) * 60.0
                        + float(partes[2])
                    )

                return float(s)
            except Exception:
                return np.nan

        df["duracion_seg"] = df["duracion"].apply(_parse_duracion)
        features.append("duracion_seg")

    # Playback
    if "playback_usado" in df.columns:
        playback = (
            df["playback_usado"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        df["playback_bin"] = playback.map(
            {
                "yes": 1,
                "y": 1,
                "true": 1,
                "1": 1,
                "si": 1,
                "sí": 1,
                "no": 0,
                "n": 0,
                "false": 0,
                "0": 0,
                "": 0,
            }
        ).fillna(0).astype(int)

        features.append("playback_bin")

    # Otras especies
    if "otras_especies" in df.columns:
        def _contar_otras(x):
            if x is None:
                return 0

            if isinstance(x, (list, tuple, np.ndarray)):
                return len(x)

            s = str(x).strip()

            if s == "" or s.lower() == "nan" or s == "[]":
                return 0

            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        return len(arr)
                except Exception:
                    pass

            return len([p for p in s.split(",") if p.strip() != ""])

        df["n_otras_especies"] = df["otras_especies"].apply(_contar_otras)
        features.append("n_otras_especies")

    if len(features) == 0:
        raise ValueError("No se encontró ninguna feature válida de metadata.")

    X_meta = df[features].copy()
    X_meta = _rellenar_nan_robusto(X_meta)

    print(f"    ✓ Features: {features}")
    print(f"    ✓ X: {X_meta.shape}")
    print(f"    ✓ Clases del encoder: {len(le.classes_)}")
    print(f"    ✓ Clases presentes en metadata: {len(np.unique(y))}")

    return X_meta, y, le, features


# ════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO K-FOLD
# ════════════════════════════════════════════════════════════════════════════

def _entrenar_kfold(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray,
    n_clases: int,
    params: dict,
    nombre_modelo: str,
    label_smoothing: float = LABEL_SMOOTHING,
    n_folds: int = N_FOLDS,
    n_rounds: int = 300,
    groups: np.ndarray | None = None,
) -> dict:
    """
    Entrena LightGBM con K-Fold.

    Si groups no es None:
        usa StratifiedGroupKFold.

    Si groups es None:
        usa StratifiedKFold.
    """
    print(f"\n{'═' * 60}")
    print(f"  ENTRENANDO: {nombre_modelo}")
    print(f"  Label smoothing post-hoc: {label_smoothing}")
    print(f"  Folds: {n_folds}")
    print(f"  Rounds max: {n_rounds}")
    print(f"  Device: {params.get('device_type', 'cpu')}")

    if params.get("device_type") == "gpu":
        print(f"  gpu_platform_id: {params.get('gpu_platform_id', 'default')}")
        print(f"  gpu_device_id:   {params.get('gpu_device_id', 'default')}")

    if groups is not None:
        print("  Split: StratifiedGroupKFold")
    else:
        print("  Split: StratifiedKFold")

    print(f"{'═' * 60}")

    X_np = X.values if isinstance(X, pd.DataFrame) else X
    feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None

    X_np = np.asarray(X_np, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if groups is not None:
        groups = np.asarray(groups)

        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        split_iter = splitter.split(X_np, y, groups=groups)
    else:
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        split_iter = splitter.split(X_np, y)

    modelos = []
    oof_preds = np.zeros((len(y), n_clases), dtype=np.float32)
    fold_metrics = []
    best_iterations = []

    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        print(f"\n  ── Fold {fold}/{n_folds} ──")

        X_train = X_np[train_idx]
        X_val = X_np[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        print(f"    Train: {len(train_idx)} | Val: {len(val_idx)}")
        print(f"    Clases train: {len(np.unique(y_train))} | Clases val: {len(np.unique(y_val))}")

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=feature_names,
            free_raw_data=False,
        )

        dval = lgb.Dataset(
            X_val,
            label=y_val,
            reference=dtrain,
            feature_name=feature_names,
            free_raw_data=False,
        )

        params_fold = params.copy()
        params_fold["num_class"] = n_clases

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=EARLY_STOPPING,
                first_metric_only=False,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ]

        modelo = lgb.train(
            params=params_fold,
            train_set=dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=callbacks,
        )

        modelos.append(modelo)
        best_iterations.append(int(modelo.best_iteration))

        preds_proba = modelo.predict(
            X_val,
            num_iteration=modelo.best_iteration,
        )

        preds_proba = np.asarray(preds_proba, dtype=np.float32)

        if preds_proba.ndim == 1:
            preds_proba = preds_proba.reshape(-1, n_clases)

        if preds_proba.shape[1] != n_clases:
            raise ValueError(
                f"Predicciones con {preds_proba.shape[1]} clases, "
                f"pero se esperaban {n_clases}."
            )

        # Label smoothing post-hoc
        if label_smoothing > 0:
            preds_proba = (
                (1.0 - label_smoothing) * preds_proba
                + label_smoothing / n_clases
            )

        oof_preds[val_idx] = preds_proba

        y_pred_fold = preds_proba.argmax(axis=1)

        acc_fold = accuracy_score(y_val, y_pred_fold)

        top3_acc = _safe_top_k_accuracy(
            y_true=y_val,
            y_score=preds_proba,
            n_clases=n_clases,
            k=3,
        )

        fold_metrics.append(
            {
                "fold": int(fold),
                "accuracy": float(acc_fold),
                "top3_acc": float(top3_acc),
                "best_iteration": int(modelo.best_iteration),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_classes_train": int(len(np.unique(y_train))),
                "n_classes_val": int(len(np.unique(y_val))),
            }
        )

        print(
            f"    Accuracy: {acc_fold:.4f} | "
            f"Top-3: {top3_acc:.4f} | "
            f"Best iter: {modelo.best_iteration}"
        )

    y_pred_oof = oof_preds.argmax(axis=1)

    acc_global = accuracy_score(y, y_pred_oof)

    top3_global = _safe_top_k_accuracy(
        y_true=y,
        y_score=oof_preds,
        n_clases=n_clases,
        k=3,
    )

    print(f"\n  ── Resultados globales OOF: {nombre_modelo} ──")
    print(f"    Accuracy:   {acc_global:.4f}")
    print(f"    Top-3 Acc:  {top3_global:.4f}")

    if all(b <= 1 for b in best_iterations):
        print("\n    ⚠ ADVERTENCIA: Todos los folds terminaron con best_iteration <= 1.")
        print("      Esto puede indicar que multi_logloss no mejora o que los embeddings")
        print("      no están siendo aprovechados bien por LightGBM.")

    importances = None

    if feature_names is not None:
        imp = np.zeros(len(feature_names), dtype=np.float64)

        for modelo in modelos:
            imp += modelo.feature_importance(importance_type="gain")

        imp /= max(len(modelos), 1)

        importances = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": imp,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return {
        "modelos": modelos,
        "oof_preds": oof_preds,
        "oof_labels": y,
        "accuracy": float(acc_global),
        "top3_accuracy": float(top3_global),
        "fold_metrics": fold_metrics,
        "importances": importances,
        "best_iterations": best_iterations,
    }


# ════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ════════════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(
    y_true,
    y_pred,
    clases,
    titulo,
    output_path,
):
    """Genera y guarda matriz de confusión normalizada."""
    n_clases = len(clases)
    labels = np.arange(n_clases)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    row_sums = cm.sum(axis=1, keepdims=True)

    cm_norm = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    if n_clases <= MAX_ANNOTATED_CM_CLASSES:
        figsize = (
            max(8, n_clases * 0.45),
            max(6, n_clases * 0.40),
        )

        annot = True
        xticklabels = clases
        yticklabels = clases
        fontsize = 7
    else:
        figsize = (18, 16)
        annot = False
        xticklabels = False
        yticklabels = False
        fontsize = 6

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt=".2f",
        cmap="Blues",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ax=ax,
        cbar=True,
    )

    ax.set_title(titulo)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    if n_clases <= MAX_ANNOTATED_CM_CLASSES:
        plt.xticks(rotation=45, ha="right", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    ✓ {output_path}")


def _plot_importances(
    importances: pd.DataFrame | None,
    titulo: str,
    output_path: str,
    top_n: int = 30,
):
    """Genera gráfico de feature importances."""
    if importances is None or len(importances) == 0:
        return

    df_plot = importances.head(top_n).copy()

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(10, max(6, len(df_plot) * 0.35)),
    )

    ax.barh(
        range(len(df_plot)),
        df_plot["importance"].values,
    )

    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(titulo)
    ax.set_xlabel("Importance gain")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"    ✓ {output_path}")


def _guardar_report_json(report: dict, path: str):
    """Guarda report de sklearn como JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"    ✓ {path}")


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════════════

def entrenar_pipeline(
    embeddings_path: str,
    metadata_path: str | None = None,
    output_dir: str = "datos_fase3/",
    label_smoothing: float = LABEL_SMOOTHING,
    n_folds: int = N_FOLDS,
    n_rounds: int = 300,
    usar_group_kfold_embeddings: bool = True,
    device: str = "cpu",
    gpu_platform_id: int | None = None,
    gpu_device_id: int | None = None,
    prefer_gpu_vendor: str = "NVIDIA",
    skip_plots: bool = False,
) -> dict:
    """
    Pipeline completo:

      1. Carga embeddings.
      2. Prepara labels.
      3. Entrena LightGBM sobre embeddings.
      4. Carga metadata si existe.
      5. Entrena LightGBM sobre metadata.
      6. Guarda modelos, métricas, reports y gráficas.
    """
    device = device.lower().strip()

    print(f"\n{'═' * 60}")
    print("  FASE 3: ENTRENAMIENTO LIGHTGBM")
    print(f"  Label smoothing post-hoc: {label_smoothing}")
    print(f"  K-Folds: {n_folds}")
    print(f"  Rounds max: {n_rounds}")
    print(f"  GroupKFold embeddings: {usar_group_kfold_embeddings}")
    print(f"  Device LightGBM: {device}")

    if device == "gpu":
        print(f"  Prefer GPU vendor: {prefer_gpu_vendor}")
        print(f"  gpu_platform_id manual: {gpu_platform_id}")
        print(f"  gpu_device_id manual:   {gpu_device_id}")

    print(f"{'═' * 60}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    resultados = {}

    # ══════════════════════════════════════════════════════════════════
    # MODELO 1: EMBEDDINGS
    # ══════════════════════════════════════════════════════════════════

    print("\n[1/3] Preparando modelo de embeddings...")

    df_emb = _cargar_archivo(embeddings_path)

    X_emb, y_emb, le_emb, df_meta_emb, groups_emb = _preparar_embeddings(
        df=df_emb,
        n_folds=n_folds,
        usar_group_kfold=usar_group_kfold_embeddings,
    )

    n_clases = len(le_emb.classes_)

    params_embeddings = _aplicar_device_lightgbm(
        PARAMS_EMBEDDINGS,
        device=device,
        gpu_platform_id=gpu_platform_id,
        gpu_device_id=gpu_device_id,
        prefer_gpu_vendor=prefer_gpu_vendor,
    )

    res_emb = _entrenar_kfold(
        X=X_emb,
        y=y_emb,
        n_clases=n_clases,
        params=params_embeddings,
        nombre_modelo="EMBEDDINGS",
        label_smoothing=label_smoothing,
        n_folds=n_folds,
        n_rounds=n_rounds,
        groups=groups_emb,
    )

    resultados["embeddings"] = res_emb

    y_pred_emb = res_emb["oof_preds"].argmax(axis=1)

    report_emb = _safe_classification_report(
        y_true=y_emb,
        y_pred=y_pred_emb,
        clases=le_emb.classes_,
        titulo="Embeddings",
    )

    # ══════════════════════════════════════════════════════════════════
    # MODELO 2: METADATA
    # ══════════════════════════════════════════════════════════════════

    res_meta = None
    report_meta = None
    feature_names = None
    y_meta = None
    le_meta = None

    if metadata_path is not None and os.path.exists(metadata_path):
        print("\n[2/3] Preparando modelo de metadata...")

        df_metadata = _cargar_archivo(metadata_path)

        if "estado" in df_metadata.columns:
            antes = len(df_metadata)
            df_metadata = df_metadata[df_metadata["estado"] == "ok"].reset_index(drop=True)
            despues = len(df_metadata)

            print(f"    ✓ Filtrado estado == 'ok': {antes} → {despues}")

        X_meta, y_meta, le_meta, feature_names = _preparar_metadata(
            df=df_metadata,
            le_target=le_emb,
            min_muestras=n_folds,
        )

        if len(X_meta) >= n_folds * 2:
            n_clases_meta = len(le_meta.classes_)

            params_metadata = _aplicar_device_lightgbm(
                PARAMS_METADATA,
                device=device,
                gpu_platform_id=gpu_platform_id,
                gpu_device_id=gpu_device_id,
                prefer_gpu_vendor=prefer_gpu_vendor,
            )

            res_meta = _entrenar_kfold(
                X=X_meta,
                y=y_meta,
                n_clases=n_clases_meta,
                params=params_metadata,
                nombre_modelo="METADATA",
                label_smoothing=label_smoothing,
                n_folds=n_folds,
                n_rounds=n_rounds,
                groups=None,
            )

            resultados["metadata"] = res_meta

            y_pred_meta = res_meta["oof_preds"].argmax(axis=1)

            report_meta = _safe_classification_report(
                y_true=y_meta,
                y_pred=y_pred_meta,
                clases=le_meta.classes_,
                titulo="Metadata",
            )
        else:
            print(f"    ⚠ Muy pocos datos de metadata ({len(X_meta)}), saltando modelo metadata")
    else:
        print("\n[2/3] Sin archivo de metadata válido, saltando modelo 2")

    # ══════════════════════════════════════════════════════════════════
    # REPORTES Y VISUALIZACIONES
    # ══════════════════════════════════════════════════════════════════

    print("\n[3/3] Generando reportes...")

    _guardar_report_json(
        report_emb,
        os.path.join(output_dir, "classification_report_embeddings.json"),
    )

    if not skip_plots:
        _plot_confusion_matrix(
            y_true=y_emb,
            y_pred=y_pred_emb,
            clases=le_emb.classes_,
            titulo=f"Modelo Embeddings LightGBM OOF Acc={res_emb['accuracy']:.3f}",
            output_path=os.path.join(output_dir, "confusion_embeddings.png"),
        )
    else:
        print("    ⚠ skip_plots=True: no se genera confusion_embeddings.png")

    if res_meta is not None:
        y_pred_meta = res_meta["oof_preds"].argmax(axis=1)

        _guardar_report_json(
            report_meta,
            os.path.join(output_dir, "classification_report_metadata.json"),
        )

        if not skip_plots:
            _plot_confusion_matrix(
                y_true=y_meta,
                y_pred=y_pred_meta,
                clases=le_meta.classes_,
                titulo=f"Modelo Metadata LightGBM OOF Acc={res_meta['accuracy']:.3f}",
                output_path=os.path.join(output_dir, "confusion_metadata.png"),
            )

            _plot_importances(
                importances=res_meta["importances"],
                titulo="Feature Importances — Metadata",
                output_path=os.path.join(output_dir, "importances_metadata.png"),
            )
        else:
            print("    ⚠ skip_plots=True: no se generan gráficas de metadata")

        if res_meta["importances"] is not None:
            importances_path = os.path.join(output_dir, "importances_metadata.csv")
            res_meta["importances"].to_csv(importances_path, index=False)
            print(f"    ✓ {importances_path}")

    # ══════════════════════════════════════════════════════════════════
    # GUARDAR MODELOS
    # ══════════════════════════════════════════════════════════════════

    for i, modelo in enumerate(res_emb["modelos"]):
        path_modelo = os.path.join(output_dir, f"modelo_embeddings_fold{i}.txt")
        modelo.save_model(path_modelo)
        print(f"    ✓ {path_modelo}")

    if res_meta is not None:
        for i, modelo in enumerate(res_meta["modelos"]):
            path_modelo = os.path.join(output_dir, f"modelo_metadata_fold{i}.txt")
            modelo.save_model(path_modelo)
            print(f"    ✓ {path_modelo}")

    # Label encoder
    le_path = os.path.join(output_dir, "label_encoder.json")

    with open(le_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clases": le_emb.classes_.tolist(),
                "n_clases": int(len(le_emb.classes_)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"    ✓ Label encoder: {le_path}")

    # Feature names metadata
    if feature_names is not None:
        feature_path = os.path.join(output_dir, "features_metadata.json")

        with open(feature_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "features": feature_names,
                    "n_features": len(feature_names),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"    ✓ Features metadata: {feature_path}")

    # Métricas
    metricas = {
        "config": {
            "n_folds": int(n_folds),
            "n_rounds": int(n_rounds),
            "label_smoothing": float(label_smoothing),
            "usar_group_kfold_embeddings": bool(usar_group_kfold_embeddings),
            "random_state": int(RANDOM_STATE),
            "device": device,
            "gpu_platform_id": int(gpu_platform_id) if gpu_platform_id is not None else None,
            "gpu_device_id": int(gpu_device_id) if gpu_device_id is not None else None,
            "prefer_gpu_vendor": prefer_gpu_vendor,
            "skip_plots": bool(skip_plots),
        },
        "embeddings": {
            "accuracy": float(res_emb["accuracy"]),
            "top3_accuracy": float(res_emb["top3_accuracy"]),
            "n_clases": int(n_clases),
            "n_muestras": int(len(y_emb)),
            "n_grupos": int(len(np.unique(groups_emb))) if groups_emb is not None else None,
            "best_iterations": [int(x) for x in res_emb["best_iterations"]],
            "fold_metrics": res_emb["fold_metrics"],
        },
    }

    if res_meta is not None:
        metricas["metadata"] = {
            "accuracy": float(res_meta["accuracy"]),
            "top3_accuracy": float(res_meta["top3_accuracy"]),
            "n_clases_encoder": int(len(le_meta.classes_)),
            "n_clases_presentes": int(len(np.unique(y_meta))),
            "n_muestras": int(len(y_meta)),
            "best_iterations": [int(x) for x in res_meta["best_iterations"]],
            "fold_metrics": res_meta["fold_metrics"],
        }

    metricas_path = os.path.join(output_dir, "metricas.json")

    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"    ✓ Métricas: {metricas_path}")

    # ══════════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'═' * 60}")
    print("  RESUMEN FASE 3")
    print(f"{'═' * 60}")

    print("    Modelo Embeddings:")
    print(f"      Accuracy:    {res_emb['accuracy']:.4f}")
    print(f"      Top-3 Acc:   {res_emb['top3_accuracy']:.4f}")
    print(f"      Clases:      {n_clases}")
    print(f"      Muestras:    {len(y_emb)}")

    if groups_emb is not None:
        print(f"      Grupos:      {len(np.unique(groups_emb))}")

    if res_meta is not None:
        print("\n    Modelo Metadata:")
        print(f"      Accuracy:           {res_meta['accuracy']:.4f}")
        print(f"      Top-3 Acc:          {res_meta['top3_accuracy']:.4f}")
        print(f"      Clases encoder:     {len(le_meta.classes_)}")
        print(f"      Clases presentes:   {len(np.unique(y_meta))}")
        print(f"      Muestras:           {len(y_meta)}")

    print(f"\n    Device: {device}")

    if device == "gpu":
        print(f"    gpu_platform_id: {gpu_platform_id}")
        print(f"    gpu_device_id:   {gpu_device_id}")

    print(f"    Carpeta de salida: {output_dir}")
    print(f"{'═' * 60}\n")

    return resultados


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 3: Entrenamiento LightGBM para reconocimiento de canto de aves"
    )

    parser.add_argument(
        "--embeddings",
        required=True,
        help="Ruta a embeddings_limpios de Fase 2, .parquet o .csv",
    )

    parser.add_argument(
        "--metadata",
        default=None,
        help="Ruta a metadata_consolidado de Fase 1, .parquet o .csv",
    )

    parser.add_argument(
        "--output_dir",
        default="datos_fase3/",
        help="Carpeta de salida",
    )

    parser.add_argument(
        "--smoothing",
        type=float,
        default=LABEL_SMOOTHING,
        help=f"Label smoothing post-hoc. Default: {LABEL_SMOOTHING}",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS,
        help=f"Número de folds. Default: {N_FOLDS}",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=300,
        help="Número máximo de boosting rounds por fold. Default: 300",
    )

    parser.add_argument(
        "--no_group_kfold_embeddings",
        action="store_true",
        help="Desactiva StratifiedGroupKFold en embeddings y usa StratifiedKFold normal",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="Dispositivo LightGBM: cpu, gpu o cuda. En Windows normalmente usa gpu.",
    )

    parser.add_argument(
        "--gpu_platform_id",
        type=int,
        default=None,
        help="OpenCL platform ID para LightGBM. Para tu NVIDIA RTX 3050 usa 0.",
    )

    parser.add_argument(
        "--gpu_device_id",
        type=int,
        default=None,
        help="OpenCL device ID para LightGBM. Para tu NVIDIA RTX 3050 usa 0.",
    )

    parser.add_argument(
        "--prefer_gpu_vendor",
        default="NVIDIA",
        help="Vendor preferido para autodetección OpenCL. Default: NVIDIA.",
    )

    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="No genera matrices de confusión ni gráficas. Útil para acelerar pruebas.",
    )

    args = parser.parse_args()

    entrenar_pipeline(
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        label_smoothing=args.smoothing,
        n_folds=args.folds,
        n_rounds=args.rounds,
        usar_group_kfold_embeddings=not args.no_group_kfold_embeddings,
        device=args.device,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        prefer_gpu_vendor=args.prefer_gpu_vendor,
        skip_plots=args.skip_plots,
    )