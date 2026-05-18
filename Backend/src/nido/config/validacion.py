# 4_feature_engineering.py

from pathlib import Path

# Guardar featurizer para inferencia
import joblib
import numpy as np
import pandas as pd

# ================================================================
# VALIDACIÓN RÁPIDA DEL DATASET
# ================================================================

print("Cargando dataset...")
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_parquet(BASE_DIR / "output" / "observations_with_elevation.parquet")
df = df.rename(columns={"elevation_y": "elevation"})

print(f"\n{'='*60}")
print("VALIDACIÓN DEL DATASET")
print(f"{'='*60}")
print(f"Filas: {len(df):,}")
print(f"Columnas: {list(df.columns)}")
print("\nNulos por columna:")
for col in df.columns:
    n_null = df[col].isna().sum()
    pct = n_null / len(df) * 100
    print(f"  {col:<25} {n_null:>10,} ({pct:.1f}%)")

print(f"\nEspecies únicas: {df['species'].nunique():,}")
print("\nElevación:")
print(df["elevation"].describe())

# Sanity check: ¿las elevaciones tienen sentido?
print("\nMuestra de datos:")
print(df.sample(5)[["species", "lat", "lon", "elevation"]].to_string())


# ================================================================
# FEATURE ENGINEERING
# ================================================================

print(f"\n{'='*60}")
print("GENERANDO FEATURES")
print(f"{'='*60}")


class GeoTemporalFeaturizer:
    """
    Genera features espaciotemporales para el Modelo B.
    Diseñado para aves de Colombia.
    """

    def __init__(self):
        self.fitted = False
        self.elevation_median = None

    def fit_transform(self, df):
        """Para entrenamiento: calcula estadísticas + transforma."""
        self.elevation_median = df["elevation"].median()
        self.fitted = True
        return self._create_features(df)

    def transform(self, df):
        """Para inferencia: usa estadísticas guardadas."""
        if not self.fitted:
            raise ValueError("Llamar fit_transform primero")
        return self._create_features(df)

    def _create_features(self, df):
        features = pd.DataFrame(index=df.index)

        # ═══════════════════════════════════════
        # TEMPORAL: Codificación cíclica
        # ═══════════════════════════════════════

        # if 'hour_decimal' in df.columns:
        #     hour = df['hour_decimal'].values
        #     features['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype('float32')
        #     features['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype('float32')

        #     # Períodos del día (importante para aves)
        #     features['is_dawn'] = ((hour >= 5) & (hour <= 7)).astype('int8')
        #     features['is_morning'] = ((hour >= 5) & (hour <= 11)).astype('int8')
        #     features['is_afternoon'] = ((hour >= 12) & (hour <= 17)).astype('int8')
        #     features['is_dusk'] = ((hour >= 17) & (hour <= 19)).astype('int8')
        #     features['is_night'] = ((hour >= 20) | (hour <= 4)).astype('int8')

        if "day_of_year" in df.columns:
            day = df["day_of_year"].values
            features["day_sin"] = np.sin(2 * np.pi * day / 365).astype("float32")
            features["day_cos"] = np.cos(2 * np.pi * day / 365).astype("float32")

        if "month" in df.columns:
            month = df["month"].values
            features["month_sin"] = np.sin(2 * np.pi * month / 12).astype("float32")
            features["month_cos"] = np.cos(2 * np.pi * month / 12).astype("float32")

        # ═══════════════════════════════════════
        # ESPACIAL: Coordenadas normalizadas
        # ═══════════════════════════════════════

        # Normalizado para Colombia (centrado y escalado)
        features["lat_norm"] = ((df["lat"] - 4.5) / 10.0).astype("float32")
        features["lon_norm"] = ((df["lon"] + 73.0) / 8.0).astype("float32")

        # ═══════════════════════════════════════
        # ELEVACIÓN
        # ═══════════════════════════════════════

        if "elevation" in df.columns:
            # Imputar nulos
            elev = df["elevation"].fillna(self.elevation_median).values

            features["elevation"] = elev.astype("float32")
            features["elevation_log"] = np.log1p(np.clip(elev, 0, None)).astype(
                "float32"
            )

            # Bandas altitudinales de Colombia
            features["is_lowland"] = (elev < 1000).astype("int8")
            features["is_submontane"] = ((elev >= 1000) & (elev < 2000)).astype("int8")
            features["is_montane"] = ((elev >= 2000) & (elev < 3000)).astype("int8")
            features["is_highland"] = ((elev >= 3000) & (elev < 3500)).astype("int8")
            features["is_paramo"] = (elev >= 3500).astype("int8")

            # Flag de dato faltante
            features["elevation_missing"] = df["elevation"].isna().astype("int8")

        # ═══════════════════════════════════════
        # INTERACCIONES
        # ═══════════════════════════════════════

        # Latitud × Mes (estacionalidad varía con latitud)
        if "month" in df.columns:
            features["lat_x_month_sin"] = (
                features["lat_norm"] * features["month_sin"]
            ).astype("float32")

        # Elevación × Latitud (andino vs no andino)
        if "elevation" in df.columns:
            features["elev_x_lat"] = (
                features["elevation"] * features["lat_norm"]
            ).astype("float32")

        return features


# ================================================================
# EJECUTAR
# ================================================================

featurizer = GeoTemporalFeaturizer()
features = featurizer.fit_transform(df)

print(f"\nFeatures generadas: {features.shape[1]}")
print("Columnas:")
for col in features.columns:
    print(f"  {col}: {features[col].dtype}")

# Agregar labels
features["species"] = df["species"].values

# Guardar features

output_path = BASE_DIR / "output" / "features_model_b.parquet"
features.to_parquet(output_path, compression="snappy")
size_mb = Path(output_path).stat().st_size / 1e6
print(f"\nGuardado: {output_path} ({size_mb:.0f} MB)")


Path("models/geo/").mkdir(parents=True, exist_ok=True)
joblib.dump(featurizer, "models/geo/featurizer.joblib")
print("Featurizer guardado: models/geo/featurizer.joblib")

print("\n✅ Feature engineering completo")
