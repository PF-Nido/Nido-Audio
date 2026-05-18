import sys
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import pandas as pd

import nido.config.validacion as validacion

sys.modules["validacion"] = validacion

BASE_DIR = Path(__file__).resolve().parents[3]
GEO_MODEL_DIR = BASE_DIR / "models" / "geo"

_featurizer = None
_label_encoder = None
_model = None


def _load_geo_model():
    global _featurizer, _label_encoder, _model

    if _model is not None:
        return

    featurizer_path = GEO_MODEL_DIR / "featurizer.joblib"
    label_encoder_path = GEO_MODEL_DIR / "label_encoder.joblib"
    model_path = GEO_MODEL_DIR / "model_b.txt"

    if not all(p.exists() for p in [featurizer_path, label_encoder_path, model_path]):
        raise FileNotFoundError(
            f"Archivos del modelo geo no encontrados en {GEO_MODEL_DIR}. "
            "Copia los archivos del modelo antes de continuar."
        )

    _featurizer = joblib.load(featurizer_path)
    _label_encoder = joblib.load(label_encoder_path)
    _model = lgb.Booster(model_file=str(model_path))
    print("Modelo geoespacial cargado")


def predict_geo(
    latitude: float,
    longitude: float,
    month: int,
    day_of_year: int,
    elevation: Optional[float] = None,
) -> dict[str, float]:
    """
    Retorna probabilidades por especie basadas en contexto geoespacial.
    {nombre_especie: probabilidad}
    """
    _load_geo_model()

    input_df = pd.DataFrame(
        [
            {
                "lat": latitude,
                "lon": longitude,
                "month": month,
                "day_of_year": day_of_year,
                "elevation": elevation,
            }
        ]
    )

    features = _featurizer.transform(input_df).astype("float32")  # type: ignore
    probs = _model.predict(features)[0]  # type: ignore
    species_names = _label_encoder.classes_  # type: ignore

    return {species: float(prob) for species, prob in zip(species_names, probs)}  # type: ignore
