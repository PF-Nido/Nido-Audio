import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import pandas as pd
from huggingface_hub import hf_hub_download

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

BASE_DIR = Path(__file__).resolve().parents[3]

HF_REPO = os.getenv("HF_REPO", "albertjjs/nido-models")
HF_TOKEN = os.getenv("HF_TOKEN")

_featurizer = None
_label_encoder = None
_model = None


def _load_geo_model():
    global _featurizer, _label_encoder, _model

    if _model is not None:
        return

    # To fix ModuleNotFoundError: No module named 'validacion' during unpickling
    import sys
    try:
        from nido.config import validacion
        sys.modules['validacion'] = validacion
    except ImportError:
        pass

    featurizer_path = hf_hub_download(HF_REPO, "models/geo/featurizer.joblib", token=HF_TOKEN)
    label_encoder_path = hf_hub_download(HF_REPO, "models/geo/label_encoder.joblib", token=HF_TOKEN)
    model_path = hf_hub_download(HF_REPO, "models/geo/model_b.txt", token=HF_TOKEN)

    _featurizer = joblib.load(featurizer_path)
    _label_encoder = joblib.load(label_encoder_path)
    _model = lgb.Booster(model_file=model_path)
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
