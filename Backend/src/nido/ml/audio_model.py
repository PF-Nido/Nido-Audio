import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import lightgbm as lgb
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[3]
AUDIO_MODEL_DIR = BASE_DIR / "models" / "audio"

# BirdNET no es thread-safe
_BIRDNET_LOCK = threading.Lock()

_clases = None
_modelos_emb = None


def _load_audio_model():
    global _clases, _modelos_emb

    if _modelos_emb is not None:
        return

    label_encoder_path = AUDIO_MODEL_DIR / "label_encoder.json"
    if not label_encoder_path.exists():
        raise FileNotFoundError(
            f"label_encoder.json no encontrado en {AUDIO_MODEL_DIR}."
        )

    with open(label_encoder_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        _clases = np.array(data["clases"])

    _modelos_emb = []
    for file in sorted(AUDIO_MODEL_DIR.glob("modelo_embeddings_fold*.txt")):
        _modelos_emb.append(lgb.Booster(model_file=str(file)))

    if not _modelos_emb:
        raise ValueError(f"No se encontraron modelos fold en {AUDIO_MODEL_DIR}.")

    print(f"Modelo de audio cargado: {len(_modelos_emb)} folds")


def _extract_embeddings(audio_path: str) -> list[np.ndarray]:
    project_root = str(BASE_DIR.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

    from birdnet_analyzer.embeddings import embeddings
    from usearch.index import Index

    with tempfile.TemporaryDirectory() as tmpdir:

        db_path = Path(tmpdir) / "embeddings_db"

        with _BIRDNET_LOCK:
            embeddings(
                audio_input=audio_path,
                database=str(db_path),
                batch_size=1,
            )

        index_path = db_path / "usearch.index"

        if not index_path.exists():
            return []

        index = Index.restore(str(index_path))

        resultados = []

        for key in index.keys:  # type: ignore
            emb = np.array(index[key], dtype=np.float32)  # type: ignore
            resultados.append(emb)

        return resultados


def predict_audio(audio_bytes: bytes) -> dict[str, float]:
    """
    Recibe bytes del audio y retorna probabilidades por especie.
    {nombre_especie: probabilidad}
    """
    _load_audio_model()

    # Escribir en archivo temporal porque BirdNET necesita ruta en disco
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        resultados = _extract_embeddings(tmp_path)
        print(f"Resultados de embeddings: {len(resultados)}")
        if not resultados:
            return {}

        # Acumular probabilidades por especie sobre todos los segmentos
        acumulado = np.zeros(len(_clases))  # type: ignore

        for emb in resultados:
            vec = np.array(emb).flatten().reshape(1, -1)

            # Promediar sobre todos los folds
            preds = np.zeros((1, len(_clases)))  # type: ignore
            for modelo in _modelos_emb:  # type: ignore
                preds += modelo.predict(vec)
            preds /= len(_modelos_emb)  # type: ignore

            acumulado += preds[0]

        # Promediar sobre segmentos
        acumulado /= len(resultados)

        return {especie: float(prob) for especie, prob in zip(_clases, acumulado)}  # type: ignore

    finally:
        os.unlink(tmp_path)
