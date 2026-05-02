"""
extraer_embeddings.py — Fase 1
───────────────────────────────────
Extrae TODOS los embeddings de BirdNET por segmento de 3s.
No filtra por score — BirdNET scores se guardan como columnas informativas.
Solo descarta silencio total (RMS < 0.0005).

La etiqueta de XC (puesta por un humano) es el ground truth.
BirdNET scores son features, no filtros.

Requisitos:
    pip install librosa numpy pandas soundfile

Uso:
    python extraer_un_embeddingv3.py --audio audios/123_Tangara_heinei.mp3 --especie "Tangara heinei"

Desde Python:
    from extraer_un_embeddingv3 import extract_embeddings_from_audio
    path = extract_embeddings_from_audio("audios/123.mp3", especie_esperada="Tangara heinei")
"""

import os
import sys
import traceback
import threading

import numpy as np
import librosa
import soundfile as sf
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from birdnet_analyzer import utils as general_utils
from birdnet_analyzer.embeddings import utils as embeddings_utils
from birdnet_analyzer import config as birdnet_config


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

UMBRAL_SILENCIO_TOTAL = 0.0005   # RMS por debajo de esto = silencio digital
TOP_N_ESPECIES        = 5
SAMPLE_RATE           = 48000

# Lock global — BirdNET usa estado global, NO es thread-safe
_BIRDNET_LOCK = threading.Lock()


def _configurar_birdnet():
    """Aplica configuración base. Llamar DENTRO del lock."""
    birdnet_config.MODEL_PATH    = birdnet_config.BIRDNET_MODEL_PATH
    birdnet_config.LABELS_FILE   = birdnet_config.BIRDNET_LABELS_FILE
    birdnet_config.SAMPLE_RATE   = birdnet_config.BIRDNET_SAMPLE_RATE
    birdnet_config.SIG_LENGTH    = birdnet_config.BIRDNET_SIG_LENGTH
    birdnet_config.AUDIO_SPEED   = 1.0
    birdnet_config.SIG_OVERLAP   = 0.0
    birdnet_config.BANDPASS_FMIN = birdnet_config.SIG_FMIN
    birdnet_config.BANDPASS_FMAX = birdnet_config.SIG_FMAX


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE AUDIO
# ════════════════════════════════════════════════════════════════════════════

def _cargar_audio(audio_path: str) -> tuple[np.ndarray | None, int]:
    """Carga el audio completo UNA sola vez."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        return y, sr
    except Exception as e:
        print(f"  ✗ No se pudo cargar: {e}")
        return None, 0


def _rms_global(y: np.ndarray) -> float:
    """RMS de todo el audio."""
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y ** 2)))


def _rms_segmento(y: np.ndarray, sr: int, start: float, end: float) -> float:
    """RMS de un segmento específico."""
    seg = y[int(start * sr): int(end * sr)]
    if len(seg) == 0:
        return 0.0
    return float(np.sqrt(np.mean(seg ** 2)))


# ════════════════════════════════════════════════════════════════════════════
# WAV TEMPORAL (fallback para mp3 que BirdNET no puede leer)
# ════════════════════════════════════════════════════════════════════════════

def _convertir_a_wav(audio_path: str) -> str | None:
    """Convierte a WAV temporal. Retorna ruta o None."""
    try:
        wav = audio_path.rsplit(".", 1)[0] + "_temp.wav"
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        sf.write(wav, y, sr)
        print(f"  ✓ Convertido a WAV temporal")
        return wav
    except Exception as e:
        print(f"  ✗ No se pudo convertir: {e}")
        return None


def _limpiar_wav(wav: str | None):
    """Borra WAV temporal si existe."""
    if wav and os.path.exists(wav):
        try:
            os.remove(wav)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# BIRDNET: embeddings + predicciones en una pasada
# ⚠️ LLAMAR SOLO DENTRO DE _BIRDNET_LOCK
# ════════════════════════════════════════════════════════════════════════════

def _birdnet_procesar(audio_path: str, top_n: int = TOP_N_ESPECIES
                      ) -> tuple[list, dict]:
    """
    Extrae embeddings y predicciones top-N.
    Si falla con mp3, reintenta con WAV temporal.
    ⚠️ Llamar DENTRO de _BIRDNET_LOCK.

    Retorna:
        resultados_emb: [(filepath, start, end, embedding_matrix), ...]
        scores_map:     {(start, end): {"top_n": [(especie, score), ...]}}
    """
    wav_temp = None
    path = audio_path

    # ── Embeddings ───────────────────────────────────────────────────
    print("  Extrayendo embeddings...")
    try:
        resultados = embeddings_utils.analyze_file_core(
            path, birdnet_config.get_config()
        )
    except Exception:
        resultados = []

    # Fallback a WAV si falla
    if not resultados:
        print("  ⚠ Reintentando con WAV temporal...")
        wav_temp = _convertir_a_wav(audio_path)
        if wav_temp is None:
            return [], {}
        path = wav_temp
        try:
            resultados = embeddings_utils.analyze_file_core(
                path, birdnet_config.get_config()
            )
        except Exception as e:
            print(f"  ✗ Fallback falló: {e}")
            _limpiar_wav(wav_temp)
            return [], {}

    if not resultados:
        _limpiar_wav(wav_temp)
        return [], {}

    print(f"  ✓ {len(resultados)} segmentos de embeddings")

    # ── Predicciones top-N ───────────────────────────────────────────
    print(f"  Extrayendo top-{top_n} predicciones...")
    scores_map = {}

    try:
        from birdnet_analyzer.analyze.utils import iterate_audio_chunks

        if not birdnet_config.LABELS:
            with open(birdnet_config.LABELS_FILE, "r", encoding="utf-8") as f:
                birdnet_config.LABELS = [line.strip() for line in f.readlines()]

        for s_start, s_end, pred in iterate_audio_chunks(path):
            p_labels = list(zip(birdnet_config.LABELS, pred))
            p_sorted = sorted(p_labels, key=lambda x: x[1], reverse=True)[:top_n]

            key = (round(float(s_start), 1), round(float(s_end), 1))
            scores_map[key] = {
                "top_n": [(e, round(float(s), 4)) for e, s in p_sorted],
            }

        print(f"  ✓ Scores para {len(scores_map)} segmentos")

    except Exception as e:
        print(f"  ⚠ Sin scores: {e}")

    _limpiar_wav(wav_temp)
    return resultados, scores_map


# ════════════════════════════════════════════════════════════════════════════
# BUSCAR ESPECIE ESPERADA EN TOP-N
# ════════════════════════════════════════════════════════════════════════════

def _normalizar_nombre(nombre: str) -> str:
    """
    Normaliza nombre científico.
    BirdNET: "Basileuterus delattrii_Chestnut-capped Warbler"
    XC:      "Basileuterus delattrii"
    → "basileuterus delattrii"
    """
    nombre = nombre.strip().lower()
    partes = nombre.replace("_", " ").split()
    if len(partes) >= 2:
        return f"{partes[0]} {partes[1]}"
    return nombre


def _buscar_especie_en_topn(
    especie_esperada: str,
    top_n: list[tuple[str, float]],
) -> tuple[float | None, int | None]:
    """
    Busca la especie esperada en la lista top-N de BirdNET.

    Retorna:
        (score, posicion) si la encuentra (posicion 1-indexed)
        (None, None)      si no la encuentra
    """
    if not especie_esperada or not top_n:
        return None, None

    esp = _normalizar_nombre(especie_esperada)
    for i, (nombre, score) in enumerate(top_n):
        if _normalizar_nombre(nombre) == esp:
            return score, i + 1
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def extract_embeddings_from_audio(
    audio_path: str,
    output_path: str = None,
    especie_esperada: str = None,
    otras_especies: str = "",
    formato: str = "csv",
) -> str | None:
    """
    Extrae TODOS los embeddings. Sin filtros de score.
    Solo descarta silencio total.

    Args:
        audio_path:       Ruta al archivo de audio
        output_path:      Ruta de salida (autogenerada si None)
        especie_esperada: Nombre científico de XC (ground truth)
        otras_especies:   Campo "also" de XC (secondary labels)
        formato:          "csv" o "parquet"

    Returns:
        Ruta al archivo generado, o None si falló.

    Columnas del CSV/Parquet:
        archivo             → nombre del archivo de audio
        xc_id               → extraído del nombre de archivo
        genero              → género de la etiqueta XC
        especie             → especie de la etiqueta XC
        inicio_seg          → inicio del segmento (segundos)
        fin_seg             → fin del segmento (segundos)
        rms_segmento        → energía RMS del segmento
        otras_especies      → secondary labels de XC
        birdnet_top1        → especie top-1 de BirdNET
        birdnet_score1      → score top-1
        birdnet_top2        → especie top-2
        birdnet_score2      → score top-2
        birdnet_top3        → especie top-3
        birdnet_score3      → score top-3
        score_especie_xc    → score que BirdNET da a la especie de XC
        posicion_especie_xc → posición de la especie XC en top-N (1-5, o vacío)
        embedding           → 1024 valores separados por ;
    """
    print(f"\n{'─' * 60}")
    print(f"Extrayendo embeddings: {os.path.basename(audio_path)}")
    if especie_esperada:
        print(f"  Etiqueta XC: {especie_esperada}")
    if otras_especies:
        print(f"  Secundarias: {otras_especies}")
    print(f"{'─' * 60}")

    if not os.path.exists(audio_path):
        print(f"  ✗ No encontrado: {audio_path}")
        return None

    general_utils.ensure_model_exists(check_perch=False)

    # Autogenerar ruta de salida
    if output_path is None:
        ext = ".parquet" if formato == "parquet" else ".csv"
        for audio_ext in [".wav", ".mp3", ".ogg", ".flac"]:
            if audio_path.lower().endswith(audio_ext):
                output_path = audio_path[:-len(audio_ext)] + ext
                break
        else:
            output_path = audio_path + ext

    try:
        # ══════════════════════════════════════════════════════════════
        # PASO 1: Cargar audio y verificar que no sea silencio total
        # ══════════════════════════════════════════════════════════════
        print("\n[1/3] Cargando audio...")
        y, sr = _cargar_audio(audio_path)
        if y is None:
            return None

        duracion = len(y) / sr
        rms = _rms_global(y)
        print(f"  ✓ {duracion:.1f}s | RMS={rms:.4f}")

        if rms < UMBRAL_SILENCIO_TOTAL:
            print(f"  ⏭️ Silencio total (RMS={rms:.6f}), saltando")
            return None

        # ══════════════════════════════════════════════════════════════
        # PASO 2: BirdNET — embeddings + predicciones (thread-safe)
        # ══════════════════════════════════════════════════════════════
        print(f"\n[2/3] BirdNET...")
        with _BIRDNET_LOCK:
            _configurar_birdnet()
            resultados_emb, scores_map = _birdnet_procesar(audio_path)

        if not resultados_emb:
            print("  ✗ No se extrajo ningún fragmento")
            return None

        # ══════════════════════════════════════════════════════════════
        # PASO 3: Construir tabla con TODOS los segmentos
        # ══════════════════════════════════════════════════════════════
        print(f"\n[3/3] Construyendo tabla ({len(resultados_emb)} segmentos)...")

        # Extraer info del nombre de archivo
        nombre_archivo = os.path.basename(audio_path)
        partes_nombre = nombre_archivo.split("_")
        xc_id = partes_nombre[0] if partes_nombre else ""

        genero = ""
        especie_sp = ""
        if especie_esperada:
            partes_esp = especie_esperada.split()
            genero = partes_esp[0] if len(partes_esp) >= 1 else ""
            especie_sp = partes_esp[1] if len(partes_esp) >= 2 else ""

        filas = []
        n_silencio = 0

        for fpath, start, end, matriz_embedding in resultados_emb:
            start_f = round(float(start), 1)
            end_f = round(float(end), 1)
            key = (start_f, end_f)

            # RMS del segmento
            rms_seg = _rms_segmento(y, sr, start_f, end_f)

            # Descartar solo silencio TOTAL
            if rms_seg < UMBRAL_SILENCIO_TOTAL:
                n_silencio += 1
                continue

            # Scores de BirdNET (como features, NO como filtros)
            info = scores_map.get(key, {})
            top_n = info.get("top_n", [])

            top1 = top_n[0] if len(top_n) >= 1 else ("", None)
            top2 = top_n[1] if len(top_n) >= 2 else ("", None)
            top3 = top_n[2] if len(top_n) >= 3 else ("", None)

            # ¿Dónde queda la especie de XC en el ranking de BirdNET?
            score_xc, posicion_xc = _buscar_especie_en_topn(
                especie_esperada, top_n
            )

            # Embedding aplanado
            str_emb = ";".join(map(str, matriz_embedding.flatten()))

            filas.append({
                "archivo":              nombre_archivo,
                "xc_id":                xc_id,
                "genero":               genero,
                "especie":              especie_sp,
                "inicio_seg":           start_f,
                "fin_seg":              end_f,
                "rms_segmento":         round(rms_seg, 6),
                "otras_especies":       otras_especies,
                "birdnet_top1":         _normalizar_nombre(top1[0]) if top1[0] else "",
                "birdnet_score1":       top1[1] if top1[1] is not None else "",
                "birdnet_top2":         _normalizar_nombre(top2[0]) if top2[0] else "",
                "birdnet_score2":       top2[1] if top2[1] is not None else "",
                "birdnet_top3":         _normalizar_nombre(top3[0]) if top3[0] else "",
                "birdnet_score3":       top3[1] if top3[1] is not None else "",
                "score_especie_xc":     score_xc if score_xc is not None else "",
                "posicion_especie_xc":  posicion_xc if posicion_xc is not None else "",
                "embedding":            str_emb,
            })

        # ── Guardar ──────────────────────────────────────────────────
        if not filas:
            print("  ✗ Todos los segmentos son silencio total")
            return None

        df = pd.DataFrame(filas)

        if formato == "parquet":
            df["embedding"] = df["embedding"].apply(
                lambda s: [float(x) for x in s.split(";")] if s else []
            )
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        # ── Resumen ──────────────────────────────────────────────────
        total = len(resultados_emb)
        guardados = len(df)
        con_especie = df["score_especie_xc"].apply(
            lambda x: x != "" and x is not None
        ).sum() if "score_especie_xc" in df.columns else 0

        print(f"\n{'═' * 60}")
        print(f"  RESUMEN")
        print(f"{'═' * 60}")
        print(f"    📊 Segmentos totales:     {total}")
        print(f"    🔕 Silencio total:         {n_silencio}")
        print(f"    💾 Guardados:              {guardados}")
        if especie_esperada:
            print(f"    🐦 BirdNET encontró '{genero} {especie_sp}' en: "
                  f"{con_especie}/{guardados} segmentos")
        print(f"    → {output_path}")
        print(f"{'═' * 60}\n")

        return output_path

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fase 1: Extrae TODOS los embeddings (sin filtros de score)"
    )
    parser.add_argument("--audio", required=True,
                        help="Ruta al archivo de audio")
    parser.add_argument("--especie", default=None,
                        help="Nombre científico (etiqueta XC)")
    parser.add_argument("--otras", default="",
                        help="Otras especies (campo 'also' de XC)")
    parser.add_argument("--formato", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--output", default=None,
                        help="Ruta de salida")
    args = parser.parse_args()

    extract_embeddings_from_audio(
        audio_path=args.audio,
        output_path=args.output,
        especie_esperada=args.especie,
        otras_especies=args.otras,
        formato=args.formato,
    )