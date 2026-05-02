"""
extraer_un_embeddingv3.py
─────────────────────────
Extrae embeddings + scores de BirdNET por segmento de 3s.
Thread-safe con lock global para BirdNET.

Requisitos:
    pip install librosa numpy pandas pyarrow soundfile
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

UMBRAL_CONFIANZA_DEFAULT = 0.5
UMBRAL_RMS_DEFAULT       = 0.002
UMBRAL_SNR_DEFAULT       = 1.0
UMBRAL_MULTI_ESPECIE     = 0.3
TOP_N_ESPECIES           = 5
SAMPLE_RATE              = 48000

# Lock global — BirdNET usa estado global, NO es thread-safe
_BIRDNET_LOCK = threading.Lock()


def _configurar_birdnet():
    """Aplica la configuración base de BirdNET. Llamar DENTRO del lock."""
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

def _cargar_audio_completo(audio_path: str) -> tuple[np.ndarray | None, int]:
    """Carga el audio completo UNA sola vez con librosa."""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        return y, sr
    except Exception as e:
        print(f"  ⚠ No se pudo cargar el audio con librosa: {e}")
        return None, 0


def _extraer_segmento(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Extrae un segmento del array ya cargado en memoria."""
    return y[int(start * sr): int(end * sr)]


def _calcular_energia_rms(segmento: np.ndarray) -> float:
    """RMS del segmento."""
    if len(segmento) == 0:
        return 0.0
    return float(np.sqrt(np.mean(segmento ** 2)))


def _calcular_snr(segmento: np.ndarray, sr: int,
                  fmin: float = 500.0, fmax: float = 12000.0) -> float:
    """SNR en dB. Señal = banda de aves (500-12kHz), ruido = el resto."""
    if len(segmento) < 512:
        return 0.0
    try:
        S = np.abs(librosa.stft(segmento, n_fft=1024, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

        mask_signal = (freqs >= fmin) & (freqs <= fmax)
        mask_noise = ~mask_signal

        e_signal = max(np.mean(S[mask_signal, :] ** 2), 1e-10) if mask_signal.any() else 1e-10
        e_noise = max(np.mean(S[mask_noise, :] ** 2), 1e-10) if mask_noise.any() else 1e-10

        return round(float(10.0 * np.log10(e_signal / e_noise)), 2)
    except Exception:
        return 0.0


def _pre_filtrar_segmentos(
    y: np.ndarray,
    sr: int,
    duracion_total: float,
    sig_length: float = 3.0,
    umbral_rms: float = UMBRAL_RMS_DEFAULT,
    umbral_snr: float = UMBRAL_SNR_DEFAULT,
) -> dict[tuple[float, float], dict]:
    """Pre-filtra segmentos por RMS y SNR ANTES de BirdNET."""
    segmentos = {}
    start = 0.0

    while start + sig_length <= duracion_total + 0.01:
        end = min(start + sig_length, duracion_total)
        seg = _extraer_segmento(y, sr, start, end)

        rms = _calcular_energia_rms(seg)
        snr = _calcular_snr(seg, sr)

        es_silencio = rms < umbral_rms
        es_ruido = snr < umbral_snr and not es_silencio

        key = (round(start, 1), round(end, 1))
        segmentos[key] = {
            "energia_rms": rms,
            "snr_db": snr,
            "es_silencio": es_silencio,
            "es_ruido": es_ruido,
            "saltar": es_silencio or es_ruido,
        }

        start += sig_length

    n_saltar = sum(1 for v in segmentos.values() if v["saltar"])
    n_total = len(segmentos)
    if n_total > 0:
        print(f"  Pre-filtro: {n_saltar}/{n_total} segmentos descartados "
              f"({n_saltar / n_total * 100:.0f}% silencio/ruido)")

    return segmentos


# ════════════════════════════════════════════════════════════════════════════
# WAV TEMPORAL (fallback)
# ════════════════════════════════════════════════════════════════════════════

def _convertir_a_wav_temporal(audio_path: str) -> str | None:
    """Convierte mp3 a WAV temporal."""
    try:
        wav_temp = audio_path.rsplit(".", 1)[0] + "_temp.wav"
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        sf.write(wav_temp, y, sr)
        print(f"  ✓ Convertido a WAV temporal")
        return wav_temp
    except Exception as e:
        print(f"  ✗ No se pudo convertir a WAV: {e}")
        return None


def _limpiar_wav_temporal(wav_temp: str | None):
    """Borra el WAV temporal si existe."""
    if wav_temp and os.path.exists(wav_temp):
        try:
            os.remove(wav_temp)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# BIRDNET: embeddings + predicciones
# ⚠️ LLAMAR SOLO DENTRO DE _BIRDNET_LOCK
# ════════════════════════════════════════════════════════════════════════════

def _birdnet_una_sola_pasada(
    audio_path: str,
    top_n: int = TOP_N_ESPECIES,
) -> tuple[list, dict]:
    """
    Embeddings + predicciones. Con fallback a WAV si falla.
    ⚠️ DEBE llamarse dentro de _BIRDNET_LOCK.
    """
    wav_temp = None
    analysis_path = audio_path

    # ── Embeddings ───────────────────────────────────────────────────
    print("  Extrayendo embeddings (3s por chunk)...")

    try:
        resultados_emb = embeddings_utils.analyze_file_core(
            analysis_path, birdnet_config.get_config()
        )
    except Exception as e:
        print(f"  ⚠ BirdNET lanzó excepción: {e}")
        resultados_emb = []

    # Fallback a WAV si resultado vacío
    if not resultados_emb:
        print(f"  ⚠ BirdNET no pudo procesar el mp3, convirtiendo a WAV...")
        wav_temp = _convertir_a_wav_temporal(audio_path)
        if wav_temp is None:
            return [], {}

        analysis_path = wav_temp
        try:
            resultados_emb = embeddings_utils.analyze_file_core(
                analysis_path, birdnet_config.get_config()
            )
        except Exception as e2:
            print(f"  ✗ Fallback WAV también falló: {e2}")
            _limpiar_wav_temporal(wav_temp)
            return [], {}

    if not resultados_emb:
        print("  ✗ 0 fragmentos extraídos (incluso con WAV)")
        _limpiar_wav_temporal(wav_temp)
        return [], {}

    print(f"  ✓ {len(resultados_emb)} fragmentos de embeddings")

    # ── Predicciones ─────────────────────────────────────────────────
    print(f"  Extrayendo top-{top_n} predicciones...")
    scores_map = {}

    try:
        from birdnet_analyzer.analyze.utils import iterate_audio_chunks

        if not birdnet_config.LABELS:
            with open(birdnet_config.LABELS_FILE, "r", encoding="utf-8") as f:
                birdnet_config.LABELS = [line.strip() for line in f.readlines()]

        for s_start, s_end, pred in iterate_audio_chunks(analysis_path):
            p_labels = list(zip(birdnet_config.LABELS, pred))
            p_sorted = sorted(p_labels, key=lambda x: x[1], reverse=True)[:top_n]

            key = (round(float(s_start), 1), round(float(s_end), 1))
            scores_map[key] = {
                "especie_detectada": p_sorted[0][0],
                "score": round(float(p_sorted[0][1]), 4),
                "top_n": [(e, round(float(s), 4)) for e, s in p_sorted],
            }

        print(f"  ✓ Scores para {len(scores_map)} segmentos")

    except Exception as e:
        print(f"  ⚠ No se pudieron obtener scores: {e}")

    _limpiar_wav_temporal(wav_temp)
    return resultados_emb, scores_map


# ════════════════════════════════════════════════════════════════════════════
# CONFIABILIDAD
# ════════════════════════════════════════════════════════════════════════════

def _normalizar_nombre(nombre: str) -> str:
    """Normaliza nombre científico para comparación."""
    nombre = nombre.strip().lower()
    partes = nombre.replace("_", " ").split()
    if len(partes) >= 2:
        return f"{partes[0]} {partes[1]}"
    return nombre


def _evaluar_confiabilidad(
    especie_detectada: str,
    score: float | None,
    especie_esperada: str | None,
    umbral: float,
    energia_rms: float,
    snr_db: float,
    top_n: list[tuple[str, float]] | None = None,
    umbral_rms: float = UMBRAL_RMS_DEFAULT,
    umbral_snr: float = UMBRAL_SNR_DEFAULT,
    umbral_multi: float = UMBRAL_MULTI_ESPECIE,
) -> dict:
    """Evalúa confiabilidad con múltiples criterios."""

    if energia_rms < umbral_rms:
        return {"confiabilidad": "silencio",
                "razon": f"RMS={energia_rms:.4f} < {umbral_rms}",
                "usar_para_entrenamiento": False}

    if snr_db < umbral_snr:
        return {"confiabilidad": "ruido_ambiental",
                "razon": f"SNR={snr_db:.1f}dB < {umbral_snr}dB",
                "usar_para_entrenamiento": False}

    if score is None:
        return {"confiabilidad": "sin_score",
                "razon": "No se obtuvo predicción de BirdNET",
                "usar_para_entrenamiento": False}

    if score < 0.1:
        return {"confiabilidad": "ruido",
                "razon": f"Score={score:.3f} < 0.1, probablemente no es ave",
                "usar_para_entrenamiento": False}

    if score < umbral:
        return {"confiabilidad": "sin_confianza",
                "razon": f"Score={score:.3f} < umbral={umbral}",
                "usar_para_entrenamiento": False}

    if especie_esperada:
        det = _normalizar_nombre(especie_detectada)
        esp = _normalizar_nombre(especie_esperada)

        if det == esp:
            if top_n and len(top_n) >= 2:
                seg_esp, seg_score = top_n[1]
                if seg_score > umbral_multi:
                    return {"confiabilidad": "multi_especie",
                            "razon": (f"Correcta (score={score:.3f}) pero "
                                      f"'{_normalizar_nombre(seg_esp)}' "
                                      f"score={seg_score:.3f}>{umbral_multi}"),
                            "usar_para_entrenamiento": False}

            return {"confiabilidad": "confiable",
                    "razon": f"Coincide con XC, score={score:.3f}",
                    "usar_para_entrenamiento": True}
        else:
            return {"confiabilidad": "otra_especie",
                    "razon": f"XC='{esp}', BirdNET='{det}' (score={score:.3f})",
                    "usar_para_entrenamiento": False}

    return {"confiabilidad": "confiable_sin_ref",
            "razon": f"Score={score:.3f} >= umbral, sin etiqueta XC",
            "usar_para_entrenamiento": True}


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def extract_embeddings_from_audio(
    audio_path: str,
    output_path: str = None,
    especie_esperada: str = None,
    umbral_confianza: float = UMBRAL_CONFIANZA_DEFAULT,
    umbral_rms: float = UMBRAL_RMS_DEFAULT,
    umbral_snr: float = UMBRAL_SNR_DEFAULT,
    formato: str = "csv",
    solo_entrenamiento: bool = False,
) -> str | None:
    """
    Pipeline optimizado y thread-safe:
      1. Carga audio UNA vez con librosa (fuera del lock)
      2. Pre-filtra silencio/ruido (fuera del lock)
      3. BirdNET: embeddings + predicciones (DENTRO del lock)
      4. Evalúa confiabilidad (fuera del lock)
      5. Guarda CSV/Parquet solo si hay datos
    """
    print(f"\n{'─' * 60}")
    print(f"Extrayendo embeddings: {audio_path}")
    if especie_esperada:
        print(f"  Especie esperada: {especie_esperada}")
    print(f"  Umbrales: confianza={umbral_confianza} | "
          f"RMS={umbral_rms} | SNR={umbral_snr}dB")
    print(f"{'─' * 60}")

    if not os.path.exists(audio_path):
        print(f"  ✗ Archivo no encontrado: {audio_path}")
        return None

    general_utils.ensure_model_exists(check_perch=False)

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
        # PASO 1: Cargar audio (fuera del lock, no usa BirdNET)
        # ══════════════════════════════════════════════════════════════
        print("\n[1/4] Cargando audio...")
        y_completo, sr = _cargar_audio_completo(audio_path)
        if y_completo is None:
            print("  ✗ Audio corrupto o ilegible")
            return None
        duracion = len(y_completo) / sr
        print(f"  ✓ {duracion:.1f}s | {sr} Hz")

        rms_global = _calcular_energia_rms(y_completo)
        if rms_global < umbral_rms:
            print(f"  ⏭️ Audio completo es silencio (RMS={rms_global:.4f})")
            return None

        # ══════════════════════════════════════════════════════════════
        # PASO 2: Pre-filtrar (fuera del lock, solo numpy/librosa)
        # ══════════════════════════════════════════════════════════════
        print("\n[2/4] Pre-filtrando segmentos...")
        pre_filtro = _pre_filtrar_segmentos(
            y_completo, sr, duracion,
            umbral_rms=umbral_rms, umbral_snr=umbral_snr,
        )

        n_utiles = sum(1 for v in pre_filtro.values() if not v["saltar"])
        if n_utiles == 0:
            print("  ⏭️ Todos los segmentos son silencio/ruido")
            return None

        # ══════════════════════════════════════════════════════════════
        # PASO 3: BirdNET — DENTRO DEL LOCK (thread-safe)
        # ══════════════════════════════════════════════════════════════
        print(f"\n[3/4] BirdNET ({n_utiles} útiles de {len(pre_filtro)})...")

        with _BIRDNET_LOCK:
            _configurar_birdnet()
            resultados_emb, scores_map = _birdnet_una_sola_pasada(audio_path)

        if not resultados_emb:
            print("  ✗ No se extrajo ningún fragmento")
            return None

        # ══════════════════════════════════════════════════════════════
        # PASO 4: Confiabilidad (fuera del lock)
        # ══════════════════════════════════════════════════════════════
        print(f"\n[4/4] Evaluando confiabilidad...")
        filas = []
        contadores = {}

        for fpath, start, end, matriz_embedding in resultados_emb:
            start_f = round(float(start), 1)
            end_f = round(float(end), 1)
            key = (start_f, end_f)

            pre = pre_filtro.get(key, {})
            energia_rms = pre.get("energia_rms", _calcular_energia_rms(
                _extraer_segmento(y_completo, sr, start_f, end_f)
            ))
            snr_db = pre.get("snr_db", _calcular_snr(
                _extraer_segmento(y_completo, sr, start_f, end_f), sr
            ))

            if pre.get("saltar", False):
                if pre.get("es_silencio"):
                    conf = "silencio"
                    razon = f"RMS={energia_rms:.4f} < {umbral_rms}"
                else:
                    conf = "ruido_ambiental"
                    razon = f"SNR={snr_db:.1f}dB < {umbral_snr}dB"

                evaluacion = {
                    "confiabilidad": conf,
                    "razon": razon,
                    "usar_para_entrenamiento": False,
                }
                score_info = {}
            else:
                score_info = scores_map.get(key, {})
                evaluacion = _evaluar_confiabilidad(
                    especie_detectada=score_info.get("especie_detectada", ""),
                    score=score_info.get("score"),
                    especie_esperada=especie_esperada,
                    umbral=umbral_confianza,
                    energia_rms=energia_rms,
                    snr_db=snr_db,
                    top_n=score_info.get("top_n", []),
                    umbral_rms=umbral_rms,
                    umbral_snr=umbral_snr,
                )

            conf = evaluacion["confiabilidad"]
            contadores[conf] = contadores.get(conf, 0) + 1

            if solo_entrenamiento and not evaluacion["usar_para_entrenamiento"]:
                continue

            top_n_preds = score_info.get("top_n", [])
            especie_2 = top_n_preds[1][0] if len(top_n_preds) >= 2 else ""
            score_2 = top_n_preds[1][1] if len(top_n_preds) >= 2 else None

            str_embedding = ";".join(map(str, matriz_embedding.flatten()))

            filas.append({
                "archivo": os.path.basename(fpath),
                "inicio_seg": start_f,
                "fin_seg": end_f,
                "especie_xc": especie_esperada or "",
                "especie_detectada": score_info.get("especie_detectada", ""),
                "score": score_info.get("score", ""),
                "especie_2": especie_2,
                "score_2": score_2 if score_2 is not None else "",
                "energia_rms": round(energia_rms, 6),
                "snr_db": snr_db,
                "confiabilidad": conf,
                "razon": evaluacion["razon"],
                "usar_para_entrenamiento": evaluacion["usar_para_entrenamiento"],
                "embedding": str_embedding,
            })

        # ══════════════════════════════════════════════════════════════
        # PASO 5: Guardar solo si hay datos
        # ══════════════════════════════════════════════════════════════
        total = len(resultados_emb)

        if not filas:
            print("  ⚠ 0 segmentos pasaron los filtros")
            _imprimir_resumen(contadores, total, 0, umbral_confianza,
                              umbral_rms, umbral_snr, "N/A")
            return None

        df = pd.DataFrame(filas)

        if formato == "parquet":
            df["embedding"] = df["embedding"].apply(
                lambda s: [float(x) for x in s.split(";")] if s else []
            )
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        _imprimir_resumen(contadores, total, len(df), umbral_confianza,
                          umbral_rms, umbral_snr, output_path)
        return output_path

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return None


def _imprimir_resumen(contadores, total, guardados, umbral_conf,
                      umbral_rms, umbral_snr, output_path):
    """Resumen con barras visuales."""
    categorias = [
        ("confiable", "✅"),
        ("confiable_sin_ref", "✅"),
        ("multi_especie", "⚠️ "),
        ("otra_especie", "🔄"),
        ("sin_confianza", "❓"),
        ("ruido", "🔇"),
        ("ruido_ambiental", "🌧️ "),
        ("silencio", "🔕"),
        ("sin_score", "❌"),
    ]

    print(f"\n{'═' * 60}")
    print(f"  RESUMEN (conf={umbral_conf} | RMS={umbral_rms} | SNR={umbral_snr}dB)")
    print(f"{'═' * 60}")

    for cat, emoji in categorias:
        n = contadores.get(cat, 0)
        if n > 0:
            pct = 100 * n / total if total > 0 else 0
            barra = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"    {emoji} {cat:<20s} {n:>4}/{total}  {barra} {pct:.0f}%")

    aptos = contadores.get("confiable", 0) + contadores.get("confiable_sin_ref", 0)
    print(f"\n    📊 Total:      {total}")
    print(f"    ✅ Aptos:      {aptos}")
    print(f"    🗑️  Descartados: {total - aptos}")
    print(f"    💾 Guardados:  {guardados}")
    print(f"    → {output_path}")
    print(f"{'═' * 60}\n")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrae embeddings de BirdNET con filtros de calidad"
    )
    parser.add_argument("--audio", required=True)
    parser.add_argument("--especie", default=None)
    parser.add_argument("--umbral", type=float, default=UMBRAL_CONFIANZA_DEFAULT)
    parser.add_argument("--rms", type=float, default=UMBRAL_RMS_DEFAULT)
    parser.add_argument("--snr", type=float, default=UMBRAL_SNR_DEFAULT)
    parser.add_argument("--formato", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--solo_entrenamiento", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    extract_embeddings_from_audio(
        audio_path=args.audio,
        output_path=args.output,
        especie_esperada=args.especie,
        umbral_confianza=args.umbral,
        umbral_rms=args.rms,
        umbral_snr=args.snr,
        formato=args.formato,
        solo_entrenamiento=args.solo_entrenamiento,
    )