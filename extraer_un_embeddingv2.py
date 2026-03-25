"""
Extrae embeddings + scores de BirdNET por segmento de 3s.
Cada segmento queda marcado como confiable o no, comparando
la especie esperada (etiqueta de XC) con lo que BirdNET detecta.

Uso directo:
    python extraer_un_embedding.py

Desde Python:
    from extraer_un_embedding import extract_embeddings_from_audio
    csv_path = extract_embeddings_from_audio(
        audio_path="audios/123_Tangara_heinei.mp3",
        especie_esperada="Tangara heinei",
        umbral_confianza=0.7,
    )
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from birdnet_analyzer import utils as general_utils
from birdnet_analyzer.embeddings import utils as embeddings_utils
from birdnet_analyzer import config as birdnet_config


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN BASE (igual que antes)
# ════════════════════════════════════════════════════════════════════════════

UMBRAL_CONFIANZA_DEFAULT = 0.7


def _configurar_birdnet():
    """Aplica la configuración base de BirdNET. Igual que en la versión original."""
    birdnet_config.MODEL_PATH    = birdnet_config.BIRDNET_MODEL_PATH
    birdnet_config.LABELS_FILE   = birdnet_config.BIRDNET_LABELS_FILE
    birdnet_config.SAMPLE_RATE   = birdnet_config.BIRDNET_SAMPLE_RATE
    birdnet_config.SIG_LENGTH    = birdnet_config.BIRDNET_SIG_LENGTH
    birdnet_config.AUDIO_SPEED   = 1.0
    birdnet_config.SIG_OVERLAP   = 0.0
    birdnet_config.BANDPASS_FMIN = birdnet_config.SIG_FMIN
    birdnet_config.BANDPASS_FMAX = birdnet_config.SIG_FMAX


# ════════════════════════════════════════════════════════════════════════════
# OBTENER SCORES POR SEGMENTO
# ════════════════════════════════════════════════════════════════════════════

def _obtener_scores_por_segmento(audio_path: str) -> dict:
    try:
        from birdnet_analyzer.analyze.utils import iterate_audio_chunks

        # ← Leer etiquetas directamente del archivo de texto
        if not birdnet_config.LABELS:
            with open(birdnet_config.LABELS_FILE, "r", encoding="utf-8") as f:
                birdnet_config.LABELS = [line.strip() for line in f.readlines()]

        scores_map = {}
        for s_start, s_end, pred in iterate_audio_chunks(audio_path):
            p_labels = list(zip(birdnet_config.LABELS, pred))
            p_sorted = sorted(p_labels, key=lambda x: x[1], reverse=True)
            especie_top, score_top = p_sorted[0]

            scores_map[(round(float(s_start), 1), round(float(s_end), 1))] = {
                "especie_detectada": especie_top,
                "score":             round(float(score_top), 4),
            }
        return scores_map

    except Exception as e:
        print(f"  ⚠ No se pudieron obtener scores de predicción: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════════
# MARCAR CONFIABILIDAD
# ════════════════════════════════════════════════════════════════════════════

def _marcar_confiabilidad(
    especie_detectada: str,
    score: float,
    especie_esperada: str | None,
    umbral: float,
) -> str:
    """
    Retorna una etiqueta de confiabilidad para el segmento:

    "confiable"      → BirdNET detectó la especie de XC con score >= umbral
    "otra_especie"   → BirdNET detectó una especie diferente con score >= umbral
    "sin_confianza"  → score < umbral (ruido, silencio, o especie muy dudosa)
    "sin_score"      → no se pudo obtener el score (fallo en predicción)
    """
    if score is None:
        return "sin_score"
    if score < umbral:
        return "sin_confianza"

    if especie_esperada is None:
        return "confiable"  # sin referencia XC, confiamos en BirdNET

    # Comparación flexible: ignorar mayúsculas y espacios extra
    det = especie_detectada.strip().lower()
    esp = especie_esperada.strip().lower()

    # XC usa "Género especie", BirdNET usa "Género especie_nombre_comun"
    # Comparamos solo las dos primeras palabras (nombre científico)
    det_cientifico = " ".join(det.split("_")[0:2]) if "_" in det else " ".join(det.split()[:2])
    esp_cientifico = " ".join(esp.split()[:2])

    if det_cientifico == esp_cientifico:
        return "confiable"
    else:
        return "otra_especie"


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def extract_embeddings_from_audio(
    audio_path: str,
    output_path: str = None,
    especie_esperada: str = None,
    umbral_confianza: float = UMBRAL_CONFIANZA_DEFAULT,
) -> str | None:
    """
    Extrae embeddings + scores de BirdNET por segmento de 3s.
    Marca cada segmento como confiable, otra_especie o sin_confianza.

    Args:
        audio_path:       Ruta al archivo de audio (.mp3 / .wav)
        output_path:      Ruta del CSV de salida (opcional)
        especie_esperada: Nombre científico de XC, ej. "Tangara heinei"
                          Si es None, no se compara con etiqueta de XC.
        umbral_confianza: Score mínimo para marcar un segmento como confiable.

    Returns:
        Ruta al CSV generado, o None si falló.

    Columnas del CSV:
        archivo, inicio_seg, fin_seg,
        especie_xc, especie_detectada, score,
        confiabilidad,           ← "confiable" | "otra_especie" | "sin_confianza"
        embedding                ← 1024 valores separados por ;
    """
    print(f"Extrayendo embeddings de: {audio_path}")
    if especie_esperada:
        print(f"  Especie esperada (XC): {especie_esperada}")
    print(f"  Umbral de confianza:   {umbral_confianza}")

    general_utils.ensure_model_exists(check_perch=False)

    if output_path is None:
        output_path = audio_path.replace(".wav", ".csv").replace(".mp3", ".csv")

    try:
        _configurar_birdnet()

        # ── Paso 1: embeddings (tu lógica original, intacta) ─────────────
        print("Extrayendo embeddings (3s por chunk)...")
        resultados_emb = embeddings_utils.analyze_file_core(
            audio_path, birdnet_config.get_config()
        )

        if not resultados_emb:
            print("  ✗ No se extrajo ningún fragmento (audio < 3s?).")
            return None

        print(f"  ✓ {len(resultados_emb)} fragmentos extraídos.")

        # ── Paso 2: scores de predicción por segmento ────────────────────
        print("Obteniendo scores de predicción por segmento...")
        scores_map = _obtener_scores_por_segmento(audio_path)
        if scores_map:
            print(f"  ✓ Scores obtenidos para {len(scores_map)} segmentos.")
        else:
            print("  ⚠ Sin scores — se guardarán todos como 'sin_score'.")

        # ── Paso 3: combinar y escribir CSV ──────────────────────────────
        print(f"Guardando resultados en: {output_path}")

        confiables    = 0
        otras         = 0
        sin_confianza = 0

        with open(output_path, "w") as f:
            f.write("archivo,inicio_seg,fin_seg,especie_xc,especie_detectada,score,confiabilidad,embedding\n")

            for fpath, start, end, matriz_embedding in resultados_emb:
                # Buscar score para este segmento
                key = (round(float(start), 1), round(float(end), 1))
                score_info = scores_map.get(key, {})

                especie_det = score_info.get("especie_detectada", "")
                score_val   = score_info.get("score", None)

                # Marcar confiabilidad
                confiabilidad = _marcar_confiabilidad(
                    especie_det, score_val or 0.0, especie_esperada, umbral_confianza
                )
                if score_val is None:
                    confiabilidad = "sin_score"

                # Contadores para el resumen
                if confiabilidad == "confiable":       confiables    += 1
                elif confiabilidad == "otra_especie":  otras         += 1
                else:                                  sin_confianza += 1

                # Embedding aplanado (igual que antes)
                embedding_plano = list(matriz_embedding.flatten())
                str_embedding   = ";".join(map(str, embedding_plano))

                f.write(
                    f"{os.path.basename(fpath)},"
                    f"{start},{end},"
                    f"{especie_esperada or ''},"
                    f"{especie_det},"
                    f"{score_val if score_val is not None else ''},"
                    f"{confiabilidad},"
                    f"{str_embedding}\n"
                )

        # ── Resumen ───────────────────────────────────────────────────────
        total = len(resultados_emb)
        print(f"\n  Resumen de confiabilidad (umbral={umbral_confianza}):")
        print(f"    confiable:     {confiables:>3} / {total}  ({100*confiables/total:.0f}%)")
        print(f"    otra_especie:  {otras:>3} / {total}  ({100*otras/total:.0f}%)")
        print(f"    sin_confianza: {sin_confianza:>3} / {total}  ({100*sin_confianza/total:.0f}%)")
        print(f"\n¡Extracción completada! → {output_path}")
        return output_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error al extraer embeddings: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    audio_de_prueba    = "audios/1069623_Microsciurus_mimulus.mp3"
    especie_de_prueba  = "Microsciurus mimulus"

    if os.path.exists(audio_de_prueba):
        output_file = extract_embeddings_from_audio(
            audio_path=audio_de_prueba,
            especie_esperada=especie_de_prueba,
            umbral_confianza=0.7,
        )
        if output_file:
            print(f"\nRevisa el archivo: {output_file}")
    else:
        print(f"El archivo '{audio_de_prueba}' no existe aún.")