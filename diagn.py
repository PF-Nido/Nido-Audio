"""
diagnostico_taxonomia.py
Muestra qué especie detecta BirdNET vs la etiqueta de XC.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from birdnet_analyzer import utils as general_utils
from birdnet_analyzer import config as birdnet_config

def configurar():
    general_utils.ensure_model_exists(check_perch=False)
    birdnet_config.MODEL_PATH    = birdnet_config.BIRDNET_MODEL_PATH
    birdnet_config.LABELS_FILE   = birdnet_config.BIRDNET_LABELS_FILE
    birdnet_config.SAMPLE_RATE   = birdnet_config.BIRDNET_SAMPLE_RATE
    birdnet_config.SIG_LENGTH    = birdnet_config.BIRDNET_SIG_LENGTH
    birdnet_config.AUDIO_SPEED   = 1.0
    birdnet_config.SIG_OVERLAP   = 0.0
    birdnet_config.BANDPASS_FMIN = birdnet_config.SIG_FMIN
    birdnet_config.BANDPASS_FMAX = birdnet_config.SIG_FMAX

def ver_predicciones(audio_path: str, especie_xc: str):
    from birdnet_analyzer.analyze.utils import iterate_audio_chunks

    if not birdnet_config.LABELS:
        with open(birdnet_config.LABELS_FILE, "r", encoding="utf-8") as f:
            birdnet_config.LABELS = [line.strip() for line in f.readlines()]

    print(f"\n{'═'*70}")
    print(f"Audio:      {os.path.basename(audio_path)}")
    print(f"Especie XC: {especie_xc}")
    print(f"{'═'*70}")

    for s_start, s_end, pred in iterate_audio_chunks(audio_path):
        p_labels = list(zip(birdnet_config.LABELS, pred))
        p_sorted = sorted(p_labels, key=lambda x: x[1], reverse=True)[:5]

        print(f"\n  Segmento {s_start}s – {s_end}s:")
        for i, (especie, score) in enumerate(p_sorted):
            marca = "  ←←←" if especie_xc.lower() in especie.lower() else ""
            print(f"    #{i+1}  {score:.4f}  {especie}{marca}")

    # Buscar si la especie de XC aparece en las labels de BirdNET
    especie_lower = especie_xc.lower()
    coincidencias = [l for l in birdnet_config.LABELS if especie_lower in l.lower()]
    sinonimos = [l for l in birdnet_config.LABELS
                 if especie_xc.split()[-1].lower() in l.lower()]

    print(f"\n{'─'*70}")
    print(f"  ¿'{especie_xc}' existe en labels de BirdNET?")
    if coincidencias:
        print(f"    ✓ Sí: {coincidencias}")
    else:
        print(f"    ✗ NO encontrada exacta")
        if sinonimos:
            print(f"    Posibles sinónimos (por epíteto '{especie_xc.split()[-1]}'):")
            for s in sinonimos[:10]:
                print(f"      → {s}")
        else:
            print(f"    Tampoco se encontró el epíteto específico")
    print(f"{'─'*70}")


if __name__ == "__main__":
    configurar()

    # Cambia por un audio que tengas (usa --keep_audio para conservarlo)
    ver_predicciones(
        "resultados_prueba/725922_Basileuterus_delattrii.mp3",
        "Basileuterus delattrii"
    )