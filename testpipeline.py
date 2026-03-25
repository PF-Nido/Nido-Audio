# test_pipeline_un_audio.py
from pruebaaudios import descargar_audio_xc
from extraer_un_embedding import extract_embeddings_from_audio
import pandas as pd

API_KEY = ""
XC_ID   = "1069623"

# Paso 1: descargar
meta, audio_path = descargar_audio_xc(
    xc_id=XC_ID,
    api_key=API_KEY,
    output_dir="audios/"
)

# Paso 2: extraer embedding
if audio_path:
    csv_path = extract_embeddings_from_audio(audio_path)

    # Paso 3: verificar resultado
    if csv_path:
        df = pd.read_csv(csv_path)
        print(f"\nResultado:")
        print(f"  Segmentos extraídos: {len(df)}")
        print(f"  Dimensión embedding: {len(df.iloc[0]['embedding'].split(';'))}")
        print(f"  Especie (del JSON):  {meta['nombre_cientifico']}")
        print(f"  Calidad XC:          {meta['calidad_xc']}")
        print(df[['archivo','inicio_seg','fin_seg']].head())