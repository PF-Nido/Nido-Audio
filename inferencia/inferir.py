import os
import sys
import json
import argparse
import numpy as np
import lightgbm as lgb
import threading

# Añadir el directorio raíz al path para poder importar módulos del proyecto (birdnet_analyzer, etc.)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from birdnet_analyzer import config as birdnet_config
from birdnet_analyzer.embeddings import utils as embeddings_utils

# Bloqueo global ya que BirdNET no es thread-safe
_BIRDNET_LOCK = threading.Lock()

class AudioPredictor:
    def __init__(self, model_dir="entrenamiento/datos_fase3"):
        # Construir ruta relativa a la carpeta raíz del proyecto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(base_dir, model_dir)
        
        le_path = os.path.join(self.model_dir, "label_encoder.json")
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"No se encontró label_encoder.json en {self.model_dir}. Debe entrenar el modelo primero.")
            
        with open(le_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.clases = np.array(data["clases"])
            
        print(f"Cargando modelos desde {self.model_dir}...")
        self.modelos_emb = []
        for file in os.listdir(self.model_dir):
            if file.startswith("modelo_embeddings_fold") and file.endswith(".txt"):
                path = os.path.join(self.model_dir, file)
                self.modelos_emb.append(lgb.Booster(model_file=path))
                
        if not self.modelos_emb:
            raise ValueError(f"No se encontraron modelos de embeddings (*.txt) en {self.model_dir}")
            
        print(f"✓ Se cargaron {len(self.modelos_emb)} modelos de embeddings (K-Folds).")
        print(f"✓ Clases totales que reconoce el modelo: {len(self.clases)}")

    def _obtener_embeddings(self, audio_path):
        """Extrae el embedding 1D usando el mismo mecanismo que con BirdNET original"""
        with _BIRDNET_LOCK:
            birdnet_config.MODEL_PATH    = birdnet_config.BIRDNET_MODEL_PATH
            birdnet_config.LABELS_FILE   = birdnet_config.BIRDNET_LABELS_FILE
            birdnet_config.SAMPLE_RATE   = birdnet_config.BIRDNET_SAMPLE_RATE
            birdnet_config.SIG_LENGTH    = birdnet_config.BIRDNET_SIG_LENGTH
            birdnet_config.AUDIO_SPEED   = 1.0
            birdnet_config.SIG_OVERLAP   = 0.0
            birdnet_config.BANDPASS_FMIN = birdnet_config.SIG_FMIN
            birdnet_config.BANDPASS_FMAX = birdnet_config.SIG_FMAX
            
            try:
                # Retorna [(path, start_seg, end_seg, vector_embedding), ...]
                resultados = embeddings_utils.analyze_file_core(
                    audio_path, birdnet_config.get_config()
                )
            except Exception as e:
                print(f"Error al extraer embeddings del audio con BirdNET: {e}")
                resultados = []
                
        return resultados

    def predecir_audio(self, audio_path):
        print(f"\nProcesando: {audio_path}")
        resultados = self._obtener_embeddings(audio_path)
        
        if not resultados:
            print("No se encontraron resultados validos ni audios con sonido en el archivo.")
            return

        predicciones_segmentos = []
        
        print("\nExtrayendo predicciones por segmentos de 3 segundos:")
        for _, start, end, emb in resultados:
            # emb suele ser un np.array o lista plana de 1024 o una matriz (1, 1024)
            vec = np.array(emb).flatten().reshape(1, -1)
            
            # Promediar las predicciones sobre todos los modelos (K-folds)
            preds = np.zeros((1, len(self.clases)))
            for modelo in self.modelos_emb:
                preds += modelo.predict(vec)
            preds /= len(self.modelos_emb)
            
            # Quitar la dimension redundante originada por el predict
            probabilidades_clases = preds[0]
            
            idx_max = np.argmax(probabilidades_clases)
            prob_max = probabilidades_clases[idx_max]
            especie = self.clases[idx_max]
            
            predicciones_segmentos.append({
                "start": start,
                "end": end,
                "especie": especie,
                "confianza": prob_max
            })
            
            print(f"  [{start:05.1f}s - {end:05.1f}s] {especie:<30} ({prob_max*100:05.2f}%)")
            
        # Agregación para obtener la especie más probable del archivo entero
        # Sumamos las probabilidades de todos los segmentos y nos quedamos con el máximo
        conteo = {}
        for p in predicciones_segmentos:
            e = p["especie"]
            conteo[e] = conteo.get(e, 0) + p["confianza"]
            
        if conteo:
            top_especie = max(conteo, key=conteo.get)
            print(f"\n{'='*60}")
            print(f"ESPECIE PREDOMINANTE (Basado en todos los segmentos):")
            print(f" ► {top_especie}")
            print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hacer inferencia sobre un audio con el modelo de embeddings")
    parser.add_argument("audio", help="Ruta al archivo de audio (.wav, .mp3, etc.)")
    parser.add_argument("--modelos", default="entrenamiento/datos_fase3", help="Carpeta (relativa a raíz del proyecto) donde se guardó el modelo al entrenar")
    
    args = parser.parse_args()
    
    # Manejar paths por si pasan un archivo que está en la misma carpeta "inferencia"
    ruta_audio = args.audio
    if not os.path.exists(ruta_audio):
        ruta_audio_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.audio)
        if os.path.exists(ruta_audio_local):
            ruta_audio = ruta_audio_local
        else:
            print(f"Error: No se encontró el archivo de audio '{args.audio}'.")
            sys.exit(1)
            
    try:
        predictor = AudioPredictor(model_dir=args.modelos)
        predictor.predecir_audio(ruta_audio)
    except Exception as e:
        print(f"\nError de inferencia: {e}")