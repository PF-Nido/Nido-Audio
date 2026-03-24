import os
import sys

# Se añade el directorio actual al path para importar módulos de birdnet_analyzer
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importamos la utlidad interna para que descargue el modelo si falta
from birdnet_analyzer import utils as general_utils
# Usamos core directamente para evitar las integraciones rotas de HopLite DB
from birdnet_analyzer.embeddings import utils as embeddings_utils
from birdnet_analyzer import config as birdnet_config


def extract_embeddings_from_audio(audio_path, output_path=None):
    """
    Extrae los embeddings de un archivo de audio utilizando BirdNET crudo.
    """
    print(f"Extrayendo embeddings de: {audio_path}")
    
    # 1. Asegurarnos que el modelo exista, si no, se descarga automáticamente
    print("Verificando existencia del modelo .tflite...", flush=True)
    general_utils.ensure_model_exists(check_perch=False)

    if output_path is None:
        output_path = audio_path.replace('.wav', '.csv').replace('.mp3', '.csv')

    try:
        # Usamos la configuración de los desarrolladores original
        birdnet_config.MODEL_PATH = birdnet_config.BIRDNET_MODEL_PATH
        birdnet_config.LABELS_FILE = birdnet_config.BIRDNET_LABELS_FILE
        birdnet_config.SAMPLE_RATE = birdnet_config.BIRDNET_SAMPLE_RATE
        birdnet_config.SIG_LENGTH = birdnet_config.BIRDNET_SIG_LENGTH
        
        # Correccion a parametros faltantes que rompen `iterate_audio_chunks`
        birdnet_config.AUDIO_SPEED = 1.0
        birdnet_config.SIG_OVERLAP = 0.0
        birdnet_config.BANDPASS_FMIN = birdnet_config.SIG_FMIN
        birdnet_config.BANDPASS_FMAX = birdnet_config.SIG_FMAX

        # Analizar archivo de forma pura ("core" evita cualquier base de datos sqlite)
        print("Procesando audio por chunks de 3 segundos con la red neuronal...")
        resultados = embeddings_utils.analyze_file_core(audio_path, birdnet_config.get_config())
        
        # resultados es una lista de tuplas: (fpath, s_start, s_end, embeddings)
        if not resultados:
            print("No se extrajo ningún fragmento (Audio quizá muy corto < 3s).")
            return None
            
        print(f"Se extrajeron {len(resultados)} fragmentos de 3 segundos.")
        print(f"Guardando resultados en: {output_path}")
        
        # Escribir manualmente nuestro propio CSV limpio:
        with open(output_path, 'w') as f:
            # Cabecera
            f.write("archivo,inicio_seg,fin_seg,embedding\n")
            
            for fpath, start, end, matriz_embedding in resultados:
                # Cada embedding suele ser un vector muy largo (ej. de forma [1, 1024])
                # Lo aplanamos usandotolist() si es un array de numpy o flatten
                embedding_plano = list(matriz_embedding.flatten())
                # Unir el vector largo usando punto y coma para que no choque con la coma del csv general
                str_embedding = ";".join(map(str, embedding_plano))
                
                f.write(f"{os.path.basename(fpath)},{start},{end},{str_embedding}\n")
                
        print("¡Extracción CRUDA completada con éxito!")
        return output_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error al extraer embeddings: {e}")
        return None

if __name__ == "__main__":
    audio_de_prueba = "audios/1069623_Microsciurus_mimulus.mp3"
    if os.path.exists(audio_de_prueba):
        output_file = extract_embeddings_from_audio(audio_de_prueba)
        if output_file:
            print(f"Revisa el archivo {output_file} para ver los embeddings generados.")
    else:
        print(f"El archivo '{audio_de_prueba}' no existe aún.")
