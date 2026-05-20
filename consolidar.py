import os
import sys
from pathlib import Path
sys.path.append(str(Path("analisis_datos").resolve()))
from descargar_xc import _consolidar_embeddings

_consolidar_embeddings("analisis_datos/datos_fase1", "parquet")
print("Hecho")
