import requests
import os

API_KEY = ""


def buscar_aves(query, pagina=1):
    r = requests.get(
        "https://xeno-canto.org/api/3/recordings",
        params={"query": query, "key": API_KEY, "page": pagina}
    )
    return r.json()

def descargar_audio(recording, carpeta="audios"):
    os.makedirs(carpeta, exist_ok=True)
    
    url = f"https://xeno-canto.org/{recording['id']}/download"
    nombre = f"{recording['id']}_{recording['gen']}_{recording['sp']}.mp3"
    ruta = os.path.join(carpeta, nombre)
    
    # Stream para no cargar todo en memoria
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(ruta, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"Descargado: {nombre}")
    return ruta

# --- Uso ---
data = buscar_aves("cnt:Colombia grp:birds q:'>C'")

for rec in data["recordings"]:
    descargar_audio(rec)