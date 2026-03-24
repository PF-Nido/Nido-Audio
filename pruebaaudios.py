"""
Descarga un solo audio de Xeno-canto por su XC ID.

Requisitos:
    pip install requests

Uso:
    python descargar_un_audio_xc.py --xc_id 1069623 --api_key TU_API_KEY
    python descargar_un_audio_xc.py --xc_id 1069623 --api_key TU_API_KEY --output_dir audios/

Desde Python:
    from descargar_un_audio_xc import descargar_audio_xc
    meta, audio_path = descargar_audio_xc(xc_id="1069623", api_key="TU_KEY")
"""

import os
import json
import argparse
import requests
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

XC_API_BASE = "https://xeno-canto.org/api/3/recordings"


# ════════════════════════════════════════════════════════════════════════════
# 1. CONSULTAR METADATOS POR XC ID
# ════════════════════════════════════════════════════════════════════════════

def obtener_metadatos_xc(xc_id: str, api_key: str) -> dict | None:
    """
    Consulta los metadatos de una grabación por su XC ID.

    La API de XC devuelve todos los campos disponibles:
    id, gen, sp, en, cnt, loc, lat, lng, q, type, date, time, file, etc.

    Args:
        xc_id:   ID numérico de la grabación en Xeno-canto (ej. "1069623")
        api_key: Tu API key de xeno-canto.org

    Returns:
        Dict con los metadatos o None si no se encontró.
    """
    params = {
        "query": f"nr:{xc_id}",
        "key": api_key,
    }

    print(f"Consultando metadatos XC:{xc_id}...", flush=True)

    try:
        r = requests.get(XC_API_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if not data.get("recordings"):
            print(f"  ✗ No se encontró ninguna grabación con ID {xc_id}")
            return None

        rec = data["recordings"][0]

        # Extraer los campos más relevantes para el pipeline
        meta = {
            "xc_id":            rec.get("id"),
            "nombre_cientifico": f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
            "nombre_ingles":     rec.get("en"),
            "pais":              rec.get("cnt"),
            "localidad":         rec.get("loc"),
            "lat":               rec.get("lat"),
            "lon":               rec.get("lng") or rec.get("lon"),
            "altura":            rec.get("alt") or rec.get("elev"),
            "fecha":             rec.get("date"),
            "hora":              rec.get("time"),
            "calidad_xc":        rec.get("q"),       # A, B, C, D, E
            "tipo_canto":        rec.get("type"),     # song, call, alarm...
            "duracion":          rec.get("length"),
            "url_audio":         rec.get("file"),     # URL de descarga
            "url_pagina":        rec.get("url"),
            "recordista":        rec.get("rec"),
            "licencia":          rec.get("lic"),
        }

        print(f"  ✓ {meta['nombre_cientifico']} | Calidad: {meta['calidad_xc']} | País: {meta['pais']}")
        return meta

    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            print("  ✗ API key inválida o expirada. Verifica tu clave en xeno-canto.org")
        elif r.status_code == 429:
            print("  ✗ Límite de requests alcanzado. Espera unos minutos.")
        else:
            print(f"  ✗ Error HTTP {r.status_code}: {e}")
        return None

    except Exception as e:
        print(f"  ✗ Error consultando API: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════
# 2. DESCARGAR EL AUDIO
# ════════════════════════════════════════════════════════════════════════════

def descargar_audio(url_audio: str, output_path: str, api_key: str) -> bool:
    """
    Descarga el archivo de audio desde la URL de Xeno-canto.

    Args:
        url_audio:   URL directa al archivo (campo 'file' de la API)
        output_path: Ruta local donde guardar el archivo
        api_key:     API key (requerida desde Oct 2025)

    Returns:
        True si la descarga fue exitosa, False si falló.
    """
    print(f"Descargando audio → {output_path}...", flush=True)

    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        r = requests.get(url_audio, headers=headers, stream=True, timeout=60)
        r.raise_for_status()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Descargar en chunks para no cargar todo en memoria
        total = int(r.headers.get("content-length", 0))
        descargado = 0

        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                descargado += len(chunk)

        size_kb = descargado / 1024
        print(f"  ✓ Descargado: {size_kb:.1f} KB")

        # Validar que el archivo no esté vacío (archivos corruptos pesan < 5KB)
        if descargado < 5_000:
            print(f"  ✗ Archivo sospechosamente pequeño ({size_kb:.1f} KB), puede estar corrupto")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error descargando audio: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════
# 3. FUNCIÓN PRINCIPAL: metadatos + descarga en un solo paso
# ════════════════════════════════════════════════════════════════════════════

def descargar_audio_xc(
    xc_id: str,
    api_key: str,
    output_dir: str = "audios/",
    guardar_json: bool = True,
) -> tuple[dict | None, str | None]:
    """
    Descarga un audio de Xeno-canto junto con sus metadatos.

    Args:
        xc_id:        ID de la grabación en XC (ej. "1069623")
        api_key:      Tu API key de xeno-canto.org
        output_dir:   Carpeta donde guardar el audio y el JSON
        guardar_json: Si True, guarda los metadatos como JSON sidecar

    Returns:
        (meta, audio_path) si fue exitoso
        (None, None)       si falló
    """
    # Paso 1: obtener metadatos
    meta = obtener_metadatos_xc(xc_id, api_key)
    if meta is None:
        return None, None

    if not meta.get("url_audio"):
        print("  ✗ Esta grabación no tiene URL de audio disponible")
        return None, None

    # Paso 2: construir nombre del archivo
    # Formato: {XC_ID}_{Genero}_{especie}.mp3
    nombre_cientifico_snake = meta["nombre_cientifico"].replace(" ", "_")
    extension = Path(meta["url_audio"]).suffix or ".mp3"
    nombre_archivo = f"{xc_id}_{nombre_cientifico_snake}{extension}"
    audio_path = str(Path(output_dir) / nombre_archivo)

    # Paso 3: descargar audio
    exito = descargar_audio(meta["url_audio"], audio_path, api_key)
    if not exito:
        return meta, None

    # Paso 4: guardar JSON sidecar con metadatos (opcional)
    if guardar_json:
        json_path = audio_path.replace(extension, ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Metadatos guardados → {json_path}")

    print(f"\n✓ Listo: {nombre_archivo}")
    print(f"  Especie:  {meta['nombre_cientifico']}")
    print(f"  Calidad:  {meta['calidad_xc']}")
    print(f"  País:     {meta['pais']}")
    print(f"  Lat/Lon:  {meta['lat']}, {meta['lon']}")
    if meta.get('altura'):
        print(f"  Altura:   {meta['altura']}")
    print(f"  Fecha:    {meta['fecha']} {meta['hora']}")

    return meta, audio_path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga un audio de Xeno-canto por su XC ID"
    )
    parser.add_argument("--xc_id",      required=True,  help="ID de la grabación (ej. 1069623)")
    parser.add_argument("--api_key",    required=True,  help="Tu API key de xeno-canto.org")
    parser.add_argument("--output_dir", default="audios/", help="Carpeta de salida (default: audios/)")
    parser.add_argument("--no_json",    action="store_true", help="No guardar JSON sidecar")
    args = parser.parse_args()

    meta, audio_path = descargar_audio_xc(
        xc_id=args.xc_id,
        api_key=args.api_key,
        output_dir=args.output_dir,
        guardar_json=not args.no_json,
    )

    if audio_path:
        print(f"\nAhora puedes procesar este audio con:")
        print(f"  python extraer_un_embedding.py  (apunta a: {audio_path})")
    else:
        print("\nLa descarga falló. Revisa tu API key o el XC ID.")