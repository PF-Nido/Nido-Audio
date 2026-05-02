"""
pruebaaudios.py  (v2)
─────────────────────
Descarga un audio de Xeno-canto.

Dos modos de uso:
  A) Desde el pipeline (sin request extra): se pasan los metadatos del registro de búsqueda
  B) Standalone (por XC ID): se consulta la API una sola vez

Requisitos:
    pip install requests

Uso standalone:
    python pruebaaudios.py --xc_id 1069623 --api_key TU_API_KEY
    python pruebaaudios.py --xc_id 1069623 --api_key TU_API_KEY --output_dir audios/

Desde el pipeline (buscar_y_descargar_xc_v2.py):
    from pruebaaudios import descargar_audio_xc, descargar_audio_desde_registro

    # Modo pipeline: sin request extra
    meta, audio_path = descargar_audio_desde_registro(rec=registro, api_key="KEY")

    # Modo standalone: consulta API por XC ID
    meta, audio_path = descargar_audio_xc(xc_id="1069623", api_key="KEY")
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
# UTILIDADES
# ════════════════════════════════════════════════════════════════════════════

def _safe_float(val) -> float | None:
    """Convierte a float de forma segura. Retorna None si no se puede."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extraer_meta_de_registro(rec: dict) -> dict:
    """
    Extrae metadatos normalizados de un registro crudo de la API de XC.
    Funciona tanto con registros de búsqueda como con registros individuales.

    El registro crudo de XC tiene campos como:
        id, gen, sp, ssp, en, cnt, loc, lat, lng, alt,
        q, type, date, time, file, url, rec, lic, length,
        sex, stage, method, playback-used, bird-seen, temp,
        also, rmk, group, sono, osci, ...

    Esta función los normaliza a nombres legibles en español.
    """
    return {
        # ── Identificación ───────────────────────────────────────────
        "xc_id":              rec.get("id"),
        "genero":             rec.get("gen", "").strip(),
        "especie":            rec.get("sp", "").strip(),
        "subespecie":         rec.get("ssp", "").strip(),
        "nombre_cientifico":  f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
        "nombre_ingles":      rec.get("en", ""),
        "grupo":              rec.get("group", ""),

        # ── Ubicación ────────────────────────────────────────────────
        "pais":               rec.get("cnt", ""),
        "localidad":          rec.get("loc", ""),
        "lat":                _safe_float(rec.get("lat")),
        "lon":                _safe_float(rec.get("lng")),
        "altura":             _safe_float(rec.get("alt")),

        # ── Temporal ─────────────────────────────────────────────────
        "fecha":              rec.get("date", ""),
        "hora":               rec.get("time", ""),

        # ── Calidad y tipo ───────────────────────────────────────────
        "calidad_xc":         rec.get("q", ""),
        "tipo_canto":         rec.get("type", ""),
        "duracion":           rec.get("length", ""),
        "sexo":               rec.get("sex", ""),
        "etapa_vida":         rec.get("stage", ""),

        # ── Contexto de grabación ────────────────────────────────────
        "metodo_grabacion":   rec.get("method", ""),
        "playback_usado":     rec.get("playback-used", ""),
        "ave_vista":          rec.get("bird-seen", ""),
        "temperatura":        rec.get("temp", ""),
        "recordista":         rec.get("rec", ""),
        "comentarios":        rec.get("rmk", ""),
        "otras_especies":     rec.get("also", ""),

        # ── URLs y licencia ──────────────────────────────────────────
        "url_audio":          rec.get("file", ""),
        "url_pagina":         rec.get("url", ""),
        "licencia":           rec.get("lic", ""),
    }


# ════════════════════════════════════════════════════════════════════════════
# 1. CONSULTAR METADATOS POR XC ID (solo para uso standalone)
# ════════════════════════════════════════════════════════════════════════════

def obtener_metadatos_xc(xc_id: str, api_key: str) -> dict | None:
    """
    Consulta los metadatos de una grabación por su XC ID.

    ⚠️  Esta función hace UN request a la API.
        Si ya tienes el registro de búsqueda, usa _extraer_meta_de_registro()
        directamente para evitar requests redundantes.

    Args:
        xc_id:   ID numérico de la grabación en Xeno-canto (ej. "1069623")
        api_key: Tu API key de xeno-canto.org

    Returns:
        Dict con los metadatos normalizados, o None si no se encontró.
    """
    params = {
        "query": f"nr:{xc_id}",
        "key":   api_key,
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
        meta = _extraer_meta_de_registro(rec)

        print(f"  ✓ {meta['nombre_cientifico']} | "
              f"Calidad: {meta['calidad_xc']} | País: {meta['pais']}")
        return meta

    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            print("  ✗ API key inválida o expirada. "
                  "Verifica tu clave en xeno-canto.org")
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

def _descargar_archivo(url_audio: str, output_path: str, api_key: str) -> bool:
    """
    Descarga el archivo de audio desde la URL de Xeno-canto.

    Args:
        url_audio:   URL directa al archivo (campo 'file' de la API)
        output_path: Ruta local donde guardar el archivo
        api_key:     API key (requerida desde Oct 2025)

    Returns:
        True si la descarga fue exitosa, False si falló.
    """
    print(f"  Descargando audio → {output_path}...", flush=True)

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        headers = {"Authorization": f"Bearer {api_key}"}
        r = requests.get(url_audio, headers=headers, stream=True, timeout=60)
        r.raise_for_status()

        descargado = 0
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                descargado += len(chunk)

        size_kb = descargado / 1024
        print(f"  ✓ Descargado: {size_kb:.1f} KB")

        # Validar tamaño mínimo (archivos corruptos/vacíos pesan < 5KB)
        if descargado < 5_000:
            print(f"  ✗ Archivo sospechosamente pequeño "
                  f"({size_kb:.1f} KB), puede estar corrupto")
            # Limpiar archivo corrupto
            if os.path.exists(output_path):
                os.remove(output_path)
            return False

        return True

    except Exception as e:
        print(f"  ✗ Error descargando audio: {e}")
        # Limpiar archivo parcial si existe
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


# ════════════════════════════════════════════════════════════════════════════
# 3. GUARDAR JSON SIDECAR
# ════════════════════════════════════════════════════════════════════════════

def _guardar_json_sidecar(meta: dict, audio_path: str) -> str:
    """Guarda los metadatos como JSON junto al audio. Retorna la ruta."""
    # Reemplazar cualquier extensión de audio por .json
    for ext in [".mp3", ".wav", ".ogg", ".flac"]:
        if audio_path.lower().endswith(ext):
            json_path = audio_path[: -len(ext)] + ".json"
            break
    else:
        json_path = audio_path + ".json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Metadatos guardados → {json_path}")
    return json_path


# ════════════════════════════════════════════════════════════════════════════
# 4. CONSTRUIR RUTA DEL AUDIO
# ════════════════════════════════════════════════════════════════════════════

def _construir_audio_path(meta: dict, output_dir: str) -> str:
    """
    Genera la ruta del archivo de audio a partir de los metadatos.
    Formato: {output_dir}/{xc_id}_{Genero}_{especie}.mp3
    """
    xc_id  = meta["xc_id"]
    genero = meta.get("genero", "").strip()
    sp     = meta.get("especie", "").strip()

    # Si no tenemos genero/especie separados, intentar desde nombre_cientifico
    if not genero and meta.get("nombre_cientifico"):
        partes = meta["nombre_cientifico"].split()
        genero = partes[0] if len(partes) >= 1 else "Unknown"
        sp     = partes[1] if len(partes) >= 2 else "sp"

    extension = Path(meta.get("url_audio", ".mp3")).suffix or ".mp3"
    nombre = f"{xc_id}_{genero}_{sp}{extension}"
    return str(Path(output_dir) / nombre)


# ════════════════════════════════════════════════════════════════════════════
# 5A. MODO PIPELINE: descargar desde registro de búsqueda (SIN request extra)
# ════════════════════════════════════════════════════════════════════════════

def descargar_audio_desde_registro(
    rec: dict,
    api_key: str,
    output_dir: str = "audios/",
    guardar_json: bool = True,
) -> tuple[dict | None, str | None]:
    """
    Descarga un audio usando directamente el registro de búsqueda de XC.
    NO hace ningún request adicional a la API para obtener metadatos.

    Este es el modo que debe usar el pipeline (buscar_y_descargar_xc_v2.py)
    para evitar requests redundantes.

    Args:
        rec:          Registro crudo de la API de XC (un elemento de "recordings")
        api_key:      Tu API key de xeno-canto.org
        output_dir:   Carpeta donde guardar el audio y el JSON
        guardar_json: Si True, guarda los metadatos como JSON sidecar

    Returns:
        (meta, audio_path) si fue exitoso
        (meta, None)       si la descarga falló
        (None, None)       si el registro no tiene URL de audio
    """
    # Extraer metadatos normalizados del registro (sin request)
    meta = _extraer_meta_de_registro(rec)
    xc_id = meta["xc_id"]

    print(f"Procesando XC:{xc_id} — {meta['nombre_cientifico']}", flush=True)

    if not meta.get("url_audio"):
        print(f"  ✗ XC:{xc_id} no tiene URL de audio disponible")
        return None, None

    # Construir ruta y descargar
    audio_path = _construir_audio_path(meta, output_dir)

    exito = _descargar_archivo(meta["url_audio"], audio_path, api_key)
    if not exito:
        return meta, None

    # JSON sidecar
    if guardar_json:
        _guardar_json_sidecar(meta, audio_path)

    _imprimir_resumen(meta, audio_path)
    return meta, audio_path


# ════════════════════════════════════════════════════════════════════════════
# 5B. MODO STANDALONE: descargar por XC ID (hace 1 request para metadatos)
# ════════════════════════════════════════════════════════════════════════════

def descargar_audio_xc(
    xc_id: str,
    api_key: str,
    output_dir: str = "audios/",
    guardar_json: bool = True,
) -> tuple[dict | None, str | None]:
    """
    Descarga un audio de Xeno-canto por su XC ID.
    Hace UN request a la API para obtener metadatos + URL de descarga.

    ⚠️  Si estás procesando muchos audios desde una búsqueda,
        usa descargar_audio_desde_registro() en su lugar para
        evitar requests redundantes.

    Args:
        xc_id:        ID de la grabación en XC (ej. "1069623")
        api_key:      Tu API key de xeno-canto.org
        output_dir:   Carpeta donde guardar el audio y el JSON
        guardar_json: Si True, guarda los metadatos como JSON sidecar

    Returns:
        (meta, audio_path) si fue exitoso
        (meta, None)       si la descarga falló
        (None, None)       si no se encontró el XC ID
    """
    # Paso 1: obtener metadatos (1 request)
    meta = obtener_metadatos_xc(xc_id, api_key)
    if meta is None:
        return None, None

    if not meta.get("url_audio"):
        print("  ✗ Esta grabación no tiene URL de audio disponible")
        return meta, None

    # Paso 2: construir ruta y descargar
    audio_path = _construir_audio_path(meta, output_dir)

    exito = _descargar_archivo(meta["url_audio"], audio_path, api_key)
    if not exito:
        return meta, None

    # Paso 3: JSON sidecar
    if guardar_json:
        _guardar_json_sidecar(meta, audio_path)

    _imprimir_resumen(meta, audio_path)
    return meta, audio_path


# ════════════════════════════════════════════════════════════════════════════
# 6. RESUMEN EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════

def _imprimir_resumen(meta: dict, audio_path: str):
    """Imprime un resumen legible de la descarga completada."""
    nombre = os.path.basename(audio_path)
    print(f"\n  ✓ Listo: {nombre}")
    print(f"    Especie:  {meta['nombre_cientifico']}")
    print(f"    Calidad:  {meta.get('calidad_xc', '?')}")
    print(f"    País:     {meta.get('pais', '?')}")
    print(f"    Lat/Lon:  {meta.get('lat', '?')}, {meta.get('lon', '?')}")
    if meta.get("altura"):
        print(f"    Altura:   {meta['altura']}")
    print(f"    Fecha:    {meta.get('fecha', '?')} {meta.get('hora', '')}")
    if meta.get("tipo_canto"):
        print(f"    Tipo:     {meta['tipo_canto']}")
    if meta.get("otras_especies"):
        print(f"    También:  {meta['otras_especies']}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga un audio de Xeno-canto por su XC ID"
    )
    parser.add_argument("--xc_id",      required=True,
                        help="ID de la grabación (ej. 1069623)")
    parser.add_argument("--api_key",    required=True,
                        help="Tu API key de xeno-canto.org")
    parser.add_argument("--output_dir", default="audios/",
                        help="Carpeta de salida (default: audios/)")
    parser.add_argument("--no_json",    action="store_true",
                        help="No guardar JSON sidecar")
    args = parser.parse_args()

    meta, audio_path = descargar_audio_xc(
        xc_id=args.xc_id,
        api_key=args.api_key,
        output_dir=args.output_dir,
        guardar_json=not args.no_json,
    )

    if audio_path:
        print(f"\nAhora puedes procesar este audio con:")
        print(f"  python extraer_un_embeddingv3.py --audio {audio_path}")
    else:
        print("\nLa descarga falló. Revisa tu API key o el XC ID.")