"""
buscar_y_descargar_xc_v2.py — Fase 1
──────────────────────────────────────
Pipeline completo:
  1. Busca grabaciones en XC (metadatos incluidos, sin requests extra)
  2. Descarga audio directo
  3. Extrae TODOS los embeddings (sin filtros de score)
  4. Genera 2 archivos consolidados:
     - embeddings_consolidado → modelo de embeddings
     - metadata_consolidado   → modelo de metadata
  5. Borra audios

Uso:
    python buscar_y_descargar_xc_v2.py --api_key TU_KEY --limit 100
    python buscar_y_descargar_xc_v2.py --api_key TU_KEY --limit 5000 --workers 3 --formato parquet
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pruebaaudiosv2 import descargar_audio_desde_registro
from extraer_embeddings import extract_embeddings_from_audio


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

XC_API_BASE   = "https://xeno-canto.org/api/3/recordings"
QUERY_DEFAULT = "cnt:Colombia grp:birds q_gt:C"


# ════════════════════════════════════════════════════════════════════════════
# 1. BUSCAR GRABACIONES (metadatos incluidos)
# ════════════════════════════════════════════════════════════════════════════

def buscar_grabaciones_xc(query: str, api_key: str, limit: int = 10) -> list[dict]:
    """Busca grabaciones en XC. Cada resultado YA incluye todos los metadatos."""
    print(f'Buscando en Xeno-canto: "{query}"')
    print(f"Límite: {limit}\n")

    params = {"query": query, "key": api_key, "page": 1}

    try:
        r = requests.get(XC_API_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        total = int(data.get("numRecordings", 0))
        num_pages = int(data.get("numPages", 1))
        print(f"Total disponible: {total:,} ({num_pages} páginas)")

        grabaciones = data.get("recordings", [])

        pagina = 1
        while len(grabaciones) < limit and pagina < num_pages:
            pagina += 1
            params["page"] = pagina
            r = requests.get(XC_API_BASE, params=params, timeout=30)
            r.raise_for_status()
            grabaciones.extend(r.json().get("recordings", []))
            if pagina % 10 == 0:
                print(f"  Página {pagina}/{num_pages} — "
                      f"{len(grabaciones)} acumuladas")

        grabaciones = grabaciones[:limit]
        print(f"Seleccionadas: {len(grabaciones)}\n")
        return grabaciones

    except requests.exceptions.HTTPError:
        if r.status_code == 401:
            print("✗ API key inválida.")
        else:
            print(f"✗ Error HTTP {r.status_code}")
        return []
    except Exception as e:
        print(f"✗ Error: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# 2. PROCESAR UNA GRABACIÓN
# ════════════════════════════════════════════════════════════════════════════

def _safe_float(val) -> float | None:
    """Convierte a float de forma segura."""
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _extraer_metadata(rec: dict) -> dict:
    """Extrae metadatos normalizados de un registro de búsqueda."""
    return {
        "xc_id":              rec.get("id"),
        "genero":             rec.get("gen", "").strip(),
        "especie":            rec.get("sp", "").strip(),
        "subespecie":         rec.get("ssp", "").strip(),
        "nombre_cientifico":  f"{rec.get('gen', '')} {rec.get('sp', '')}".strip(),
        "nombre_ingles":      rec.get("en", ""),
        "grupo":              rec.get("group", ""),
        "pais":               rec.get("cnt", ""),
        "localidad":          rec.get("loc", ""),
        "lat":                _safe_float(rec.get("lat")),
        "lon":                _safe_float(rec.get("lng")),
        "altura":             _safe_float(rec.get("alt")),
        "fecha":              rec.get("date", ""),
        "hora":               rec.get("time", ""),
        "calidad_xc":         rec.get("q", ""),
        "tipo_canto":         rec.get("type", ""),
        "duracion":           rec.get("length", ""),
        "sexo":               rec.get("sex", ""),
        "etapa_vida":         rec.get("stage", ""),
        "metodo_grabacion":   rec.get("method", ""),
        "playback_usado":     rec.get("playback-used", ""),
        "ave_vista":          rec.get("bird-seen", ""),
        "temperatura":        rec.get("temp", ""),
        "recordista":         rec.get("rec", ""),
        "comentarios":        rec.get("rmk", ""),
        "otras_especies":     rec.get("also", ""),
        "url_audio":          rec.get("file", ""),
        "url_pagina":         rec.get("url", ""),
        "licencia":           rec.get("lic", ""),
    }


def _procesar_una_grabacion(
    rec: dict,
    api_key: str,
    output_dir: str,
    delete_audio_after: bool,
    formato: str,
) -> dict | None:
    """
    Procesa una grabación:
      1. Descarga audio (sin request extra de metadatos)
      2. Extrae TODOS los embeddings (sin filtros de score)
      3. Borra audio
    """
    meta = _extraer_metadata(rec)
    xc_id = meta["xc_id"]
    especie = meta["nombre_cientifico"]
    otras = meta.get("otras_especies", "")

    # Formatear secondary labels como string
    if isinstance(otras, list):
        otras = ", ".join(otras)
    elif not isinstance(otras, str):
        otras = str(otras) if otras else ""

    # ── 1. Descargar ─────────────────────────────────────────────────
    meta_dl, audio_path = descargar_audio_desde_registro(
        rec=rec, api_key=api_key, output_dir=output_dir, guardar_json=True,
    )

    if audio_path is None:
        return {**meta, "estado": "error_descarga",
                "segmentos_totales": 0, "segmentos_guardados": 0,
                "embeddings_path": None}

    # ── 2. Extraer TODOS los embeddings ──────────────────────────────
    ext = ".parquet" if formato == "parquet" else ".csv"
    extension = Path(audio_path).suffix
    emb_path = audio_path.replace(extension, f"_embeddings{ext}")

    resultado_path = extract_embeddings_from_audio(
        audio_path=audio_path,
        output_path=emb_path,
        especie_esperada=especie,
        otras_especies=otras,
        formato=formato,
    )

    # ── 3. Borrar audio ─────────────────────────────────────────────
    if delete_audio_after and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            print(f"  🗑️ Audio eliminado")
        except Exception as e:
            print(f"  ⚠ No se pudo borrar: {e}")

    # ── 4. Contar resultados ─────────────────────────────────────────
    n_total = 0
    n_guardados = 0
    if resultado_path and os.path.exists(resultado_path):
        try:
            df = (pd.read_parquet(resultado_path) if formato == "parquet"
                  else pd.read_csv(resultado_path))
            n_guardados = len(df)
            n_total = n_guardados  # ya no filtramos, son los mismos
        except Exception:
            pass

    return {
        **meta,
        "estado":              "ok" if resultado_path else "error_embedding",
        "segmentos_totales":   n_total,
        "segmentos_guardados": n_guardados,
        "embeddings_path":     resultado_path,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. CONSOLIDAR ARCHIVOS PARA LOS 2 MODELOS
# ════════════════════════════════════════════════════════════════════════════

def _consolidar_embeddings(output_dir: str, formato: str = "csv") -> str | None:
    """
    Une todos los _embeddings en UN archivo.
    Para el modelo de embeddings (LightGBM #1).
    """
    ext = ".parquet" if formato == "parquet" else ".csv"
    archivos = list(Path(output_dir).glob(f"*_embeddings{ext}"))

    if not archivos:
        print("  No hay archivos de embeddings.")
        return None

    print(f"\n  Consolidando {len(archivos)} archivos de embeddings...")
    frames = []
    vacios = 0

    for f in archivos:
        try:
            if f.stat().st_size < 50:
                vacios += 1
                continue

            df = pd.read_parquet(f) if formato == "parquet" else pd.read_csv(f)
            if len(df) == 0:
                vacios += 1
                continue

            frames.append(df)
        except Exception as e:
            print(f"    ⚠ Error leyendo {f.name}: {e}")
            vacios += 1

    if vacios > 0:
        print(f"    ℹ️ {vacios} archivos vacíos ignorados")

    if not frames:
        print("    ✗ Sin datos")
        return None

    df_all = pd.concat(frames, ignore_index=True)

    salida = str(Path(output_dir) / f"embeddings_consolidado{ext}")
    if formato == "parquet":
        df_all.to_parquet(salida, index=False)
    else:
        df_all.to_csv(salida, index=False)

    print(f"    ✓ {len(df_all)} segmentos → {salida}")
    return salida


def _consolidar_metadata(resultados: list[dict], output_dir: str,
                         formato: str = "csv") -> str | None:
    """
    Genera archivo de metadata (una fila por audio).
    Para el modelo de metadata (LightGBM #2).
    """
    ext = ".parquet" if formato == "parquet" else ".csv"
    df = pd.DataFrame(resultados)

    cols = [
        "xc_id", "genero", "especie", "nombre_cientifico",
        "pais", "localidad", "lat", "lon", "altura",
        "fecha", "hora", "calidad_xc", "tipo_canto", "duracion",
        "sexo", "etapa_vida", "metodo_grabacion", "playback_usado",
        "ave_vista", "temperatura", "otras_especies",
        "recordista", "licencia",
        "estado", "segmentos_totales", "segmentos_guardados",
    ]
    cols = [c for c in cols if c in df.columns]

    salida = str(Path(output_dir) / f"metadata_consolidado{ext}")
    if formato == "parquet":
        df[cols].to_parquet(salida, index=False)
    else:
        df[cols].to_csv(salida, index=False)

    print(f"    ✓ {len(df)} audios → {salida}")
    return salida


# ════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════════════

def buscar_y_procesar(
    api_key: str,
    query: str = QUERY_DEFAULT,
    limit: int = 10,
    output_dir: str = "datos/",
    delete_audio_after: bool = True,
    formato: str = "csv",
    max_workers: int = 1,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """
    Pipeline Fase 1:
      1. Busca grabaciones (metadatos incluidos)
      2. Descarga + extrae TODOS los embeddings
      3. Consolida en 2 archivos separados

    Returns:
        (df_resumen, path_embeddings, path_metadata)
    """
    # ── 1. Buscar ────────────────────────────────────────────────────
    grabaciones = buscar_grabaciones_xc(query, api_key, limit)
    if not grabaciones:
        return pd.DataFrame(), None, None

    total = len(grabaciones)
    resultados = []

    kwargs = dict(
        api_key=api_key, output_dir=output_dir,
        delete_audio_after=delete_audio_after,
        formato=formato,
    )

    # ── 2. Procesar ──────────────────────────────────────────────────
    if max_workers <= 1:
        for i, rec in enumerate(grabaciones, 1):
            xc_id = rec.get("id", "?")
            especie = f"{rec.get('gen', '')} {rec.get('sp', '')}".strip()
            print(f"\n{'─' * 60}")
            print(f"[{i}/{total}] XC:{xc_id} — {especie}")
            print(f"{'─' * 60}")

            resultado = _procesar_una_grabacion(rec=rec, **kwargs)
            if resultado:
                resultados.append(resultado)
    else:
        print(f"\n🚀 Procesando con {max_workers} workers...\n")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futuros = {
                pool.submit(_procesar_una_grabacion, rec=rec, **kwargs): rec
                for rec in grabaciones
            }

            for i, futuro in enumerate(as_completed(futuros), 1):
                rec = futuros[futuro]
                xc_id = rec.get("id", "?")
                especie = f"{rec.get('gen', '')} {rec.get('sp', '')}".strip()
                try:
                    resultado = futuro.result()
                    if resultado:
                        resultados.append(resultado)
                    estado = resultado.get("estado", "?") if resultado else "?"
                    print(f"[{i}/{total}] XC:{xc_id} {especie} → {estado}")
                except Exception as e:
                    print(f"[{i}/{total}] XC:{xc_id} → ERROR: {e}")

    if not resultados:
        print("No se procesó ninguna grabación.")
        return pd.DataFrame(), None, None

    # ── 3. Consolidar ────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("CONSOLIDANDO PARA LOS 2 MODELOS")
    print(f"{'═' * 60}")

    path_emb = _consolidar_embeddings(output_dir, formato)
    path_meta = _consolidar_metadata(resultados, output_dir, formato)

    # ── 4. Resumen ───────────────────────────────────────────────────
    df = pd.DataFrame(resultados)
    ok = (df["estado"] == "ok").sum()
    err = (df["estado"] != "ok").sum()
    segs = df["segmentos_guardados"].sum()
    n_especies = 0
    if "nombre_cientifico" in df.columns:
        n_especies = df[df["estado"] == "ok"]["nombre_cientifico"].nunique()

    print(f"\n{'═' * 60}")
    print(f"RESUMEN FINAL — FASE 1")
    print(f"{'═' * 60}")
    print(f"  Audios descargados:    {len(df)}")
    print(f"  Exitosos:              {ok}")
    print(f"  Con errores:           {err}")
    print(f"  Especies únicas:       {n_especies}")
    print(f"  Segmentos guardados:   {segs}")
    print(f"\n  📁 Embeddings: {path_emb}")
    print(f"  📁 Metadata:   {path_meta}")
    print(f"{'═' * 60}\n")

    return df, path_emb, path_meta


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 1: XC → TODOS los embeddings + metadata"
    )
    parser.add_argument("--api_key",    required=True)
    parser.add_argument("--query",      default=QUERY_DEFAULT)
    parser.add_argument("--limit",      type=int, default=10)
    parser.add_argument("--output_dir", default="datos/")
    parser.add_argument("--formato",    choices=["csv", "parquet"], default="csv")
    parser.add_argument("--workers",    type=int, default=1,
                        help="1=secuencial (seguro), 3+=paralelo")
    parser.add_argument("--keep_audio", action="store_true")
    args = parser.parse_args()

    buscar_y_procesar(
        api_key=args.api_key,
        query=args.query,
        limit=args.limit,
        output_dir=args.output_dir,
        delete_audio_after=not args.keep_audio,
        formato=args.formato,
        max_workers=args.workers,
    )