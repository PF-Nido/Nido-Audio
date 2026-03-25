"""
Busca y descarga N audios de Xeno-canto usando una query personalizada,
luego extrae sus embeddings con BirdNET.

Requisitos:
    pip install requests pandas

Uso:
    python buscar_y_descargar_xc.py --api_key TU_KEY
    python buscar_y_descargar_xc.py --api_key TU_KEY --limit 10
    python buscar_y_descargar_xc.py --api_key TU_KEY --limit 50 --query "cnt:Colombia grp:birds q:A"

Desde Python:
    from buscar_y_descargar_xc import buscar_y_procesar
    resultados = buscar_y_procesar(api_key="TU_KEY", limit=10)
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path

# ── Importar los módulos ya existentes en el repo ───────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pruebaaudios import descargar_audio_xc
from extraer_un_embeddingv2 import extract_embeddings_from_audio


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN POR DEFECTO
# ════════════════════════════════════════════════════════════════════════════

XC_API_BASE   = "https://xeno-canto.org/api/3/recordings"
QUERY_DEFAULT = "cnt:Colombia grp:birds q_gt:C"   # solo calidad A, B o C


# ════════════════════════════════════════════════════════════════════════════
# 1. BUSCAR GRABACIONES
# ════════════════════════════════════════════════════════════════════════════

def buscar_grabaciones_xc(query: str, api_key: str, limit: int = 10) -> list[dict]:
    """
    Busca grabaciones en Xeno-canto con la query dada y retorna
    los primeros `limit` resultados.

    Args:
        query:   Query de XC (ej. "cnt:Colombia grp:birds q:>C")
        api_key: Tu API key de xeno-canto.org
        limit:   Máximo de grabaciones a retornar

    Returns:
        Lista de dicts con los metadatos crudos de XC.
    """
    print(f'Buscando en Xeno-canto: "{query}"')
    print(f"Límite: {limit} grabaciones\n")

    params = {
        "query": query,
        "key":   api_key,
        "page":  1,
    }

    try:
        r = requests.get(XC_API_BASE, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        total_disponible = int(data.get("numRecordings", 0))
        print(f"Total disponible en XC para esta query: {total_disponible:,}")

        grabaciones = data.get("recordings", [])

        # Si hay más páginas y necesitamos más resultados, seguir paginando
        pagina_actual = 1
        while len(grabaciones) < limit and pagina_actual < int(data.get("numPages", 1)):
            pagina_actual += 1
            params["page"] = pagina_actual
            r = requests.get(XC_API_BASE, params=params, timeout=30)
            r.raise_for_status()
            grabaciones.extend(r.json().get("recordings", []))

        # Recortar al límite pedido
        grabaciones = grabaciones[:limit]
        print(f"Grabaciones seleccionadas: {len(grabaciones)}\n")
        return grabaciones

    except requests.exceptions.HTTPError as e:
        if r.status_code == 401:
            print("✗ API key inválida o expirada.")
        else:
            print(f"✗ Error HTTP {r.status_code}: {e}")
        return []

    except Exception as e:
        print(f"✗ Error en la búsqueda: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# 2. PROCESAR CADA GRABACIÓN: descarga + embedding
# ════════════════════════════════════════════════════════════════════════════

def procesar_grabacion(
    rec: dict,
    api_key: str,
    output_dir: str,
    umbral_confianza: float = 0.7,
    delete_audio_after: bool = True,
) -> dict | None:
    """
    Dado un registro crudo de la API de XC:
      1. Descarga el audio
      2. Extrae embeddings con BirdNET (pasando especie esperada y umbral)
      3. Borra el audio si delete_audio_after=True

    Retorna un dict resumen del resultado, o None si falló.
    """
    xc_id = rec.get("id")

    # Paso 1: descargar audio + JSON sidecar
    meta, audio_path = descargar_audio_xc(
        xc_id=xc_id,
        api_key=api_key,
        output_dir=output_dir,
        guardar_json=True,
    )

    if audio_path is None:
        return {
            "xc_id":     xc_id,
            "especie":   f"{rec.get('gen','')} {rec.get('sp','')}".strip(),
            "estado":    "error_descarga",
            "segmentos": 0,
            "csv_path":  None,
        }

    # Paso 2: extraer embeddings — pasando especie y umbral correctamente
    especie_esperada = meta.get("nombre_cientifico", "")
    csv_path = extract_embeddings_from_audio(
        audio_path=audio_path,
        especie_esperada=especie_esperada,
        umbral_confianza=umbral_confianza,
    )

    # Paso 3: borrar audio después de procesar (ahorra disco)
    if delete_audio_after and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            print(f"  🗑  Audio eliminado: {os.path.basename(audio_path)}")
        except Exception as e:
            print(f"  ⚠  No se pudo borrar el audio: {e}")

    if csv_path is None:
        return {
            "xc_id":     xc_id,
            "especie":   especie_esperada,
            "calidad":   meta.get("calidad_xc"),
            "estado":    "error_embedding",
            "segmentos": 0,
            "csv_path":  None,
        }

    # Contar segmentos extraídos
    try:
        df_emb = pd.read_csv(csv_path)
        n_segmentos = len(df_emb)
        n_confiables = (df_emb["confiabilidad"] == "confiable").sum()
    except Exception:
        n_segmentos  = -1
        n_confiables = 0

    return {
        "xc_id":        xc_id,
        "especie":      especie_esperada,
        "calidad":      meta.get("calidad_xc"),
        "pais":         meta.get("pais"),
        "lat":          meta.get("lat"),
        "lon":          meta.get("lon"),
        "altura":       meta.get("altura"),
        "fecha":        meta.get("fecha"),
        "hora":         meta.get("hora"),
        "tipo_canto":   meta.get("tipo_canto"),
        "estado":       "ok",
        "segmentos":    n_segmentos,
        "confiables":   n_confiables,
        "csv_path":     csv_path,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════════════

def buscar_y_procesar(
    api_key:            str,
    query:              str   = QUERY_DEFAULT,
    limit:              int   = 10,
    output_dir:         str   = "audios/",
    umbral_confianza:   float = 0.7,
    delete_audio_after: bool  = True,       # ← True por defecto: ahorra disco
) -> pd.DataFrame:
    """
    Pipeline completo: busca → descarga → extrae embeddings → borra audio.

    Args:
        api_key:            Tu API key de xeno-canto.org
        query:              Query de búsqueda de XC
        limit:              Número de grabaciones a procesar
        output_dir:         Carpeta donde guardar CSVs y JSONs
        umbral_confianza:   Score mínimo para marcar un segmento como confiable
        delete_audio_after: Si True (default), borra el audio tras extraer embeddings

    Returns:
        DataFrame con el resumen de todos los audios procesados.
    """
    # Paso 1: buscar
    grabaciones = buscar_grabaciones_xc(query, api_key, limit)
    if not grabaciones:
        print("No se encontraron grabaciones.")
        return pd.DataFrame()

    # Paso 2: procesar cada una
    resultados = []
    total = len(grabaciones)

    for i, rec in enumerate(grabaciones, 1):
        xc_id   = rec.get("id", "?")
        especie = f"{rec.get('gen','')} {rec.get('sp','')}".strip()
        print(f"\n{'─'*60}")
        print(f"[{i}/{total}] XC:{xc_id} — {especie}")
        print(f"{'─'*60}")

        resultado = procesar_grabacion(
            rec=rec,
            api_key=api_key,
            output_dir=output_dir,
            umbral_confianza=umbral_confianza,
            delete_audio_after=delete_audio_after,
        )

        if resultado:
            resultados.append(resultado)

    # Paso 3: resumen final
    df = pd.DataFrame(resultados)
    print(f"\n{'═'*60}")
    print(f"RESUMEN FINAL")
    print(f"{'═'*60}")
    print(f"  Total procesados:    {len(df)}")
    if len(df) > 0:
        ok  = (df["estado"] == "ok").sum()
        err = (df["estado"] != "ok").sum()
        segs       = df["segmentos"].sum()
        confiables = df.get("confiables", pd.Series([0])).sum()
        print(f"  Exitosos:            {ok}")
        print(f"  Con errores:         {err}")
        print(f"  Segmentos totales:   {segs}")
        print(f"  Confiables (≥{umbral_confianza}):   {confiables}  ({100*confiables/segs:.0f}%)" if segs > 0 else "")
        print(f"\nEspecies procesadas:")
        for _, row in df[df["estado"] == "ok"].iterrows():
            print(f"  XC:{row['xc_id']:>8} | {row.get('calidad','?')} | {row['especie']}")

    # Guardar resumen como CSV
    resumen_path = Path(output_dir) / "resumen_descarga.csv"
    df.to_csv(resumen_path, index=False)
    print(f"\nResumen guardado en: {resumen_path}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Busca y descarga audios de Xeno-canto + extrae embeddings + borra audio"
    )
    parser.add_argument("--api_key",    required=True,
                        help="Tu API key de xeno-canto.org")
    parser.add_argument("--query",      default=QUERY_DEFAULT,
                        help=f'Query de búsqueda (default: "{QUERY_DEFAULT}")')
    parser.add_argument("--limit",      type=int, default=10,
                        help="Número de grabaciones a descargar (default: 10)")
    parser.add_argument("--output_dir", default="audios/",
                        help="Carpeta de salida para CSVs y JSONs (default: audios/)")
    parser.add_argument("--umbral",     type=float, default=0.7,
                        help="Umbral de confianza BirdNET (default: 0.7)")
    parser.add_argument("--keep_audio", action="store_true",
                        help="Conservar el audio después de procesar (por defecto se borra)")
    args = parser.parse_args()

    df_resultados = buscar_y_procesar(
        api_key=args.api_key,
        query=args.query,
        limit=args.limit,
        output_dir=args.output_dir,
        umbral_confianza=args.umbral,
        delete_audio_after=not args.keep_audio,
    )