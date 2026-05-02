"""
buscar_y_descargar_xc_v2.py
────────────────────────────
Pipeline completo con descargas en paralelo y archivos separados
para los 2 modelos (embeddings + metadata).

Optimizaciones:
  • No re-pide metadatos (usa registro de búsqueda directo)
  • Descargas en paralelo (configurable con --workers)
  • Pre-filtro de silencio antes de BirdNET
  • Genera 2 archivos consolidados separados

Uso:
    python buscar_y_descargar_xc_v2.py --api_key TU_KEY --limit 100
    python buscar_y_descargar_xc_v2.py --api_key TU_KEY --limit 5000 --workers 4 --formato parquet
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pruebaaudiosv2 import descargar_audio_desde_registro
from extraer_un_embeddingv4 import extract_embeddings_from_audio


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

XC_API_BASE   = "https://xeno-canto.org/api/3/recordings"
QUERY_DEFAULT = "cnt:Colombia grp:birds q_gt:C"


# ════════════════════════════════════════════════════════════════════════════
# 1. BUSCAR GRABACIONES
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

def _procesar_una_grabacion(
    rec: dict,
    api_key: str,
    output_dir: str,
    umbral_confianza: float,
    umbral_rms: float,
    umbral_snr: float,
    delete_audio_after: bool,
    formato: str,
) -> dict | None:
    """
    Procesa una grabación:
      1. Descarga audio (usando metadatos del registro, sin request extra)
      2. Extrae embeddings filtrados
      3. Borra audio
    """
    xc_id = rec.get("id", "?")
    especie = f"{rec.get('gen', '')} {rec.get('sp', '')}".strip()

    # ── 1. Descargar (sin request extra de metadatos) ────────────────
    meta, audio_path = descargar_audio_desde_registro(
        rec=rec, api_key=api_key, output_dir=output_dir, guardar_json=True,
    )

    if audio_path is None:
        return {
            "xc_id": xc_id, "nombre_cientifico": especie,
            "genero": rec.get("gen", ""), "especie": rec.get("sp", ""),
            "estado": "error_descarga", "segmentos": 0, "confiables": 0,
            "embeddings_path": None,
            **(meta or {}),
        }

    # ── 2. Extraer embeddings (v3: pre-filtro + una pasada) ──────────
    ext_salida = ".parquet" if formato == "parquet" else ".csv"
    extension = Path(audio_path).suffix
    embeddings_path = audio_path.replace(extension, f"_embeddings{ext_salida}")

    csv_path = extract_embeddings_from_audio(
        audio_path=audio_path,
        output_path=embeddings_path,
        especie_esperada=meta.get("nombre_cientifico", especie),
        umbral_confianza=umbral_confianza,
        umbral_rms=umbral_rms,
        umbral_snr=umbral_snr,
        formato=formato,
        solo_entrenamiento=True,
    )

    # ── 3. Borrar audio ─────────────────────────────────────────────
    if delete_audio_after and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
            print(f"  🗑️ Audio eliminado")
        except Exception as e:
            print(f"  ⚠ No se pudo borrar: {e}")

    # ── 4. Contar resultados ─────────────────────────────────────────
    n_segmentos = 0
    n_confiables = 0
    if csv_path and os.path.exists(csv_path):
        try:
            df_emb = (pd.read_parquet(csv_path) if formato == "parquet"
                      else pd.read_csv(csv_path))
            n_segmentos = len(df_emb)
            if "usar_para_entrenamiento" in df_emb.columns:
                n_confiables = int(df_emb["usar_para_entrenamiento"].sum())
        except Exception:
            pass

    return {
        **(meta or {}),
        "estado":          "ok" if csv_path else "error_embedding",
        "segmentos":       n_segmentos,
        "confiables":      n_confiables,
        "embeddings_path": csv_path,
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. CONSOLIDAR ARCHIVOS PARA LOS 2 MODELOS
# ════════════════════════════════════════════════════════════════════════════

def _consolidar_embeddings(output_dir: str, formato: str = "csv") -> str | None:
    """
    Une todos los _embeddings individuales en UN archivo.
    Ignora archivos vacíos o corruptos.
    """
    ext = ".parquet" if formato == "parquet" else ".csv"
    archivos = list(Path(output_dir).glob(f"*_embeddings{ext}"))

    if not archivos:
        print("No hay archivos de embeddings para consolidar.")
        return None

    print(f"\nConsolidando {len(archivos)} archivos de embeddings...")
    frames = []
    vacios = 0
    errores = 0

    for f in archivos:
        try:
            # Saltar archivos vacíos (< 50 bytes = solo headers o nada)
            if f.stat().st_size < 50:
                vacios += 1
                continue

            df = pd.read_parquet(f) if formato == "parquet" else pd.read_csv(f)

            # Saltar DataFrames sin filas
            if len(df) == 0:
                vacios += 1
                continue

            frames.append(df)

        except Exception as e:
            print(f"  ⚠ Error leyendo {f.name}: {e}")
            errores += 1

    if vacios > 0:
        print(f"  ℹ️  {vacios} archivos vacíos ignorados")
    if errores > 0:
        print(f"  ⚠ {errores} archivos con errores")

    if not frames:
        print("  ✗ No hay datos para consolidar")
        return None

    df_all = pd.concat(frames, ignore_index=True)

    # Extraer genero y especie de especie_xc
    if "especie_xc" in df_all.columns:
        df_all["genero"] = df_all["especie_xc"].apply(
            lambda x: x.split()[0] if isinstance(x, str) and x.split() else "")
        df_all["especie"] = df_all["especie_xc"].apply(
            lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) >= 2 else "")

    # xc_id del nombre de archivo
    if "xc_id" not in df_all.columns and "archivo" in df_all.columns:
        df_all["xc_id"] = df_all["archivo"].apply(
            lambda x: x.split("_")[0] if isinstance(x, str) else "")

    cols = ["xc_id", "genero", "especie", "inicio_seg", "fin_seg",
            "especie_detectada", "score", "especie_2", "score_2",
            "energia_rms", "snr_db", "confiabilidad",
            "usar_para_entrenamiento", "embedding"]
    cols = [c for c in cols if c in df_all.columns]

    salida = str(Path(output_dir) / f"embeddings_consolidado{ext}")
    if formato == "parquet":
        df_all[cols].to_parquet(salida, index=False)
    else:
        df_all[cols].to_csv(salida, index=False)

    print(f"  ✓ {len(df_all)} segmentos → {salida}")
    return salida


def _consolidar_metadata(resultados: list[dict], output_dir: str,
                         formato: str = "csv") -> str | None:
    """
    Genera archivo de metadata (una fila por audio).
    Para el modelo de metadata (LightGBM #2).
    """
    ext = ".parquet" if formato == "parquet" else ".csv"
    df = pd.DataFrame(resultados)

    cols = ["xc_id", "genero", "especie", "nombre_cientifico",
            "pais", "localidad", "lat", "lon", "altura",
            "fecha", "hora", "calidad_xc", "tipo_canto", "duracion",
            "sexo", "etapa_vida", "metodo_grabacion", "playback_usado",
            "ave_vista", "temperatura", "otras_especies",
            "recordista", "licencia",
            "estado", "segmentos", "confiables"]
    cols = [c for c in cols if c in df.columns]

    salida = str(Path(output_dir) / f"metadata_consolidado{ext}")
    if formato == "parquet":
        df[cols].to_parquet(salida, index=False)
    else:
        df[cols].to_csv(salida, index=False)

    print(f"  ✓ {len(df)} audios → {salida}")
    return salida


# ════════════════════════════════════════════════════════════════════════════
# 4. PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════════════

def buscar_y_procesar(
    api_key: str,
    query: str = QUERY_DEFAULT,
    limit: int = 10,
    output_dir: str = "audios/",
    umbral_confianza: float = 0.7,
    umbral_rms: float = 0.01,
    umbral_snr: float = 5.0,
    delete_audio_after: bool = True,
    formato: str = "csv",
    max_workers: int = 1,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """
    Pipeline completo:
      1. Busca grabaciones (metadatos incluidos, sin requests extra)
      2. Descarga + extrae embeddings (secuencial o paralelo)
      3. Consolida en 2 archivos:
         - embeddings_consolidado → modelo de embeddings
         - metadata_consolidado   → modelo de metadata

    Args:
        max_workers: 1 = secuencial (seguro), 3-5 = paralelo (rápido)

    Returns:
        (df_resumen, path_embeddings, path_metadata)
    """
    # ── 1. Buscar ────────────────────────────────────────────────────
    grabaciones = buscar_grabaciones_xc(query, api_key, limit)
    if not grabaciones:
        return pd.DataFrame(), None, None

    total = len(grabaciones)
    resultados = []

    # ── 2. Procesar ──────────────────────────────────────────────────
    kwargs_comunes = dict(
        api_key=api_key, output_dir=output_dir,
        umbral_confianza=umbral_confianza,
        umbral_rms=umbral_rms, umbral_snr=umbral_snr,
        delete_audio_after=delete_audio_after,
        formato=formato,
    )

    if max_workers <= 1:
        # ── Secuencial ───────────────────────────────────────────────
        for i, rec in enumerate(grabaciones, 1):
            xc_id = rec.get("id", "?")
            especie = f"{rec.get('gen', '')} {rec.get('sp', '')}".strip()
            print(f"\n{'─'*60}")
            print(f"[{i}/{total}] XC:{xc_id} — {especie}")
            print(f"{'─'*60}")

            resultado = _procesar_una_grabacion(rec=rec, **kwargs_comunes)
            if resultado:
                resultados.append(resultado)
    else:
        # ── Paralelo ─────────────────────────────────────────────────
        print(f"\n🚀 Procesando con {max_workers} workers en paralelo...\n")

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futuros = {
                pool.submit(_procesar_una_grabacion, rec=rec, **kwargs_comunes): rec
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

    # ── 3. Consolidar para los 2 modelos ─────────────────────────────
    print(f"\n{'═'*60}")
    print("CONSOLIDANDO PARA LOS 2 MODELOS")
    print(f"{'═'*60}")

    path_emb  = _consolidar_embeddings(output_dir, formato)
    path_meta = _consolidar_metadata(resultados, output_dir, formato)

    # ── 4. Resumen final ─────────────────────────────────────────────
    df = pd.DataFrame(resultados)
    ok  = (df["estado"] == "ok").sum()
    err = (df["estado"] != "ok").sum()
    segs       = df["segmentos"].sum()
    confiables = df["confiables"].sum()
    n_especies = df[df["estado"] == "ok"]["nombre_cientifico"].nunique() if "nombre_cientifico" in df.columns else 0

    print(f"\n{'═'*60}")
    print(f"RESUMEN FINAL")
    print(f"{'═'*60}")
    print(f"  Audios:              {len(df)} ({ok} ok, {err} errores)")
    print(f"  Especies únicas:     {n_especies}")
    print(f"  Segmentos totales:   {segs}")
    print(f"  Segmentos confiables:{confiables}")
    if segs > 0:
        print(f"  Aprovechamiento:     {100*confiables/segs:.1f}%")
    print(f"\n  📁 Embeddings: {path_emb}")
    print(f"  📁 Metadata:   {path_meta}")
    print(f"{'═'*60}\n")

    return df, path_emb, path_meta


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline: XC → embeddings + metadata separados"
    )
    parser.add_argument("--api_key",    required=True)
    parser.add_argument("--query",      default=QUERY_DEFAULT)
    parser.add_argument("--limit",      type=int, default=10)
    parser.add_argument("--output_dir", default="audios/")
    parser.add_argument("--umbral",     type=float, default=0.7)
    parser.add_argument("--rms",        type=float, default=0.01)
    parser.add_argument("--snr",        type=float, default=5.0)
    parser.add_argument("--formato",    choices=["csv", "parquet"], default="csv")
    parser.add_argument("--workers",    type=int, default=1,
                        help="1=secuencial, 3-5=paralelo")
    parser.add_argument("--keep_audio", action="store_true")
    args = parser.parse_args()

    buscar_y_procesar(
        api_key=args.api_key,
        query=args.query,
        limit=args.limit,
        output_dir=args.output_dir,
        umbral_confianza=args.umbral,
        umbral_rms=args.rms,
        umbral_snr=args.snr,
        delete_audio_after=not args.keep_audio,
        formato=args.formato,
        max_workers=args.workers,
    )