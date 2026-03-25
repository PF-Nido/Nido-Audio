"""
Consolida todos los CSVs de embeddings generados por extraer_un_embedding.py
en un único archivo Parquet listo para entrenar XGBoost.

Uso:
    python consolidar_parquet.py
    python consolidar_parquet.py --csv_dir audios/ --output parquet/embeddings.parquet
    python consolidar_parquet.py --csv_dir audios/ --output parquet/embeddings.parquet --excluir_sin_confianza
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _leer_json_sidecar(csv_path: Path) -> dict:
    """
    Lee el JSON sidecar asociado al CSV (mismo nombre, distinta extensión).
    Retorna dict vacío si no existe.
    """
    json_path = csv_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _parsear_embedding(raw: str) -> np.ndarray | None:
    """Convierte el string 'v1;v2;...;v1024' en un np.array float32."""
    try:
        return np.array(raw.strip().split(";"), dtype=np.float32)
    except Exception:
        return None


def _codificar_ciclico(valor, periodo: float) -> tuple[float, float]:
    """
    Codifica una variable cíclica en sin/cos para que el modelo
    entienda que hora 23 está cerca de hora 0, etc.
    Retorna (sin, cos) o (None, None) si el valor es nulo.
    """
    try:
        v = float(valor)
        s = np.sin(2 * np.pi * v / periodo)
        c = np.cos(2 * np.pi * v / periodo)
        return round(float(s), 6), round(float(c), 6)
    except (TypeError, ValueError):
        return None, None


def normalizar_especie(nombre: str) -> str:
    """
    Limpia y normaliza el nombre de la especie para XGBoost.

    BirdNET usa formato "Género especie_Nombre común"
    XC usa formato     "Género especie"

    Ejemplos:
        "Saltator olivascens_Olivaceous Saltator" → "Saltator olivascens"
        "Grallaria saltuensis "                   → "Grallaria saltuensis"
        ""  o  None                               → "desconocido"
    """
    if not nombre or str(nombre).strip() in ("", "nan"):
        return "desconocido"
    return str(nombre).split("_")[0].strip()


def _extraer_temporales(fecha: str, hora: str) -> dict:
    """
    Extrae y codifica variables temporales desde strings de XC.
    fecha: "2022-03-18"
    hora:  "06:30"
    """
    resultado = {
        "mes": None, "mes_sin": None, "mes_cos": None,
        "hora": None, "hora_sin": None, "hora_cos": None,
        "dia_semana": None,
    }
    try:
        from datetime import datetime
        if fecha:
            dt = datetime.strptime(fecha, "%Y-%m-%d")
            resultado["mes"] = dt.month
            resultado["mes_sin"], resultado["mes_cos"] = _codificar_ciclico(dt.month, 12)
            resultado["dia_semana"] = dt.weekday()  # 0=lunes, 6=domingo
        if hora:
            # hora puede venir como "06:30" o "06:30:00"
            h, m = hora.split(":")[:2]
            hora_decimal = int(h) + int(m) / 60
            resultado["hora"] = round(hora_decimal, 2)
            resultado["hora_sin"], resultado["hora_cos"] = _codificar_ciclico(hora_decimal, 24)
    except Exception:
        pass
    return resultado


# ════════════════════════════════════════════════════════════════════════════
# CONSOLIDACIÓN
# ════════════════════════════════════════════════════════════════════════════

def consolidar_parquet(
    csv_dir: str = "audios/",
    output_path: str = "parquet/embeddings.parquet",
    excluir_sin_confianza: bool = False,
) -> pd.DataFrame:
    """
    Lee todos los CSVs de embeddings en csv_dir, los enriquece con los
    metadatos de sus JSONs sidecar, y los guarda como un único Parquet.

    Opción permisiva (default):
        Incluye confiable + sin_confianza + otra_especie
        (otra_especie conserva la etiqueta original de XC)

    Opción estricta (--excluir_sin_confianza):
        Solo incluye confiable + otra_especie

    Args:
        csv_dir:               Carpeta con los CSVs y JSONs
        output_path:           Ruta del Parquet de salida
        excluir_sin_confianza: Si True, excluye segmentos sin_confianza

    Returns:
        DataFrame consolidado.
    """
    csv_paths = sorted(Path(csv_dir).rglob("*.csv"))

    # Excluir el resumen que genera buscar_y_descargar_xc.py
    csv_paths = [p for p in csv_paths if p.stem != "resumen_descarga"]

    if not csv_paths:
        print(f"No se encontraron CSVs en {csv_dir}")
        return pd.DataFrame()

    print(f"CSVs encontrados: {len(csv_paths)}")
    print(f"Modo: {'estricto (solo confiable)' if excluir_sin_confianza else 'permisivo (confiable + sin_confianza)'}\n")

    filas = []
    errores = 0

    for csv_path in csv_paths:
        # ── Leer CSV de embeddings ────────────────────────────────────────
        try:
            df_csv = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ✗ Error leyendo {csv_path.name}: {e}")
            errores += 1
            continue

        if df_csv.empty:
            continue

        # ── Leer JSON sidecar ─────────────────────────────────────────────
        meta = _leer_json_sidecar(csv_path)

        # ── Filtrar por confiabilidad ─────────────────────────────────────
        if excluir_sin_confianza:
            df_csv = df_csv[df_csv["confiabilidad"] != "sin_confianza"]
        else:
            # Permisivo: incluir todo menos sin_score
            df_csv = df_csv[df_csv["confiabilidad"] != "sin_score"]

        if df_csv.empty:
            continue

        # ── Variables temporales codificadas ──────────────────────────────
        temporales = _extraer_temporales(
            meta.get("fecha", ""),
            meta.get("hora", ""),
        )

        # ── Construir una fila por segmento ───────────────────────────────
        for _, row in df_csv.iterrows():
            embedding = _parsear_embedding(str(row.get("embedding", "")))
            if embedding is None or len(embedding) != 1024:
                continue

            # Etiqueta final: siempre la especie de XC, normalizada
            # (otra_especie conserva etiqueta XC según decisión del usuario)
            especie_final = normalizar_especie(
                row.get("especie_xc")
                or meta.get("nombre_cientifico")
                or ""
            )

            fila = {
                # ── Identificación ────────────────────────────────────────
                "xc_id":             meta.get("xc_id", ""),
                "archivo":           row.get("archivo", csv_path.stem),
                "inicio_seg":        row.get("inicio_seg"),
                "fin_seg":           row.get("fin_seg"),

                # ── Etiqueta (normalizada, lista para XGBoost) ────────────
                "especie":           especie_final,
                "confiabilidad":     row.get("confiabilidad", ""),
                "score_birdnet":     row.get("score", None),
                # especie_detectada también normalizada para comparar fácil
                "especie_detectada": normalizar_especie(row.get("especie_detectada", "")),

                # ── Geoespaciales ─────────────────────────────────────────
                "lat":               meta.get("lat"),
                "lon":               meta.get("lon"),
                "altura":            meta.get("altura"),

                # ── Temporales crudos ─────────────────────────────────────
                "fecha":             meta.get("fecha"),
                "hora_raw":          meta.get("hora"),

                # ── Temporales codificados (listos para XGBoost) ──────────
                "mes":               temporales["mes"],
                "mes_sin":           temporales["mes_sin"],
                "mes_cos":           temporales["mes_cos"],
                "hora":              temporales["hora"],
                "hora_sin":          temporales["hora_sin"],
                "hora_cos":          temporales["hora_cos"],
                "dia_semana":        temporales["dia_semana"],

                # ── Embedding ─────────────────────────────────────────────
                "embedding":         embedding,
            }
            filas.append(fila)

    if not filas:
        print("No hay segmentos para consolidar.")
        return pd.DataFrame()

    # ── Construir DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(filas)

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"{'═'*55}")
    print(f"RESUMEN CONSOLIDACIÓN")
    print(f"{'═'*55}")
    print(f"  Segmentos totales:   {len(df):>6}")
    print(f"  Especies únicas:     {df['especie'].nunique():>6}")
    print(f"  Archivos procesados: {df['xc_id'].nunique():>6}")
    print(f"\n  Por confiabilidad:")
    for conf, count in df["confiabilidad"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"    {conf:<20} {count:>5}  ({pct:.0f}%)")
    print(f"\n  Especies más frecuentes:")
    for esp, count in df["especie"].value_counts().head(10).items():
        print(f"    {esp:<35} {count:>4} segs")

    # ── Guardar Parquet ───────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    size_mb = Path(output_path).stat().st_size / 1_048_576
    print(f"\n✓ Parquet guardado: {output_path}  ({size_mb:.1f} MB)")

    return df


# ════════════════════════════════════════════════════════════════════════════
# CARGAR PARA XGBOOST
# ════════════════════════════════════════════════════════════════════════════

def cargar_para_xgboost(
    parquet_path: str,
    incluir_geo: bool = True,
    incluir_temporales: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Carga el Parquet y devuelve X, y listos para XGBoost.

    Las variables temporales ya vienen codificadas en sin/cos.

    Args:
        parquet_path:       Ruta al Parquet consolidado
        incluir_geo:        Si True, agrega lat, lon, altura a X
        incluir_temporales: Si True, agrega hora_sin/cos, mes_sin/cos a X

    Returns:
        X:        np.ndarray (n_segmentos, n_features)
        y:        np.ndarray con nombres de especies
        especies: lista de especies únicas (para el label encoder)
    """
    df = pd.read_parquet(parquet_path)

    # Embedding base (1024 dims)
    X = np.vstack(df["embedding"].values)

    # Features geoespaciales
    if incluir_geo:
        geo_cols = ["lat", "lon", "altura"]
        geo = df[geo_cols].fillna(0).values.astype(np.float32)
        X = np.hstack([X, geo])

    # Features temporales codificadas
    if incluir_temporales:
        temp_cols = ["hora_sin", "hora_cos", "mes_sin", "mes_cos", "dia_semana"]
        temp = df[temp_cols].fillna(0).values.astype(np.float32)
        X = np.hstack([X, temp])

    especies = sorted(df["especie"].unique().tolist())

    # Label encoding: "Grallaria saltuensis" → 0, "Pyriglena maura" → 1, etc.
    especie_a_idx = {esp: i for i, esp in enumerate(especies)}
    y = np.array([especie_a_idx[e] for e in df["especie"].values], dtype=np.int32)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}  (enteros 0..{len(especies)-1})")
    print(f"\nEspecies ({len(especies)}):")
    for i, esp in enumerate(especies):
        count = (df["especie"] == esp).sum()
        print(f"  {i:>3} → {esp:<35} ({count} segs)")

    return X, y, especies


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolida CSVs de embeddings en un Parquet para XGBoost"
    )
    parser.add_argument("--csv_dir",    default="audios/",
                        help="Carpeta con los CSVs y JSONs (default: audios/)")
    parser.add_argument("--output",     default="parquet/embeddings.parquet",
                        help="Ruta del Parquet de salida")
    parser.add_argument("--excluir_sin_confianza", action="store_true",
                        help="Modo estricto: excluir segmentos sin_confianza")
    args = parser.parse_args()

    df = consolidar_parquet(
        csv_dir=args.csv_dir,
        output_path=args.output,
        excluir_sin_confianza=args.excluir_sin_confianza,
    )