"""
fase2_clustering.py
───────────────────
Limpieza de datos por clustering sobre los embeddings de la Fase 1.

Pipeline:
  1. Carga embeddings consolidados de Fase 1
  2. Reduce dimensionalidad (UMAP: 1024 → 50)
  3. Clustering POR ESPECIE (HDBSCAN dentro de cada especie)
     → Escala correctamente con datasets grandes (>50k segmentos)
     → Detecta outliers dentro de cada especie, no entre especies
  4. Identifica clusters de ruido y posibles mislabels
  5. Genera dataset limpio + reporte visual

Requisitos:
    pip install umap-learn hdbscan matplotlib seaborn scikit-learn pandas numpy

Uso:
    python fase2_clustering.py --input datos_fase1/embeddings_consolidado.parquet
    python fase2_clustering.py --input datos_fase1/embeddings_consolidado.parquet --visualizar

Desde Python:
    from fase2_clustering import limpiar_por_clustering
    df_limpio, reporte = limpiar_por_clustering("datos_fase1/embeddings_consolidado.parquet")
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

UMAP_N_COMPONENTS    = 50      # dimensiones para clustering
UMAP_N_NEIGHBORS     = 30      # vecinos para UMAP
UMAP_MIN_DIST        = 0.0     # permite clusters más compactos
UMAP_METRIC          = "cosine"

# HDBSCAN por especie:
#   min_cluster_size se calcula automáticamente como max(5, n_especie // 20)
#   es decir, ~5% del tamaño de la especie, mínimo 5 puntos.
#   Puedes sobreescribirlo con --min_cluster en CLI.
HDBSCAN_MIN_SAMPLES  = 5       # controla densidad mínima
HDBSCAN_MIN_CLUSTER_PCT = 0.05 # 5% del tamaño de la especie
HDBSCAN_MIN_CLUSTER_ABS = 5    # mínimo absoluto de puntos por cluster


# ════════════════════════════════════════════════════════════════════════════
# 1. CARGAR EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

def _cargar_embeddings(input_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carga el archivo consolidado de Fase 1.
    Retorna:
        df:      DataFrame completo
        X_emb:   numpy array de shape (n_segmentos, 1024)
    """
    print(f"Cargando embeddings de: {input_path}")

    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"  ✓ {len(df)} segmentos, columnas: {df.columns.tolist()}")

    if "embedding" not in df.columns:
        raise ValueError("No se encontró la columna 'embedding'")

    print("  Parseando embeddings...")

    if isinstance(df["embedding"].iloc[0], str):
        X_emb = np.array(
            df["embedding"].apply(lambda s: [float(x) for x in s.split(";")]).tolist()
        )
    elif isinstance(df["embedding"].iloc[0], (list, np.ndarray)):
        X_emb = np.array(df["embedding"].tolist())
    else:
        raise ValueError(f"Formato de embedding no reconocido: {type(df['embedding'].iloc[0])}")

    print(f"  ✓ Matriz de embeddings: {X_emb.shape}")

    n_nan = np.isnan(X_emb).any(axis=1).sum()
    n_inf = np.isinf(X_emb).any(axis=1).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠ {n_nan} filas con NaN, {n_inf} con Inf — se reemplazan con 0")
        X_emb = np.nan_to_num(X_emb, nan=0.0, posinf=0.0, neginf=0.0)

    return df, X_emb


# ════════════════════════════════════════════════════════════════════════════
# 2. REDUCIR DIMENSIONALIDAD (UMAP)
# ════════════════════════════════════════════════════════════════════════════

def _reducir_umap(X: np.ndarray, n_components: int = UMAP_N_COMPONENTS) -> np.ndarray:
    """Reduce 1024 dimensiones a n_components con UMAP (métrica coseno)."""
    import umap

    print(f"\n  Reduciendo dimensionalidad: {X.shape[1]} → {n_components} (UMAP)...")
    print(f"    n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
        verbose=False,
        low_memory=True,   # importante para datasets grandes
    )

    X_reduced = reducer.fit_transform(X)
    print(f"  ✓ Reducido a {X_reduced.shape}")
    return X_reduced


def _reducir_umap_2d(X: np.ndarray) -> np.ndarray:
    """Reduce a 2D para visualización."""
    import umap

    print(f"  Generando proyección 2D para visualización...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.1,
        metric=UMAP_METRIC,
        random_state=42,
        verbose=False,
        low_memory=True,
    )
    return reducer.fit_transform(X)


# ════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERING POR ESPECIE (HDBSCAN)
# ════════════════════════════════════════════════════════════════════════════

def _clustering_por_especie(
    df: pd.DataFrame,
    X_reduced: np.ndarray,
    min_cluster_size_pct: float = HDBSCAN_MIN_CLUSTER_PCT,
    min_cluster_size_abs: int   = HDBSCAN_MIN_CLUSTER_ABS,
    min_samples: int            = HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    """
    Clustering HDBSCAN dentro de cada especie por separado.

    Ventajas frente al clustering global:
      - Escala correctamente con datasets grandes (no colapsa todo en 2 clusters)
      - min_cluster_size se adapta al tamaño de cada especie
      - Detecta outliers dentro de cada especie, no confunde especies entre sí

    Retorna:
        labels_global: array de int con IDs de cluster globales.
                       -1 = ruido/outlier dentro de esa especie.
    """
    import hdbscan

    labels_global = np.full(len(df), -1, dtype=int)
    offset = 0

    df_work = df.copy().reset_index(drop=True)
    df_work["nombre_completo"] = df_work["genero"] + " " + df_work["especie"]
    especies = sorted(df_work["nombre_completo"].unique())

    print(f"\n  Clustering por especie ({len(especies)} especies)...")
    print(f"  {'─' * 65}")
    print(f"  {'Especie':<30} {'N':>7} {'min_cl':>7} {'Clusters':>9} {'Ruido':>7}")
    print(f"  {'─' * 65}")

    total_ruido = 0

    for especie in especies:
        mask = (df_work["nombre_completo"] == especie).values
        idx_especie = np.where(mask)[0]
        X_esp = X_reduced[mask]
        n = len(X_esp)

        # min_cluster_size adaptativo: 5% del tamaño, con un mínimo absoluto
        min_cl = max(min_cluster_size_abs, int(n * min_cluster_size_pct))

        if n < min_cl * 2:
            # Especie demasiado pequeña para clusterizar → conservar todo
            labels_global[idx_especie] = offset
            n_clusters_esp = 1
            n_ruido_esp = 0
            offset += 1
        else:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cl,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels_esp = clusterer.fit_predict(X_esp)

            n_ruido_esp = (labels_esp == -1).sum()
            n_clusters_esp = len(set(labels_esp)) - (1 if -1 in labels_esp else 0)

            # Reasignar IDs para que no colisionen entre especies
            for lbl in np.unique(labels_esp):
                if lbl == -1:
                    continue
                mask_lbl = labels_esp == lbl
                labels_global[idx_especie[mask_lbl]] = offset + lbl

            offset += (labels_esp.max() + 1) if labels_esp.max() >= 0 else 1

        total_ruido += n_ruido_esp
        print(f"  {especie:<30} {n:>7} {min_cl:>7} {n_clusters_esp:>9} {n_ruido_esp:>7}")

    print(f"  {'─' * 65}")
    n_clusters_total = len(set(labels_global)) - (1 if -1 in labels_global else 0)
    print(f"  ✓ Total clusters: {n_clusters_total}")
    print(f"  ✓ Total ruido:    {total_ruido} ({100 * total_ruido / len(df):.1f}%)")

    return labels_global


# ════════════════════════════════════════════════════════════════════════════
# 4. ANALIZAR CLUSTERS
# ════════════════════════════════════════════════════════════════════════════

def _analizar_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Analiza cada cluster para determinar si es:
      - especie:          dominado por una especie (>= 70%) → datos buenos
      - mixto_dominante:  una especie entre 40-70%          → conservar dominante
      - mixto:            sin especie dominante (<40%)       → revisar
      - ruido:            label = -1 de HDBSCAN             → descartar

    Como el clustering es por especie, la gran mayoría serán tipo 'especie'.
    Los 'mixto' indican segmentos que acústicamente se parecen a otra especie.
    """
    df_work = df.copy()
    df_work["cluster"] = labels
    df_work["nombre_completo"] = df_work["genero"] + " " + df_work["especie"]

    print(f"\n  Analizando clusters...")

    analisis = []

    for cluster_id in sorted(df_work["cluster"].unique()):
        mask = df_work["cluster"] == cluster_id
        sub = df_work[mask]
        n = len(sub)

        if cluster_id == -1:
            analisis.append({
                "cluster": cluster_id,
                "n_segmentos": n,
                "tipo": "ruido",
                "especie_dominante": "",
                "pct_dominante": 0,
                "n_especies": sub["nombre_completo"].nunique(),
                "accion": "descartar",
            })
            continue

        conteo = sub["nombre_completo"].value_counts()
        especie_top = conteo.index[0]
        pct_top = conteo.iloc[0] / n
        n_especies = len(conteo)

        if pct_top >= 0.7:
            tipo   = "especie"
            accion = "conservar"
        elif pct_top >= 0.4:
            tipo   = "mixto_dominante"
            accion = "conservar"
        else:
            tipo   = "mixto"
            accion = "revisar"

        analisis.append({
            "cluster": cluster_id,
            "n_segmentos": n,
            "tipo": tipo,
            "especie_dominante": especie_top,
            "pct_dominante": round(pct_top * 100, 1),
            "n_especies": n_especies,
            "accion": accion,
        })

    df_analisis = pd.DataFrame(analisis)

    # Resumen por tipo
    resumen = df_analisis["tipo"].value_counts()
    print(f"\n  Resumen de tipos de cluster:")
    for tipo, n in resumen.items():
        print(f"    {tipo:<20} {n:>5} clusters")

    return df_analisis


# ════════════════════════════════════════════════════════════════════════════
# 5. DETECTAR MISLABELS
# ════════════════════════════════════════════════════════════════════════════

def _detectar_mislabels(
    df: pd.DataFrame,
    labels: np.ndarray,
    df_analisis: pd.DataFrame,
) -> pd.Series:
    """
    Para cada segmento decide:
      - "ok"       → su especie coincide con la dominante del cluster
      - "mislabel" → su especie NO coincide con la dominante (posible error XC)
      - "ruido"    → HDBSCAN lo marcó como outlier dentro de su especie
      - "revisar"  → cluster mixto sin especie clara

    Con clustering por especie, los mislabels son raros pero posibles:
    ocurren cuando un segmento de especie A queda en un cluster dominado por B.
    Esto puede pasar si hay grabaciones muy ruidosas o cantos atípicos.
    """
    df_work = df.copy()
    df_work["cluster"] = labels
    df_work["nombre_completo"] = df_work["genero"] + " " + df_work["especie"]

    cluster_info = df_analisis.set_index("cluster")

    estado = []
    for idx, row in df_work.iterrows():
        cl = row["cluster"]

        if cl == -1:
            estado.append("ruido")
            continue

        info = cluster_info.loc[cl]

        if info["tipo"] == "mixto" or info["accion"] == "revisar":
            estado.append("revisar")
        elif row["nombre_completo"] == info["especie_dominante"]:
            estado.append("ok")
        else:
            estado.append("mislabel")

    return pd.Series(estado, index=df.index)


# ════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIÓN
# ════════════════════════════════════════════════════════════════════════════

def _visualizar(
    df: pd.DataFrame,
    X_2d: np.ndarray,
    labels: np.ndarray,
    estados: pd.Series,
    output_dir: str,
):
    """Genera las tres gráficas de diagnóstico."""
    print(f"\n  Generando visualizaciones...")

    df_viz = pd.DataFrame({
        "umap_1": X_2d[:, 0],
        "umap_2": X_2d[:, 1],
        "cluster": labels,
        "especie": df["genero"].values + " " + df["especie"].values,
        "estado": estados.values,
    })

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Clusters coloreados ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))
    ruido = df_viz["cluster"] == -1

    ax.scatter(
        df_viz.loc[ruido, "umap_1"], df_viz.loc[ruido, "umap_2"],
        c="lightgray", s=5, alpha=0.3, label="ruido",
    )
    if (~ruido).sum() > 0:
        scatter = ax.scatter(
            df_viz.loc[~ruido, "umap_1"], df_viz.loc[~ruido, "umap_2"],
            c=df_viz.loc[~ruido, "cluster"], cmap="tab20",
            s=8, alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster ID")

    ax.set_title("Clusters HDBSCAN por especie (UMAP 2D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    path1 = os.path.join(output_dir, "clusters.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {path1}")

    # ── 2. Coloreados por especie ────────────────────────────────────
    especies_unicas = df_viz["especie"].unique()
    n_especies = len(especies_unicas)

    fig, ax = plt.subplots(figsize=(14, 10))
    palette = sns.color_palette("husl", n_especies)
    color_map = {sp: palette[i] for i, sp in enumerate(especies_unicas)}

    for sp in especies_unicas:
        mask = df_viz["especie"] == sp
        ax.scatter(
            df_viz.loc[mask, "umap_1"], df_viz.loc[mask, "umap_2"],
            c=[color_map[sp]], s=8, alpha=0.5, label=sp,
        )

    ax.set_title("Embeddings coloreados por especie (etiqueta XC)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if n_especies <= 20:
        ax.legend(markerscale=3, fontsize=7, loc="best")

    path2 = os.path.join(output_dir, "especies.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {path2}")

    # ── 3. Coloreados por estado (ok/mislabel/ruido/revisar) ─────────
    fig, ax = plt.subplots(figsize=(14, 10))
    colores_estado = {
        "ok":      "#2ecc71",
        "mislabel":"#e74c3c",
        "ruido":   "#95a5a6",
        "revisar": "#f39c12",
    }

    for estado, color in colores_estado.items():
        mask = df_viz["estado"] == estado
        if mask.sum() > 0:
            ax.scatter(
                df_viz.loc[mask, "umap_1"], df_viz.loc[mask, "umap_2"],
                c=color, s=8, alpha=0.5, label=f"{estado} ({mask.sum()})",
            )

    ax.set_title("Estado de limpieza por segmento")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(markerscale=3)

    path3 = os.path.join(output_dir, "estados_limpieza.png")
    fig.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {path3}")


# ════════════════════════════════════════════════════════════════════════════
# 7. PIPELINE COMPLETO
# ════════════════════════════════════════════════════════════════════════════

def limpiar_por_clustering(
    input_path: str,
    output_dir: str = None,
    formato: str = "parquet",
    visualizar: bool = True,
    min_cluster_size_pct: float = HDBSCAN_MIN_CLUSTER_PCT,
    min_cluster_size_abs: int   = HDBSCAN_MIN_CLUSTER_ABS,
    min_samples: int            = HDBSCAN_MIN_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de Fase 2:
      1. Carga embeddings de Fase 1
      2. Escala + UMAP (1024 → 50 dims)
      3. HDBSCAN por especie (escala bien con datasets grandes)
      4. Analiza clusters
      5. Detecta mislabels
      6. Guarda dataset limpio + análisis
      7. Visualizaciones opcionales

    Args:
        input_path:           Ruta al parquet/csv de Fase 1
        output_dir:           Carpeta de salida (default: misma carpeta que input)
        formato:              "parquet" (recomendado) o "csv"
        visualizar:           Si True, genera PNGs de diagnóstico
        min_cluster_size_pct: Fracción del tamaño de la especie para min_cluster_size
        min_cluster_size_abs: Mínimo absoluto de min_cluster_size
        min_samples:          min_samples de HDBSCAN

    Returns:
        (df_limpio, df_analisis_clusters)
    """
    print(f"\n{'═' * 60}")
    print(f"  FASE 2: LIMPIEZA POR CLUSTERING (por especie)")
    print(f"{'═' * 60}")

    if output_dir is None:
        output_dir = str(Path(input_path).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar ────────────────────────────────────────────────────
    print(f"\n[1/6] Cargando datos...")
    df, X_emb = _cargar_embeddings(input_path)

    if "genero" not in df.columns or "especie" not in df.columns:
        raise ValueError("El DataFrame debe tener columnas 'genero' y 'especie'.")

    df["_nombre_completo"] = df["genero"] + " " + df["especie"]
    n_especies = df["_nombre_completo"].nunique()
    print(f"  ✓ {n_especies} especies detectadas")

    # ── 2. UMAP ──────────────────────────────────────────────────────
    print(f"\n[2/6] Reducción de dimensionalidad (UMAP)...")
    X_scaled = StandardScaler().fit_transform(X_emb)
    X_reduced = _reducir_umap(X_scaled, n_components=UMAP_N_COMPONENTS)

    # ── 3. Clustering por especie ────────────────────────────────────
    print(f"\n[3/6] Clustering HDBSCAN por especie...")
    labels = _clustering_por_especie(
        df=df,
        X_reduced=X_reduced,
        min_cluster_size_pct=min_cluster_size_pct,
        min_cluster_size_abs=min_cluster_size_abs,
        min_samples=min_samples,
    )

    # ── 4. Analizar clusters ─────────────────────────────────────────
    print(f"\n[4/6] Analizando clusters...")
    df_analisis = _analizar_clusters(df, labels)

    # ── 5. Detectar mislabels ────────────────────────────────────────
    print(f"\n[5/6] Detectando posibles mislabels...")
    estados = _detectar_mislabels(df, labels, df_analisis)
    df["estado_limpieza"] = estados
    df["cluster"] = labels
    df.drop(columns=["_nombre_completo"], inplace=True)

    conteo = estados.value_counts()
    total  = len(df)
    print(f"\n  Resultado de limpieza:")
    for estado, n in conteo.items():
        pct   = 100 * n / total
        emoji = {"ok": "✅", "mislabel": "❌", "ruido": "🔇", "revisar": "⚠️ "}.get(estado, "?")
        print(f"    {emoji} {estado:<12} {n:>6} ({pct:.1f}%)")

    # ── 6. Guardar ───────────────────────────────────────────────────
    print(f"\n[6/6] Guardando resultados...")
    ext = ".parquet" if formato == "parquet" else ".csv"

    def _guardar(df_out, nombre):
        path = os.path.join(output_dir, f"{nombre}{ext}")
        if formato == "parquet":
            df_out.to_parquet(path, index=False)
        else:
            df_out.to_csv(path, index=False)
        return path

    out_completo = _guardar(df, "embeddings_con_limpieza")
    print(f"  ✓ Completo: {out_completo}")

    df_limpio = df[df["estado_limpieza"] == "ok"].copy()
    out_limpio = _guardar(df_limpio, "embeddings_limpios")
    print(f"  ✓ Limpio (solo 'ok'): {out_limpio} ({len(df_limpio)} segmentos)")

    out_analisis = _guardar(df_analisis, "analisis_clusters")
    print(f"  ✓ Análisis clusters: {out_analisis}")

    # ── 7. Visualización ─────────────────────────────────────────────
    if visualizar:
        print(f"\n  Generando visualizaciones (puede tardar un momento)...")
        X_2d = _reducir_umap_2d(X_scaled)
        _visualizar(df, X_2d, labels, estados, output_dir)

    # ── Resumen final ────────────────────────────────────────────────
    n_clusters_total = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"\n{'═' * 60}")
    print(f"  RESUMEN FASE 2")
    print(f"{'═' * 60}")
    print(f"    Segmentos entrada:       {total}")
    print(f"    Especies:                {n_especies}")
    print(f"    Clusters encontrados:    {n_clusters_total}")
    print(f"    Segmentos OK:            {conteo.get('ok', 0)}")
    print(f"    Posibles mislabels:      {conteo.get('mislabel', 0)}")
    print(f"    Ruido:                   {conteo.get('ruido', 0)}")
    print(f"    Para revisar:            {conteo.get('revisar', 0)}")
    print(f"    Tasa de conservación:    {100 * conteo.get('ok', 0) / total:.1f}%")
    print(f"\n    📁 {out_completo}")
    print(f"    📁 {out_limpio}")
    print(f"    📁 {out_analisis}")
    if visualizar:
        print(f"    📊 {output_dir}/clusters.png")
        print(f"    📊 {output_dir}/especies.png")
        print(f"    📊 {output_dir}/estados_limpieza.png")
    print(f"{'═' * 60}\n")

    return df_limpio, df_analisis


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 2: Limpieza por clustering HDBSCAN por especie"
    )
    parser.add_argument("--input", required=True,
                        help="Ruta al embeddings_consolidado de Fase 1 (.parquet o .csv)")
    parser.add_argument("--output_dir", default=None,
                        help="Carpeta de salida (default: misma carpeta que input)")
    parser.add_argument("--formato", choices=["csv", "parquet"], default="parquet",
                        help="Formato de salida (default: parquet)")
    parser.add_argument("--visualizar", action="store_true",
                        help="Generar gráficas PNG de diagnóstico")
    parser.add_argument("--min_cluster_pct", type=float, default=HDBSCAN_MIN_CLUSTER_PCT,
                        help=f"Fracción del tamaño de especie para min_cluster_size "
                             f"(default: {HDBSCAN_MIN_CLUSTER_PCT})")
    parser.add_argument("--min_cluster_abs", type=int, default=HDBSCAN_MIN_CLUSTER_ABS,
                        help=f"Mínimo absoluto de min_cluster_size "
                             f"(default: {HDBSCAN_MIN_CLUSTER_ABS})")
    parser.add_argument("--min_samples", type=int, default=HDBSCAN_MIN_SAMPLES,
                        help=f"min_samples de HDBSCAN (default: {HDBSCAN_MIN_SAMPLES})")
    args = parser.parse_args()

    df_limpio, df_analisis = limpiar_por_clustering(
        input_path=args.input,
        output_dir=args.output_dir,
        formato=args.formato,
        visualizar=args.visualizar,
        min_cluster_size_pct=args.min_cluster_pct,
        min_cluster_size_abs=args.min_cluster_abs,
        min_samples=args.min_samples,
    )