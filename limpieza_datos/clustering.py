"""
fase2_clustering.py
───────────────────
Limpieza de datos por clustering sobre los embeddings de la Fase 1.

Pipeline:
  1. Carga embeddings consolidados de Fase 1
  2. Reduce dimensionalidad (UMAP: 1024 → 50)
  3. Clustering (HDBSCAN)
  4. Identifica clusters de ruido y posibles mislabels
  5. Genera dataset limpio + reporte visual

Requisitos:
    pip install umap-learn hdbscan matplotlib seaborn scikit-learn pandas numpy

Uso:
    python fase2_clustering.py --input datos_fase1/embeddings_consolidado.csv
    python fase2_clustering.py --input datos_fase1/embeddings_consolidado.csv --visualizar

Desde Python:
    from fase2_clustering import limpiar_por_clustering
    df_limpio, reporte = limpiar_por_clustering("datos_fase1/embeddings_consolidado.csv")
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # para servidores sin display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════════

UMAP_N_COMPONENTS    = 50     # dimensiones para clustering
UMAP_N_COMPONENTS_2D = 2      # dimensiones para visualización
UMAP_N_NEIGHBORS     = 30     # vecinos para UMAP
UMAP_MIN_DIST        = 0.0    # permite clusters más compactos
UMAP_METRIC          = "cosine"  # embeddings de redes neuronales → cosine

HDBSCAN_MIN_CLUSTER  = 10     # mínimo de puntos para formar un cluster
HDBSCAN_MIN_SAMPLES  = 5      # controla densidad mínima


# ════════════════════════════════════════════════════════════════════════════
# 1. CARGAR EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════

def _cargar_embeddings(input_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Carga el archivo consolidado de Fase 1.
    Separa las columnas de metadata del embedding numérico.

    Retorna:
        df:         DataFrame completo (sin columna embedding como string)
        X_emb:      numpy array de shape (n_segmentos, 1024)
    """
    print(f"Cargando embeddings de: {input_path}")

    if input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"  ✓ {len(df)} segmentos, {df.columns.tolist()}")

    # Parsear embedding
    if "embedding" not in df.columns:
        raise ValueError("No se encontró la columna 'embedding'")

    print("  Parseando embeddings...")

    if isinstance(df["embedding"].iloc[0], str):
        # CSV: embedding es string separado por ;
        X_emb = np.array(
            df["embedding"].apply(lambda s: [float(x) for x in s.split(";")]).tolist()
        )
    elif isinstance(df["embedding"].iloc[0], (list, np.ndarray)):
        # Parquet: embedding ya es lista
        X_emb = np.array(df["embedding"].tolist())
    else:
        raise ValueError(f"Formato de embedding no reconocido: "
                         f"{type(df['embedding'].iloc[0])}")

    print(f"  ✓ Matriz de embeddings: {X_emb.shape}")

    # Verificar NaN/Inf
    n_nan = np.isnan(X_emb).any(axis=1).sum()
    n_inf = np.isinf(X_emb).any(axis=1).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"  ⚠ {n_nan} filas con NaN, {n_inf} con Inf — se reemplazan con 0")
        X_emb = np.nan_to_num(X_emb, nan=0.0, posinf=0.0, neginf=0.0)

    return df, X_emb


# ════════════════════════════════════════════════════════════════════════════
# 2. REDUCIR DIMENSIONALIDAD (UMAP)
# ════════════════════════════════════════════════════════════════════════════

def _reducir_umap(X: np.ndarray, n_components: int = UMAP_N_COMPONENTS,
                  ) -> np.ndarray:
    """
    Reduce 1024 dimensiones a n_components con UMAP.
    Usa métrica coseno (ideal para embeddings de redes neuronales).
    """
    import umap

    print(f"\n  Reduciendo dimensionalidad: {X.shape[1]} → {n_components} (UMAP)...")
    print(f"    n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, "
          f"metric={UMAP_METRIC}")

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
        verbose=False,
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
        min_dist=0.1,  # un poco más separado para ver mejor
        metric=UMAP_METRIC,
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(X)


# ════════════════════════════════════════════════════════════════════════════
# 3. CLUSTERING (HDBSCAN)
# ════════════════════════════════════════════════════════════════════════════

def _clustering_hdbscan(X_reduced: np.ndarray) -> np.ndarray:
    """
    Clustering con HDBSCAN.
    Retorna etiquetas: -1 = ruido (no pertenece a ningún cluster).
    """
    import hdbscan

    print(f"\n  Clustering HDBSCAN (min_cluster={HDBSCAN_MIN_CLUSTER}, "
          f"min_samples={HDBSCAN_MIN_SAMPLES})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(X_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_ruido = (labels == -1).sum()
    print(f"  ✓ {n_clusters} clusters encontrados")
    print(f"  ✓ {n_ruido} puntos marcados como ruido ({100*n_ruido/len(labels):.1f}%)")

    return labels


# ════════════════════════════════════════════════════════════════════════════
# 4. ANALIZAR CLUSTERS
# ════════════════════════════════════════════════════════════════════════════

def _analizar_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Analiza cada cluster para determinar si es:
      - cluster_especie:  dominado por una especie (>= 70%) → datos buenos
      - cluster_mixto:    sin especie dominante → posibles mislabels
      - cluster_ruido:    label = -1 de HDBSCAN → ruido/outliers

    Retorna DataFrame con el análisis por cluster.
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
                "accion": "revisar",
            })
            continue

        # Distribución de especies en el cluster
        conteo = sub["nombre_completo"].value_counts()
        especie_top = conteo.index[0]
        pct_top = conteo.iloc[0] / n

        n_especies = len(conteo)

        if pct_top >= 0.7:
            tipo = "especie"
            accion = "conservar"
        elif pct_top >= 0.4:
            tipo = "mixto_dominante"
            accion = "conservar"  # la especie dominante es probablemente correcta
        else:
            tipo = "mixto"
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

    # Resumen
    print(f"\n  {'─'*70}")
    print(f"  {'Cluster':>8} {'Tipo':<18} {'N':>6} {'Especie dominante':<30} {'%':>6} {'Acción':<10}")
    print(f"  {'─'*70}")
    for _, row in df_analisis.iterrows():
        print(f"  {row['cluster']:>8} {row['tipo']:<18} {row['n_segmentos']:>6} "
              f"{row['especie_dominante']:<30} {row['pct_dominante']:>5.1f}% "
              f"{row['accion']:<10}")
    print(f"  {'─'*70}")

    return df_analisis


# ════════════════════════════════════════════════════════════════════════════
# 5. DETECTAR MISLABELS DENTRO DE CADA CLUSTER
# ════════════════════════════════════════════════════════════════════════════

def _detectar_mislabels(df: pd.DataFrame, labels: np.ndarray,
                         df_analisis: pd.DataFrame) -> pd.Series:
    """
    Para cada segmento, decide si es:
      - "ok"         → su especie coincide con la especie dominante del cluster
      - "mislabel"   → su especie NO coincide con la dominante (posible error de XC)
      - "ruido"      → HDBSCAN lo marcó como outlier (cluster -1)
      - "revisar"    → cluster mixto sin especie clara

    Retorna una Series con la etiqueta para cada fila.
    """
    df_work = df.copy()
    df_work["cluster"] = labels
    df_work["nombre_completo"] = df_work["genero"] + " " + df_work["especie"]

    # Mapear cluster → especie dominante y tipo
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

def _visualizar(df: pd.DataFrame, X_2d: np.ndarray, labels: np.ndarray,
                estados: pd.Series, output_dir: str):
    """Genera gráficas de los clusters."""
    print(f"\n  Generando visualizaciones...")

    df_viz = pd.DataFrame({
        "umap_1": X_2d[:, 0],
        "umap_2": X_2d[:, 1],
        "cluster": labels,
        "especie": df["genero"] + " " + df["especie"],
        "estado": estados,
    })

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Clusters coloreados ───────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ruido = df_viz["cluster"] == -1

    # Ruido en gris
    ax.scatter(df_viz.loc[ruido, "umap_1"], df_viz.loc[ruido, "umap_2"],
               c="lightgray", s=5, alpha=0.3, label="ruido")

    # Clusters con colores
    scatter = ax.scatter(
        df_viz.loc[~ruido, "umap_1"], df_viz.loc[~ruido, "umap_2"],
        c=df_viz.loc[~ruido, "cluster"], cmap="tab20",
        s=8, alpha=0.6,
    )
    ax.set_title("Clusters HDBSCAN sobre embeddings BirdNET (UMAP 2D)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.colorbar(scatter, ax=ax, label="Cluster ID")

    path1 = os.path.join(output_dir, "clusters.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ {path1}")

    # ── 2. Coloreados por especie ────────────────────────────────────
    especies_unicas = df_viz["especie"].unique()
    n_especies = len(especies_unicas)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
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

    # ── 3. Coloreados por estado (ok/mislabel/ruido) ─────────────────
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    colores_estado = {
        "ok": "#2ecc71",
        "mislabel": "#e74c3c",
        "ruido": "#95a5a6",
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
# 7. PIPELINE COMPLETO DE FASE 2
# ════════════════════════════════════════════════════════════════════════════

def limpiar_por_clustering(
    input_path: str,
    output_dir: str = None,
    formato: str = "csv",
    visualizar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo de Fase 2:
      1. Carga embeddings de Fase 1
      2. UMAP para reducción de dimensionalidad
      3. HDBSCAN para clustering
      4. Analiza clusters (especie dominante, mixtos, ruido)
      5. Detecta mislabels
      6. Genera dataset limpio
      7. Visualizaciones opcionales

    Args:
        input_path: Ruta al embeddings_consolidado de Fase 1
        output_dir: Carpeta de salida (default: misma que input)
        formato:    "csv" o "parquet"
        visualizar: Si True, genera PNGs con las gráficas

    Returns:
        (df_limpio, df_analisis_clusters)
    """
    print(f"\n{'═' * 60}")
    print(f"  FASE 2: LIMPIEZA POR CLUSTERING")
    print(f"{'═' * 60}")

    if output_dir is None:
        output_dir = str(Path(input_path).parent)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Cargar ────────────────────────────────────────────────────
    print(f"\n[1/6] Cargando datos...")
    df, X_emb = _cargar_embeddings(input_path)

    if len(df) < HDBSCAN_MIN_CLUSTER * 2:
        print(f"  ⚠ Solo {len(df)} segmentos — muy pocos para clustering")
        print(f"  ⚠ Se necesitan al menos {HDBSCAN_MIN_CLUSTER * 2}")
        print(f"  → Guardando todo como 'ok' sin filtrar")

        df["estado_limpieza"] = "ok"
        df["cluster"] = 0

        ext = ".parquet" if formato == "parquet" else ".csv"
        out = os.path.join(output_dir, f"embeddings_limpios{ext}")
        if formato == "parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)

        return df, pd.DataFrame()

    # ── 2. UMAP ─────────────────────────────────────────────────────
    print(f"\n[2/6] Reducción de dimensionalidad (UMAP)...")
    X_scaled = StandardScaler().fit_transform(X_emb)
    X_reduced = _reducir_umap(X_scaled, n_components=UMAP_N_COMPONENTS)

    # ── 3. HDBSCAN ──────────────────────────────────────────────────
    print(f"\n[3/6] Clustering (HDBSCAN)...")
    labels = _clustering_hdbscan(X_reduced)

    # ── 4. Analizar clusters ─────────────────────────────────────────
    print(f"\n[4/6] Analizando clusters...")
    df_analisis = _analizar_clusters(df, labels)

    # ── 5. Detectar mislabels ────────────────────────────────────────
    print(f"\n[5/6] Detectando posibles mislabels...")
    estados = _detectar_mislabels(df, labels, df_analisis)
    df["estado_limpieza"] = estados
    df["cluster"] = labels

    # Resumen
    conteo = estados.value_counts()
    total = len(df)
    print(f"\n  Resultado de limpieza:")
    for estado, n in conteo.items():
        pct = 100 * n / total
        emoji = {"ok": "✅", "mislabel": "❌", "ruido": "🔇", "revisar": "⚠️ "}.get(estado, "?")
        print(f"    {emoji} {estado:<12} {n:>6} ({pct:.1f}%)")

    # ── 6. Guardar ───────────────────────────────────────────────────
    print(f"\n[6/6] Guardando resultados...")

    ext = ".parquet" if formato == "parquet" else ".csv"

    # Dataset completo con etiquetas de limpieza
    out_completo = os.path.join(output_dir, f"embeddings_con_limpieza{ext}")
    if formato == "parquet":
        df.to_parquet(out_completo, index=False)
    else:
        df.to_csv(out_completo, index=False)
    print(f"  ✓ Completo (con etiquetas): {out_completo}")

    # Dataset limpio (solo "ok")
    df_limpio = df[df["estado_limpieza"] == "ok"].copy()
    out_limpio = os.path.join(output_dir, f"embeddings_limpios{ext}")
    if formato == "parquet":
        df_limpio.to_parquet(out_limpio, index=False)
    else:
        df_limpio.to_csv(out_limpio, index=False)
    print(f"  ✓ Limpio (solo 'ok'): {out_limpio} ({len(df_limpio)} segmentos)")

    # Análisis de clusters
    out_analisis = os.path.join(output_dir, f"analisis_clusters{ext}")
    if formato == "parquet":
        df_analisis.to_parquet(out_analisis, index=False)
    else:
        df_analisis.to_csv(out_analisis, index=False)
    print(f"  ✓ Análisis: {out_analisis}")

    # ── Visualización ────────────────────────────────────────────────
    if visualizar:
        print(f"\n  Generando visualizaciones (esto puede tomar un momento)...")
        X_2d = _reducir_umap_2d(X_scaled)
        _visualizar(df, X_2d, labels, estados, output_dir)

    # ── Resumen final ────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  RESUMEN FASE 2")
    print(f"{'═' * 60}")
    print(f"    Segmentos entrada:       {total}")
    print(f"    Clusters encontrados:    {(labels != -1).max() + 1 if len(labels) > 0 else 0}")
    print(f"    Segmentos OK:            {conteo.get('ok', 0)}")
    print(f"    Posibles mislabels:      {conteo.get('mislabel', 0)}")
    print(f"    Ruido:                   {conteo.get('ruido', 0)}")
    print(f"    Para revisar:            {conteo.get('revisar', 0)}")
    print(f"    Tasa de conservación:    "
          f"{100 * conteo.get('ok', 0) / total:.1f}%")
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
        description="Fase 2: Limpieza por clustering (UMAP + HDBSCAN)"
    )
    parser.add_argument("--input", required=True,
                        help="Ruta al embeddings_consolidado de Fase 1")
    parser.add_argument("--output_dir", default=None,
                        help="Carpeta de salida (default: misma que input)")
    parser.add_argument("--formato", choices=["csv", "parquet"], default="csv")
    parser.add_argument("--visualizar", action="store_true",
                        help="Generar gráficas PNG")
    parser.add_argument("--min_cluster", type=int, default=HDBSCAN_MIN_CLUSTER,
                        help=f"Min puntos por cluster (default: {HDBSCAN_MIN_CLUSTER})")
    parser.add_argument("--min_samples", type=int, default=HDBSCAN_MIN_SAMPLES,
                        help=f"Min samples HDBSCAN (default: {HDBSCAN_MIN_SAMPLES})")
    args = parser.parse_args()

    # Permitir ajustar parámetros desde CLI
    HDBSCAN_MIN_CLUSTER = args.min_cluster
    HDBSCAN_MIN_SAMPLES = args.min_samples

    df_limpio, df_analisis = limpiar_por_clustering(
        input_path=args.input,
        output_dir=args.output_dir,
        formato=args.formato,
        visualizar=args.visualizar,
    )