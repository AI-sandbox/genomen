import logging
import re
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .. import utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_shap_embedding(
    local_df,
    phenotype_name: str,
    out_path: Path | str,
    method: str = "umap",            # "umap" | "tsne" | "pca"
    n_components: int = 2,           # 2 or 3
    labels=None,                     # Optional 1D array-like of class/cluster labels (for coloring only)
    standardize: bool = True,        # Standardize features before DR
    pca_components: int = 50,        # PCA precompress for tsne/umap
    tsne_perplexity: float = 30.0,
    tsne_learning_rate: float = "auto",
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    metric: str = "euclidean",       # or "cosine" (often good for SHAP)
    random_state: int = 42,
    figsize=(6, 5),
    # NEW: clustering controls
    cluster_method: str = "kmeans",  # "dbscan" | "kmeans" | None
    cluster_params: dict | None = None,
):
    """
    Reduce dimensionality of a SHAP value matrix, plot the embedding,
    and return a DataFrame mapping each sample to a discovered cluster.

    Returns
    -------
    clusters_df : pandas.DataFrame with columns:
        - 'cluster' : cluster id (NaN for noise if DBSCAN)
        - 'x','y'[, 'z'] : embedding coordinates
        - 'label' : optional provided label (for reference)
    embedding : np.ndarray (n_samples, n_components)
    fig, ax : matplotlib Figure and Axes (if n_components==2)
              or (Figure, Ax3D Axes) if n_components==3
    """
    # ---- Prepare X and sample index
    X = local_df.to_numpy()
    sample_index = local_df.index.values

    if X.ndim != 2:
        raise ValueError("shap_values must be a 2D array of shape (n_samples, n_features).")
    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3.")

    # Standardize (SHAP magnitudes can vary widely across features)
    if standardize:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # ---- Dimensionality reduction
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        embedding = reducer.fit_transform(X)

    elif method == "tsne":
        X_in = X
        if X.shape[1] > pca_components:
            X_in = PCA(n_components=pca_components, random_state=random_state).fit_transform(X)
        reducer = TSNE(
            n_components=n_components,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            init="pca",
            metric=metric,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(X_in)

    elif method == "umap":
        X_in = X
        if X.shape[1] > pca_components:
            X_in = PCA(n_components=pca_components, random_state=random_state).fit_transform(X)
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=metric,
            random_state=random_state,
        )
        embedding = reducer.fit_transform(X_in)
    else:
        raise ValueError("method must be one of {'umap','tsne','pca'}.")

    # ---- Clustering on the embedding
    # Defaults chosen for 2D embeddings; override via cluster_params
    cl_params = dict(cluster_params or {})
    cluster_ids = None

    if cluster_method is None:
        pass  # No clustering
    elif cluster_method.lower() == "dbscan":
        # Reasonable defaults for 2D embeddings; tweak eps/min_samples as needed
        eps = cl_params.pop("eps", 0.5)
        min_samples = cl_params.pop("min_samples", 5)
        db = DBSCAN(eps=eps, min_samples=min_samples, **cl_params)
        cluster_ids = db.fit_predict(embedding)
        # Convert DBSCAN noise label -1 -> NaN
        cluster_ids = pd.Series(cluster_ids).replace({-1: np.nan}).to_numpy()
    elif cluster_method.lower() == "kmeans":
        n_clusters = cl_params.pop("n_clusters", 8)
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state, **cl_params)
        cluster_ids = km.fit_predict(embedding)
    else:
        raise ValueError("cluster_method must be one of {'dbscan','kmeans', None}.")

    # ---- Build result DataFrame
    cols = ["x", "y"] if n_components == 2 else ["x", "y", "z"]
    emb_df = pd.DataFrame(embedding, index=sample_index, columns=cols)
    if labels is not None:
        emb_df["label"] = np.asarray(labels)
    if cluster_ids is not None:
        emb_df["cluster"] = cluster_ids
        # Optional: order rows by cluster then index (so samples are "sorted to a cluster")
        emb_df = emb_df.sort_values(by=["cluster"] + ([] if n_components == 2 else []) + [], kind="mergesort")
        # mergesort keeps a stable order within clusters

    # ---- Plot
    fig = plt.figure(figsize=figsize)
    if n_components == 2:
        ax = fig.add_subplot(111)
        if cluster_ids is not None:
            # Color by discovered clusters; NaN (noise) shown as a separate scatter
            clusters = pd.Series(emb_df["cluster"], index=emb_df.index)
            mask_noise = clusters.isna()
            # Plot clustered points
            for cid in sorted(clusters.dropna().unique()):
                m = clusters == cid
                ax.scatter(emb_df.loc[m, "x"], emb_df.loc[m, "y"], s=12, label=f"cluster {int(cid)}")
            # Plot noise (if any)
            if mask_noise.any():
                ax.scatter(emb_df.loc[mask_noise, "x"], emb_df.loc[mask_noise, "y"], s=12, label="noise")
        else:
            # Fallback coloring with provided labels (if any)
            if labels is None:
                ax.scatter(emb_df["x"], emb_df["y"], s=12)
            else:
                labs = np.asarray(labels)
                for lab in np.unique(labs):
                    m = labs == lab
                    ax.scatter(emb_df.iloc[m, 0], emb_df.iloc[m, 1], s=12, label=str(lab))
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        ttl = f"{method.upper()} of local shap values for phenotype {phenotype_name}"
        ax.set_title(ttl)
        if (cluster_ids is not None) or (labels is not None):
            ax.legend(title="Group", frameon=False)
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection="3d")
        if cluster_ids is not None:
            clusters = pd.Series(emb_df["cluster"], index=emb_df.index)
            mask_noise = clusters.isna()
            for cid in sorted(clusters.dropna().unique()):
                m = clusters == cid
                ax.scatter(emb_df.loc[m, "x"], emb_df.loc[m, "y"], emb_df.loc[m, "z"], s=12, label=f"cluster {int(cid)}")
            if mask_noise.any():
                ax.scatter(emb_df.loc[mask_noise, "x"], emb_df.loc[mask_noise, "y"], emb_df.loc[mask_noise, "z"], s=12, label="noise")
        else:
            if labels is None:
                ax.scatter(emb_df["x"], emb_df["y"], emb_df["z"], s=12)
            else:
                labs = np.asarray(labels)
                for lab in np.unique(labs):
                    m = labs == lab
                    ax.scatter(emb_df.iloc[m, 0], emb_df.iloc[m, 1], emb_df.iloc[m, 2], s=12, label=str(lab))
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
        ttl = f"SHAP {method.upper()} embedding (3D)"
        ttl += " + clustering" if cluster_ids is not None else ""
        ax.set_title(ttl)
        if (cluster_ids is not None) or (labels is not None):
            ax.legend(title="Group", frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    return emb_df

def plot_gwas_from_local_shap(
    local_shap_df: pd.DataFrame,         # rows: sample_idx ; cols: variant_idx
    clusters: List[Sequence[int]],       # list of arrays/lists of sample_idx (one per cluster)
    annotation_df: pd.DataFrame,
    out_path: Path | str,
    *,
    chr_col: str = "chr_name",
    pos_col: str = "chr_position",
    cluster_labels: Optional[Sequence[str]] = None,
    title: str = "GWAS by cluster (mean SHAP)",
    y_label: str = "Mean SHAP",
    figsize: Tuple[float, float] = (20, 5),
    point_size: float = 6,
    alpha: float = 0.7,
    colors: Optional[Sequence[str]] = None,
    only_chr: Optional[Union[str, int]] = None,   # if provided, plot a single chromosome
    # --- NEW ---
    annotate: bool = False,
    top_k: int = 10,
    label_col: Optional[str] = "snp",              # e.g., "snp" in annotation_df
    max_labels_per_chr: int = 5,                  # soft cap per chromosome for GW-wide mode
):
    """
    Overlay multiple GWAS (mean SHAP per cluster) in one plot.

    Annotation mode:
        If annotate=True, label up to `top_k` variants whose BETWEEN-CLUSTER
        mean SHAP differs the most (range across clusters).
        Labels use `annotation_df[label_col]` if provided/available,
        otherwise the variant index.
    """
    # ---- Validate annotation columns
    required_cols = {chr_col, pos_col}
    if not required_cols.issubset(annotation_df.columns):
        missing = required_cols - set(annotation_df.columns)
        raise ValueError(f"annotation_df missing columns: {sorted(missing)}")

    # Work on a copy
    ann = annotation_df[[chr_col, pos_col] + ([label_col] if label_col and label_col in annotation_df.columns else [])].copy()

    # --- Align variants between SHAP columns and annotation index
    shap_cols = pd.Index(local_shap_df.columns)
    ann_idx = ann.index

    common = shap_cols.intersection(ann_idx)

    # Fallback: try string alignment if types differ (e.g., int vs str ids)
    if len(common) == 0:
        shap_cols_str = shap_cols.astype(str)
        ann_idx_str = ann_idx.astype(str)
        common_str = shap_cols_str.intersection(ann_idx_str)
        if len(common_str) == 0:
            raise ValueError(
                "No overlapping variants between annotation_df.index and local_shap_df.columns"
            )
        # map string → original dtypes
        shap_map = {str(c): c for c in shap_cols}
        # reduce to common in original SHAP dtype
        common = pd.Index([shap_map[s] for s in common_str])
        # also reduce annotation to common (via string)
        ann = ann[ann_idx_str.isin(common_str)]

    # keep only overlapping variants, then (optionally) filter to a chromosome
    ann = ann.loc[common]

    # normalize chromosome labels
    def _norm_chr(x):
        s = str(x).strip()
        s = re.sub(r"^chr", "", s, flags=re.IGNORECASE).upper()
        if s == "23": s = "X"
        elif s == "24": s = "Y"
        elif s in {"M", "MT", "MITO"}: s = "MT"
        return s

    ann["_chr"] = ann[chr_col].map(_norm_chr)

    if only_chr is not None:
        target_chr = _norm_chr(only_chr)
        ann = ann[ann["_chr"] == target_chr]
        if ann.empty:
            raise ValueError(f"No variants found for chromosome '{only_chr}'")

    # numeric pos and ordering
    ann[pos_col] = pd.to_numeric(ann[pos_col], errors="coerce")
    ann = ann.dropna(subset=[pos_col, "_chr"])

    def _order_key(s):
        if s == "X": return 23
        if s == "Y": return 24
        if s == "MT": return 25
        try:
            return int(s)
        except Exception:
            return np.nan

    ann["_chr_order"] = ann["_chr"].map(_order_key)
    ann = ann.dropna(subset=["_chr_order"]).sort_values(by=["_chr_order", pos_col])

    # final variant order to plot
    variant_order = ann.index.tolist()

    # align SHAP to these variants
    shap_aligned = local_shap_df.loc[:, [v for v in variant_order if v in local_shap_df.columns]]

    # ---- Compute mean SHAP per variant for each cluster
    if cluster_labels is None:
        cluster_labels = [f"Cluster {i+1}" for i in range(len(clusters))]
    if len(cluster_labels) != len(clusters):
        raise ValueError("cluster_labels length must match clusters length.")

    cluster_means = {}
    for label, sample_ids in zip(cluster_labels, clusters):
        sel = pd.Index(sample_ids).intersection(shap_aligned.index)
        if len(sel) == 0:
            cluster_means[label] = pd.Series(index=variant_order, dtype=float)
        else:
            # compute mean along samples (rows) → per-variant series
            m = shap_aligned.loc[sel].mean(axis=0)
            # reindex to variant_order to keep strict plotting order
            cluster_means[label] = m.reindex(variant_order)

    # ---- Build plotting frame
    d = ann[["_chr", "_chr_order", pos_col] + ([label_col] if label_col and label_col in ann.columns else [])].copy()
    d.index.name = "variant_idx"

    # x-axis
    if only_chr is not None:
        d["_x"] = d[pos_col].astype(float)
    else:
        chr_unique = pd.Index(d["_chr"]).drop_duplicates().tolist()
        chr_max_pos = d.groupby("_chr", sort=False)[pos_col].max().reindex(chr_unique)
        gap = max(int(0.005 * (d[pos_col].max() or 1)), 1_000_000)
        prev_lengths = chr_max_pos.shift(fill_value=0).cumsum()
        gaps = pd.Series(np.arange(len(chr_max_pos)), index=chr_max_pos.index) * gap
        offsets = (prev_lengths + gaps).astype(np.int64)
        d["_offset"] = d["_chr"].map(offsets)
        d["_x"] = d[pos_col].astype(float) + d["_offset"].astype(float)

    # colors per cluster
    if colors is None:
        base = list(plt.cm.tab10.colors)
        colors = [base[i % len(base)] for i in range(len(cluster_labels))]
    color_map = dict(zip(cluster_labels, colors))

    # ---- Plot
    fig, ax = plt.subplots(figsize=figsize)
    for label in cluster_labels:
        y = pd.to_numeric(cluster_means[label], errors="coerce")
        dd = d.copy()
        dd["y"] = y.values
        dd = dd.dropna(subset=["y"])
        if dd.empty:
            continue
        ax.scatter(
            dd["_x"], dd["y"],
            s=point_size, alpha=alpha, linewidths=0, color=color_map[label], label=label
        )

    ax.set_title(title)
    ax.set_ylabel(y_label)

    if only_chr is not None:
        ax.set_xlabel("Position (bp)")
        ax.margins(x=0.02)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(axis="x", visible=False)
    else:
        chr_unique = pd.Index(d["_chr"]).drop_duplicates().tolist()
        grp = d.groupby("_chr")["_x"]
        tick_pos = {chrom: int((grp.min().loc[chrom] + grp.max().loc[chrom]) / 2) for chrom in chr_unique}

        def _label(s):
            return "X" if s in ("23", "X") else ("Y" if s in ("24", "Y") else ("MT" if s in ("25","MT") else str(s)))

        ax.set_xlabel("Chromosome")
        ax.set_xticks([tick_pos[c] for c in chr_unique])
        ax.set_xticklabels([_label(c) for c in chr_unique])

        x_left, x_right = d["_x"].min(), d["_x"].max()
        pad = max(int(0.005 * (d[pos_col].max() or 1)), 1_000_000) // 2
        ax.set_xlim(x_left - pad, x_right + pad)
        ax.margins(x=0.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.grid(axis="x", visible=False)

    # ---------- NEW: Annotation of top variants by between-cluster difference ----------
    if annotate and len(cluster_means) >= 2:
        # Build matrix: rows = variant_idx in plotting order; cols = clusters
        Y = pd.DataFrame({lbl: pd.to_numeric(cluster_means[lbl], errors="coerce") for lbl in cluster_labels})
        Y = Y.loc[variant_order]  # ensure plotting order
        # Compute range across clusters per variant (handles NaNs by skipping them)
        diff = Y.max(axis=1, skipna=True) - Y.min(axis=1, skipna=True)
        diff = diff.fillna(-np.inf)  # ensure NaNs don't get picked

        # Select top_k variants
        top_variants = diff.nlargest(top_k)
        if not top_variants.empty:
            # For each selected variant, choose the cluster giving the extreme value for anchor point
            y_range = (pd.concat([Y.min(axis=1), Y.max(axis=1)]).max() -
                       pd.concat([Y.min(axis=1), Y.max(axis=1)]).min())
            y_range = float(y_range) if np.isfinite(y_range) and y_range != 0 else 1.0

            # Offsets to help reduce label collisions
            x_offsets = np.array([0.0, 0.6, -0.6, 1.2, -1.2, 1.8, -1.8, 2.4, -2.4, 3.0])  # in millions of bp (approx)
            y_offsets = np.array([0.18, -0.22, 0.24, -0.20, 0.16, -0.26, 0.28, -0.18, 0.22, -0.24]) * y_range

            # Prepare label strings
            if label_col and label_col in d.columns:
                labels_series = d[label_col].astype(str)
            else:
                labels_series = pd.Series(d.index.astype(str).values, index=d.index)

            used_per_chr = {}  # soft cap per chromosome (GW mode)

            for i, (var_idx, delta) in enumerate(top_variants.items()):
                # Skip if chromosome cap exceeded (GW mode); no cap in single-chr mode
                chr_i = d.loc[var_idx, "_chr"]
                if only_chr is None:
                    cnt = used_per_chr.get(chr_i, 0)
                    if cnt >= max_labels_per_chr:
                        continue

                # Anchor at the cluster with the larger absolute mean (more visually separated)
                row = Y.loc[var_idx]
                # Prefer the max absolute value; if tie, fall back to max
                abs_choice = row.abs().idxmax()
                y_anchor = row[abs_choice]
                if pd.isna(y_anchor):
                    # fallback to max (ignoring NaNs)
                    y_anchor = row.max(skipna=True)

                x_anchor = float(d.loc[var_idx, "_x"])
                label_text = str(labels_series.loc[var_idx])

                # Convert x-offset from "millions of bp" to x-scale:
                # In genome-wide view, _x is bp+offsets; in single-chr it's bp -> same unit.
                xo = x_offsets[i % len(x_offsets)] * 1e6
                yo = float(y_offsets[i % len(y_offsets)])

                ax.annotate(
                    label_text,
                    xy=(x_anchor, y_anchor),
                    xytext=(x_anchor + xo, y_anchor + yo),
                    textcoords="data",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.75),
                    ha="left" if xo >= 0 else "right",
                    va="bottom" if yo >= 0 else "top",
                    clip_on=False,
                )

                if only_chr is None:
                    used_per_chr[chr_i] = used_per_chr.get(chr_i, 0) + 1

    ax.legend(title="Clusters", markerscale=1.5, frameon=False, ncol=min(4, len(cluster_labels)))
    plt.tight_layout()
    fig.savefig(out_path)


def main(
    local_data_dir: Path | str,
    annotation_data_dir: Path | str = None,
    standardize: bool = False,
    task_ids: List[int] | int = list(range(1, 13)),
    method: Literal["umap", "tsne", "pca"] = "pca",
    plot_manhattan_diff: bool = False,
    annotate: bool = False,
    n_clusters: int = 2,
    **kwargs
):
    if isinstance(task_ids, int):
        task_ids = [task_ids]

    local_data_dir = Path(local_data_dir)
    if not local_data_dir.exists():
        raise ValueError("Provided local_data_dir does not exist.")
    plot_path = local_data_dir / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    if plot_manhattan_diff:
        if annotation_data_dir is None:
            raise ValueError("Have to provide annotation df to plot manhattan plot")
        annotation_data_dir = Path(annotation_data_dir)
        if not annotation_data_dir.exists():
            raise ValueError("Provided annotation_data_dir does not exist.")
        

    for task_id in task_ids:
        phenotype_name, phenotype_id = utils.setup(task_id)
        logger.info(f"Plotting local shap values for phenotype {phenotype_name} ({phenotype_id})...")

        phenotype_input_path = local_data_dir / f"local_shap_{phenotype_id}.parquet"
        if not phenotype_input_path.is_file():
            logger.warning(f"Could not find local shap file for phenotype {phenotype_name}. Skipping...")
            continue   
        reduc_output_path = plot_path / f"local_shap_reduc_plot_{method}_{phenotype_name}.png"

        local_shap_df = pd.read_parquet(phenotype_input_path)
        logger.info(f"Plotting local shap for {len(local_shap_df)} samples.")

        try:
            cluster_df = plot_shap_embedding(
                local_shap_df,
                phenotype_name,
                reduc_output_path,
                method,
                standardize=standardize,
                cluster_params={"n_clusters": n_clusters} if n_clusters else None,
            )
            logger.info(f"Dim. reduction done. Plot can be found under {reduc_output_path}.")
        except Exception as e:
            logger.error(f"Failed to plot SHAP embedding for {phenotype_name}: {e}")
            continue

        if plot_manhattan_diff:
            phenotype_annotation_path = annotation_data_dir / f"{phenotype_name}_annotation_df.parquet"
            if not phenotype_annotation_path.is_file():
                logger.warning(f"Could not find annotation file for phenotype {phenotype_name}. Skipping...")
                continue    
            else:
                annotation_df = pd.read_parquet(phenotype_annotation_path)

            cluster_idxs = []
            clusters = cluster_df["cluster"].unique()
            for cluster_id in clusters:
                local_cluster_df = cluster_df[cluster_df["cluster"] == cluster_id]
                local_cluster_idxs = local_cluster_df.index.values
                cluster_idxs.append(local_cluster_idxs)

            manhattan_output_path = plot_path / f"local_shap_manhattan_plot_{method}_{phenotype_name}.png"
            plot_gwas_from_local_shap(
                local_shap_df,
                cluster_idxs,
                annotation_df,
                manhattan_output_path,
                annotate=annotate
            )   
            logger.info(f"Manhattan diff. plot done. Plot can be found under {manhattan_output_path}.")


if __name__ == "__main__":
    fire.Fire(main)
