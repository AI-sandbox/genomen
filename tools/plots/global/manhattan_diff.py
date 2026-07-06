import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire

from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_shap_diff_gwas(
    linear_shap_df: pd.DataFrame,
    non_linear_shap_df: pd.DataFrame,
    phenotype: str,
    shap_col: str = "shap_values",
    chr_col: str = "chr_name",
    pos_col: str = "chr_position",
    snp_col: str = "snp",
    how: str = "inner",
    log_scale: bool = False,
    log_base: float = 10.0,
    symlog_linthresh: float = 1e-3,
    point_size: float = 6.0,
    alpha: float = 0.8,
    figsize: Tuple[float, float] = (12.0, 5.0),
    annotate: bool = False,
    top_k: int = 10,
    peak_window: int = 0,  # bp window to group peak "sets"
):
    """
    Align two SHAP DataFrames by chromosome and position, compute
    Δ = |SHAP_non_linear| - |SHAP_linear|, and plot a Manhattan-style GWAS scatter.

    Parameters
    ----------
    snp_col : str
        Column containing SNP identifier (used when annotate=True).
    annotate : bool
        If True, annotate up to `top_k` variants by |Δ|, with max 3 labels per peak set.
    top_k : int
        Number of top variants (by |Δ|) to consider for annotation.
    peak_window : int
        Proximity window (in base pairs) for grouping labels into peak "sets".
    """
    # Copy to avoid modifying inputs
    lin = linear_shap_df[
        [chr_col, pos_col, shap_col] + ([snp_col] if snp_col in linear_shap_df.columns else [])
    ].copy()
    nonlin = non_linear_shap_df[
        [chr_col, pos_col, shap_col] + ([snp_col] if snp_col in non_linear_shap_df.columns else [])
    ].copy()
    lin.rename(columns={shap_col: f"{shap_col}_linear"}, inplace=True)
    nonlin.rename(columns={shap_col: f"{shap_col}_non_linear"}, inplace=True)

    # Ensure position is numeric
    for df in (lin, nonlin):
        df[pos_col] = pd.to_numeric(df[pos_col], errors="coerce")

    # Clean chromosome labels and define a stable numeric order
    def normalize_chr(val):
        """Return both a cleaned display label and a numeric order key."""
        s = str(val)
        s_clean = re.sub(r"^(?i:chr|chrom)", "", s).strip()
        s_clean = s_clean.upper()
        mapping = {"X": 23, "Y": 24, "M": 25, "MT": 25, "MITO": 25}
        if s_clean in mapping:
            order = mapping[s_clean]
            label = "MT" if order == 25 else s_clean
        else:
            try:
                order = int(re.match(r"^\d+", s_clean).group())
                label = str(order)
            except Exception:
                order = 100
                label = s_clean
        return label, order

    for df in (lin, nonlin):
        lbl_ord = df[chr_col].apply(normalize_chr)
        df["_chr_label"] = lbl_ord.map(lambda x: x[0])
        df["_chr_order"] = lbl_ord.map(lambda x: x[1])

    # Merge
    merged = pd.merge(
        lin,
        nonlin,
        on=[chr_col, pos_col]
        + ([snp_col] if (snp_col in lin.columns and snp_col in nonlin.columns) else []),
        how=how,
        suffixes=("_linear", "_non_linear"),
        validate="m:m",
    )

    # Recompute normalized labels/order on merged (if how='outer' is used)
    lbl_ord = merged[chr_col].apply(normalize_chr)
    merged["_chr_label"] = lbl_ord.map(lambda x: x[0])
    merged["_chr_order"] = lbl_ord.map(lambda x: x[1])

    # Compute delta = |non-linear| - |linear|
    merged["delta_shap"] = (
        merged[f"{shap_col}_non_linear"].abs() - merged[f"{shap_col}_linear"].abs()
    )

    # Drop rows with missing essentials
    merged = merged.dropna(subset=[pos_col, "delta_shap", "_chr_order"]).copy()

    # Sort within chromosome by position
    merged.sort_values(by=["_chr_order", pos_col], inplace=True)

    # Build cumulative position per chromosome for Manhattan plot
    chr_groups = merged.groupby("_chr_order", sort=True)
    chr_sizes = chr_groups[pos_col].max().sort_index().fillna(0)

    offsets: Dict[int, float] = {}
    cumulative = 0.0
    for order, size in chr_sizes.items():
        offsets[order] = cumulative
        cumulative += float(size) if np.isfinite(size) else 0.0

    merged["pos_cum"] = merged.apply(
        lambda r: float(r[pos_col]) + offsets.get(int(r["_chr_order"]), 0.0), axis=1
    )

    tick_positions: List[float] = []
    tick_labels: List[str] = []
    for order, grp in merged.groupby("_chr_order", sort=True):
        if grp.empty:
            continue
        center = float(grp["pos_cum"].median())
        label = grp["_chr_label"].mode().iat[0]
        tick_positions.append(center)
        tick_labels.append(label)

    # Colors by chromosome
    unique_orders = sorted(merged["_chr_order"].unique())
    cmap = plt.get_cmap("tab20")
    color_map = {order: cmap(i % 20) for i, order in enumerate(unique_orders)}
    merged["_color"] = merged["_chr_order"].map(color_map)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        merged["pos_cum"].to_numpy(),
        merged["delta_shap"].to_numpy(),
        s=point_size,
        c=merged["_color"].to_numpy(),
        alpha=alpha,
        linewidths=0,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=0)
    ax.set_xlim(float(merged["pos_cum"].min()), float(merged["pos_cum"].max()))
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(
        f"{'log(' if log_scale else ''}|SHAP (non-linear)| − |SHAP (linear)|{')' if log_scale else ''}"
    )
    ax.set_title(f"ΔSHAP (non-linear vs linear) for phenotype {phenotype}")

    if log_scale:
        ax.set_yscale("symlog", base=log_base, linthresh=symlog_linthresh)

    for _, grp in merged.groupby("_chr_order", sort=True):
        x_max = float(grp["pos_cum"].max())
        ax.axvline(x_max, color="lightgray", linewidth=0.5, alpha=0.5)

    # --- Annotations (optional) ---
    if annotate:
        if snp_col not in merged.columns:
            logger.warning(
                f"annotate=True but '{snp_col}' column not found in merged dataframe; skipping labels."
            )
        else:
            # pick top variants by absolute delta magnitude
            merged["_delta_abs"] = merged["delta_shap"]
            top = merged.nlargest(top_k, "_delta_abs").copy()

            # group into "peak sets" by chromosome and proximity window
            # We'll sweep left-to-right and create groups whose centers are updated by running mean.
            groups: Dict[int, List[Dict]] = {}  # chr_order -> list of groups
            top = top.sort_values(
                by=["_chr_order", pos_col, "_delta_abs"], ascending=[True, True, False]
            )

            def assign_group(chr_order: int, pos_bp: float):
                if chr_order not in groups:
                    groups[chr_order] = []
                for g in groups[chr_order]:
                    if abs(pos_bp - g["center"]) <= peak_window:
                        # update group center (weighted by count) for stability
                        g["count"] += 1
                        g["center"] = (g["center"] * (g["count"] - 1) + pos_bp) / g["count"]
                        return g
                # create new group
                g = {"center": pos_bp, "count": 1, "labels": 0}
                groups[chr_order].append(g)
                return g

            # annotate up to 3 per group, prioritizing larger |Δ|
            # Also spread labels vertically and horizontally to reduce collisions.
            # Offsets cycle through a small set of patterns.
            x_offsets = np.array(
                [8e6, -8e6, 0.0, 12e6, -12e6]
            )  # in bp (converted via pos_cum by local scale)
            y_offsets = np.array([0.15, 0.2, -0.18, -0.25, 0.22])  # fraction of y-range
            y_range = float(merged["delta_shap"].max() - merged["delta_shap"].min()) or 1.0

            # map from bp to pos_cum scale: pos_cum increases by bp within a chromosome since we used offsets
            # So we can add/subtract directly to pos_cum using the same bp offsets.

            idx = 0
            for _, r in top.iterrows():
                grp = assign_group(int(r["_chr_order"]), float(r[pos_col]))
                if grp["labels"] >= 3:
                    continue  # limit 3 labels per peak set
                snp_label = str(r[snp_col])
                x = float(r["pos_cum"])
                y = float(r["delta_shap"])

                xo = x_offsets[idx % len(x_offsets)]
                yo = y_offsets[idx % len(y_offsets)] * y_range
                idx += 1

                # Keep label readable: white bbox, small font, arrow
                ax.annotate(
                    snp_label,
                    xy=(x, y),
                    xytext=(x + xo, y + yo),
                    textcoords="data",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.8, alpha=0.7),
                    ha="left" if xo >= 0 else "right",
                    va="bottom" if yo >= 0 else "top",
                    clip_on=False,
                )
                grp["labels"] += 1

    fig.tight_layout()
    return merged, (fig, ax)


def main(
    linear_data_dir: Path | str,
    non_linear_data_dir: Path | str,
    task_ids: List[int] | int = list(range(1, 21)),
    shap_col: str = "shap_values",
    chr_col: str = "chr_name",
    pos_col: str = "chr_position",
    snp_col: str = "snp",
    log_scale: bool = False,
    annotate: bool = False,
    top_k: int = 10,
    peak_window: int = 1_000_000,
    point_size: float = 6.0,
    alpha: float = 0.8,
    width: float = 12.0,
    height: float = 5.0,
):
    if isinstance(task_ids, int):
        task_ids = [task_ids]

    linear_data_dir = Path(linear_data_dir)
    non_linear_data_dir = Path(non_linear_data_dir)
    if not linear_data_dir.exists() or not non_linear_data_dir.exists():
        raise ValueError("Not all necessary data files exist.")

    for task_id in task_ids:
        phenotype_name, phenotype_id = utils.setup(task_id)
        logger.info(
            f"Plotting global shap values for phenotype {phenotype_name} ({phenotype_id})..."
        )

        linear_path = linear_data_dir / f"{phenotype_name}_annotation_df.parquet"
        non_linear_path = non_linear_data_dir / f"{phenotype_name}_annotation_df.parquet"

        if not linear_path.is_file():
            logger.warning(f"Could not find linear annotation file for phenotype {phenotype_name}")
            continue
        elif not non_linear_path.is_file():
            logger.warning(
                f"Could not find non-linear annotation file for phenotype {phenotype_name}"
            )
            continue
        else:
            # Load inputs (Parquet as per your setup)
            df_lin = pd.read_parquet(linear_path)
            df_nonlin = pd.read_parquet(non_linear_path)

        merged, (fig, ax) = plot_shap_diff_gwas(
            linear_shap_df=df_lin,
            non_linear_shap_df=df_nonlin,
            phenotype=phenotype_name,
            shap_col=shap_col,
            chr_col=chr_col,
            pos_col=pos_col,
            snp_col=snp_col,
            log_scale=log_scale,
            point_size=point_size,
            alpha=alpha,
            figsize=(width, height),
            annotate=annotate,
            top_k=top_k,
            peak_window=peak_window,
        )

        out_dir = non_linear_data_dir / "plots" / "manhattan_diff"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_plot_path = out_dir / f"{phenotype_name}.png"
        logger.info(f"Saving plot for phenotype {phenotype_name} to {out_plot_path}")
        fig.savefig(out_plot_path, bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    Fire(main)
