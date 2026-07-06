import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire
from matplotlib import gridspec
from matplotlib.colors import Normalize

from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_gwas_with_interactions(
    gwas_df: pd.DataFrame,
    inter_df: pd.DataFrame,
    phenotype: str,
    *,
    # GWAS (Manhattan) params
    chr_col: str = "chr_name",
    pos_col: str = "chr_position",
    y_col: str = "shap_values",
    y_label: str = "Shap values",
    point_size: float = 6,
    alpha: float = 0.8,
    colors: Optional[Sequence[str]] = None,
    only_chr: Optional[Union[str, int]] = None,
    # Interaction-triangle params
    chr_i_col: str = "chr_i",
    chr_j_col: str = "chr_j",
    pos_i_col: str = "pos_i",
    pos_j_col: str = "pos_j",
    val_col: str = "interact_values",
    triangle_title: Optional[str] = None,
    vmax: Optional[float] = None,  # if None, will be set from data
    bins_per_chr: Optional[int] = 120,  # cap number of position bins per chromosome (auto if None)
    bin_size_bp: Optional[int] = None,  # alternatively fix bin size in base pairs
    cmap: str = "viridis",
    show_colorbar: bool = False,
    # Layout
    figsize: Tuple[float, float] = (20, 8),
    height_ratios: Tuple[float, float] = (3.0, 1.2),  # [Manhattan, triangles]
    bottom_pad: float = 0.25,  # extra bottom space (title/labels)
) -> Dict[str, Any]:
    """
    Draws a Manhattan plot on the top row and, underneath, one triangular
    interaction heatmap per chromosome (upper triangle), aligned left-to-right.

    Parameters
    ----------
    gwas_df : DataFrame
        Columns: [chr_col, pos_col, y_col]
    inter_df : DataFrame
        Columns: [chr_i_col, chr_j_col, pos_i_col, pos_j_col, val_col]
        All missing pairs are assumed 0.0 (sparse).
        Only entries with chr_i == chr_j are rendered (within-chromosome triangles).
    only_chr : str|int|None
        If provided, restricts both plots to that chromosome.
    bins_per_chr : int|None
        If provided, adaptively bins positions to at most this many bins per chromosome.
        Use this to keep triangles reasonably sized.
    bin_size_bp : int|None
        If provided, uses fixed bin width (bp). If both given, bin_size_bp takes precedence.
    vmax : float|None
        Upper bound for color scale (vmin=0). If None, inferred across all chrom.

    Returns
    -------
    dict with keys:
        - fig: matplotlib Figure
        - ax_top: Manhattan axis
        - ax_tris: dict mapping chrom -> Axes for that triangle
        - chrom_order: ordered list of chromosomes
        - chrom_offsets: mapping chrom -> x-offset used for Manhattan alignment
        - chrom_lengths: mapping chrom -> max position per chromosome
    """

    # ----------------------------
    # Helpers (match your GWAS code)
    # ----------------------------
    def _norm_chr(x):
        s = str(x).strip()
        s = re.sub(r"^chr", "", s, flags=re.IGNORECASE).upper()
        if s == "23":
            s = "X"
        elif s == "24":
            s = "Y"
        return s

    def _order_key(s):
        if s == "X":
            return 23
        if s == "Y":
            return 24
        if s in {"MT", "M"}:
            return 25
        try:
            return int(s)
        except Exception:
            return np.nan

    # Minimal copy/clean for GWAS df
    d = gwas_df[[chr_col, pos_col, y_col]].copy()
    d["_chr"] = d[chr_col].map(_norm_chr)

    if only_chr is not None:
        target = _norm_chr(only_chr)
        d = d[d["_chr"] == target]

    d["_chr_order"] = d["_chr"].map(_order_key)
    d[pos_col] = pd.to_numeric(d[pos_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=["_chr_order", pos_col, y_col]).sort_values(by=["_chr_order", pos_col])

    # If no data after filtering, warn and return
    if d.empty:
        warnings.warn("No GWAS points to plot after filtering/cleaning.")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return {
            "fig": fig,
            "ax_top": ax,
            "ax_tris": {},
            "chrom_order": [],
            "chrom_offsets": {},
            "chrom_lengths": {},
        }

    # Chromosomes in order
    chr_unique = pd.Index(d["_chr"]).drop_duplicates().tolist()

    # Per-chr max positions
    chr_max_pos = d.groupby("_chr", sort=False)[pos_col].max().reindex(chr_unique)

    # Gap between chromosomes
    gap = max(int(0.005 * (d[pos_col].max() or 1)), 1_000_000)

    # Offsets (same logic as your function)
    prev_lengths = chr_max_pos.shift(fill_value=0).cumsum()
    gaps = pd.Series(np.arange(len(chr_max_pos)), index=chr_max_pos.index) * gap
    offsets = (prev_lengths + gaps).astype(np.int64)

    d["_x"] = d[pos_col] + d["_chr"].map(offsets)

    # Tick positions at midpoints
    grp = d.groupby("_chr", sort=False)["_x"]
    tick_pos = {
        chrom: int((grp.min().loc[chrom] + grp.max().loc[chrom]) / 2) for chrom in chr_unique
    }

    # ----------------------------
    # Prepare interactions (within-chromosome)
    # ----------------------------
    E = inter_df[[chr_i_col, chr_j_col, pos_i_col, pos_j_col, val_col]].copy()
    # normalize chroms
    E["_chr_i"] = E[chr_i_col].map(_norm_chr)
    E["_chr_j"] = E[chr_j_col].map(_norm_chr)

    # keep only within-chromosome interactions (i==j)
    E = E[E["_chr_i"] == E["_chr_j"]].copy()
    if only_chr is not None:
        target = _norm_chr(only_chr)
        E = E[E["_chr_i"] == target]

    # numeric-ize
    E[pos_i_col] = pd.to_numeric(E[pos_i_col], errors="coerce")
    E[pos_j_col] = pd.to_numeric(E[pos_j_col], errors="coerce")
    E[val_col] = pd.to_numeric(E[val_col], errors="coerce")
    E = E.dropna(subset=[pos_i_col, pos_j_col, val_col, "_chr_i"])

    # Restrict to chromosomes that appear in GWAS plot (alignment)
    E = E[E["_chr_i"].isin(chr_unique)]
    # optionally clip to known chromosome lengths (robustness)
    # not strictly required; we trust inputs.

    # determine global vmax if needed (consider non-negative scale)
    if vmax is None:
        vmax = max(E[val_col].max() if not E.empty else 0.0, 1e-12)
    vmin = 0.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    # ----------------------------
    # Layout: 2-row GridSpec (Manhattan on top, triangles bottom)
    # Bottom row split into N vertical panels with width proportional to chr length
    # ----------------------------
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.12, figure=fig)

    ax_top = fig.add_subplot(gs[0, 0])

    # Manhattan: color palette alternating by chromosome unless provided
    if colors is None:
        base = plt.cm.tab20.colors
        colors = list(base)
    cmap_chr = {chrom: colors[i % len(colors)] for i, chrom in enumerate(chr_unique)}

    for chrom in chr_unique:
        sel = d["_chr"] == chrom
        ax_top.scatter(
            d.loc[sel, "_x"],
            d.loc[sel, y_col],
            s=point_size,
            alpha=alpha,
            color=cmap_chr[chrom],
            linewidths=0,
        )

    # Axes styling
    ax_top.set_title(f"Global SHAP GWAS + Interactions for {phenotype}")
    ax_top.set_ylabel(y_label)

    # explicit ticks/labels
    def _label(s):
        return "X" if s in ("23", "X") else ("Y" if s in ("24", "Y") else str(s))

    ax_top.set_xticks([tick_pos[c] for c in chr_unique])
    ax_top.set_xticklabels([_label(c) for c in chr_unique])

    # x-lims with padding
    left = d["_x"].min() - gap // 2
    right = d["_x"].max() + gap
    ax_top.set_xlim(left, right)

    ax_top.margins(x=0.0)
    ax_top.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_top.grid(axis="x", visible=False)

    # ----------------------------
    # Bottom row: triangles per chromosome
    # ----------------------------
    # create a nested GridSpec from the bottom cell, with width ratios ~ chromosome lengths + gaps
    bottom_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        len(chr_unique),
        subplot_spec=gs[1, 0],
        wspace=0.02,
        width_ratios=[max(int(chr_max_pos.loc[c]), 1) for c in chr_unique],
    )

    ax_tris: Dict[str, plt.Axes] = {}

    # helper to build bins for a chromosome (positions -> bin indices)
    def _build_bins(chrom, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            return np.array([0.0, 1.0]), np.array([0])  # trivial

        pmin, pmax = float(np.min(positions)), float(np.max(positions))
        if pmax <= pmin:
            # avoid degenerate
            pmax = pmin + 1.0

        if bin_size_bp is not None and bin_size_bp > 0:
            n_bins = int(np.ceil((pmax - pmin) / bin_size_bp))
            n_bins = max(n_bins, 1)
        elif bins_per_chr is not None and bins_per_chr > 0:
            n_bins = min(int(bins_per_chr), max(int(len(np.unique(positions))), 1))
        else:
            # no bin cap, use unique positions (could be large)
            n_bins = max(int(len(np.unique(positions))), 1)

        edges = np.linspace(pmin, pmax, num=n_bins + 1, endpoint=True)
        idx = np.clip(np.digitize(positions, edges) - 1, 0, n_bins - 1)
        return edges, idx

    # pre-split interactions by chromosome
    inter_by_chr = {c: E[E["_chr_i"] == c] for c in chr_unique}

    # for colorbar aggregation
    mappable_for_cbar = None

    for j, chrom in enumerate(chr_unique):
        ax = fig.add_subplot(bottom_gs[0, j])
        ax_tris[chrom] = ax

        Ec = inter_by_chr[chrom]
        if Ec.empty:
            # Draw an empty triangle outline (optional) or just blank
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)
            ax.set_title(f"chr{_label(chrom)}", fontsize=9, pad=2)
            continue

        # Collect position universe for bins
        pos_union = np.concatenate([Ec[pos_i_col].values, Ec[pos_j_col].values])
        edges, _ = _build_bins(chrom, pos_union)
        n = len(edges) - 1

        # Build upper-triangular matrix of values
        M = np.zeros((n, n), dtype=float)  # implicit 0.0 for missing pairs
        # map each interaction to bin indices
        i_idx = np.clip(np.digitize(Ec[pos_i_col].values, edges) - 1, 0, n - 1)
        j_idx = np.clip(np.digitize(Ec[pos_j_col].values, edges) - 1, 0, n - 1)
        vals = Ec[val_col].values

        # fill only upper triangle (i <= j); if i>j, swap
        ii, jj, vv = [], [], []
        for a, b, v in zip(i_idx, j_idx, vals):
            if a <= b:
                ii.append(a)
                jj.append(b)
                vv.append(v)
            else:
                ii.append(b)
                jj.append(a)
                vv.append(v)
        M[np.array(ii), np.array(jj)] = vv

        M_plot = M.astype(float)
        M_plot[np.tril_indices_from(M_plot, k=-1)] = np.nan  # hide lower triangle

        g = np.arange(n + 1, dtype=float)
        I, J = np.meshgrid(g, g, indexing="ij")

        # rotate so the diagonal is horizontal
        X = (I + J) / 2.0
        Y = (J - I) / 2.0

        im = ax.pcolormesh(
            X,
            Y,
            M_plot,
            cmap=cmap,
            norm=norm,
            shading="flat",  # <-- use 'flat' for corner grids
            edgecolors="none",
            rasterized=True,
        )

        if mappable_for_cbar is None:
            mappable_for_cbar = im

        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Optional colorbar (shared)
    if show_colorbar and mappable_for_cbar is not None:
        cax = fig.add_axes([0.92, 0.12, 0.012, 0.22])  # [left, bottom, width, height]
        cb = fig.colorbar(mappable_for_cbar, cax=cax)
        cb.set_label(val_col, rotation=90)

    # Shared x-label below both rows
    if triangle_title:
        fig.suptitle(triangle_title, y=0.995, fontsize=12)

    ax_top.set_xlabel("Chromosome")
    fig.subplots_adjust(bottom=bottom_pad)

    return {
        "fig": fig,
        "ax_top": ax_top,
        "ax_tris": ax_tris,
        "chrom_order": chr_unique,
        "chrom_offsets": offsets.to_dict(),
        "chrom_lengths": chr_max_pos.fillna(0).to_dict(),
    }


def main(
    data_dir: Path | str,
    task_ids: List[int] | int = list(range(1, 21)),
    shap_col: str = "shap_values",
    chr_col: str = "chr_name",
    pos_col: str = "chr_position",
    snp_col: str = "snp",
    log_scale: bool = False,
    out_dir_path: str | None = None,
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

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError("Provided data dir does not exist.")

    for task_id in task_ids:
        phenotype_name, phenotype_id = utils.setup(task_id)
        logger.info(
            f"Plotting global shap values for phenotype {phenotype_name} ({phenotype_id})..."
        )

        annotation_path = data_dir / f"{phenotype_name}_annotation_df.parquet"
        interactions_path = data_dir / f"{phenotype_name}_interactions.parquet"

        if not annotation_path.is_file():
            logger.warning(f"Could not find annotation file for phenotype {phenotype_name}")
            continue
        elif not interactions_path.is_file():
            logger.warning(f"Could not find interactions file for phenotype {phenotype_name}")
            continue
        else:
            # Load inputs (Parquet as per your setup)
            annotation_df = pd.read_parquet(annotation_path)
            interaction_df = pd.read_parquet(interactions_path)

        result = plot_gwas_with_interactions(annotation_df, interaction_df, phenotype_name)

        out_dir = data_dir / "plots" / "manhattan_with_ld"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_plot_path = out_dir / f"{phenotype_name}.png"
        logger.info(f"Saving plot for phenotype {phenotype_name} to {out_plot_path}")
        result["fig"].savefig(out_plot_path, bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    Fire(main)
