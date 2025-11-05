#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire

from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_gwas_from_df(
    df: pd.DataFrame,
    phenotype: str,
    *,
    chr_col: str = "chr_name",
    pos_col: str = "chr_position",
    y_col: str = "shap_values",
    y_label: str = "|SHAP|",
    figsize: Tuple[float, float] = (20, 5),
    point_size: float = 200,
    alpha: float = 0.8,
    colors: Optional[Sequence[str]] = None,
    only_chr: Optional[Union[str, int]] = None,
    abs_only: bool = False,
    y_log_scale: bool = False,
    log_base: float = 10.0,
    symlog_linthresh: float = 1e-3,
    annotate: bool = False,
    top_k: int = 10,
    snp_col: Optional[str] = None,
    peak_window: int = 50_000,
    show_title: bool = True,
    label_size: Optional[float] = None,
    show_every: int = 1,
):
    """Plot a Manhattan-style scatter of per-variant SHAP values, optionally annotated."""
    d = df[
        [chr_col, pos_col, y_col] + ([snp_col] if snp_col and snp_col in df.columns else [])
    ].copy()
    title = f"Manhattan plot of SHAP values for phenotype {phenotype}"

    # --- Normalize chromosomes
    def _norm_chr(x):
        s = str(x).strip()
        s = re.sub(r"^chr", "", s, flags=re.IGNORECASE).upper()
        if s == "23":
            s = "X"
        elif s == "24":
            s = "Y"
        return s

    d["_chr"] = d[chr_col].map(_norm_chr)
    if only_chr is not None:
        d = d[d["_chr"] == _norm_chr(only_chr)]

    # --- Sort chromosomes
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

    d["_chr_order"] = d["_chr"].map(_order_key)
    d[pos_col] = pd.to_numeric(d[pos_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if abs_only:
        d[y_col] = d[y_col].abs()
    d = d.dropna(subset=["_chr_order", pos_col, y_col]).sort_values(by=["_chr_order", pos_col])

    def _apply_y_scale(ax, y_series):
        if not y_log_scale:
            return
        if (y_series <= 0).any():
            ax.set_yscale("symlog", linthresh=symlog_linthresh, base=log_base)
        else:
            ax.set_yscale("log", base=log_base)

    # --- Compute genome-wide offsets
    chr_unique = pd.Index(d["_chr"]).drop_duplicates().tolist()
    chr_max_pos = d.groupby("_chr", sort=False)[pos_col].max().reindex(chr_unique)
    gap = max(int(0.005 * (d[pos_col].max() or 1)), 1_000_000)
    prev_lengths = chr_max_pos.shift(fill_value=0).cumsum()
    gaps = pd.Series(np.arange(len(chr_max_pos)), index=chr_max_pos.index) * gap
    offsets = (prev_lengths + gaps).astype(np.int64)
    d["_x"] = d[pos_col] + d["_chr"].map(offsets)

    # --- Plot base Manhattan points
    grp = d.groupby("_chr", sort=False)["_x"]
    tick_pos = {
        chrom: int((grp.min().loc[chrom] + grp.max().loc[chrom]) / 2) for chrom in chr_unique
    }
    if colors is None:
        colors = list(plt.cm.tab20.colors)
    cmap = {chrom: colors[i % len(colors)] for i, chrom in enumerate(chr_unique)}

    fig, ax = plt.subplots(figsize=figsize)
    for chrom in chr_unique:
        sel = d["_chr"] == chrom
        ax.scatter(
            d.loc[sel, "_x"],
            d.loc[sel, y_col],
            s=point_size,
            alpha=alpha,
            color=cmap[chrom],
            linewidths=0,
            rasterized=True,
        )

    # --- Labels and axes
    if show_title:
        ax.set_title(title)
    ax.set_xlabel("Chromosome")
    ax.set_ylabel(y_label)
    if label_size is not None:
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.tick_params(axis="both", labelsize=label_size)

    _apply_y_scale(ax, d[y_col])

    # --- Chromosome labels (show every N)
    def _label_chr(s):
        return "X" if s in ("23", "X") else ("Y" if s in ("24", "Y") else str(s))

    labels_all = [_label_chr(c) for c in chr_unique]
    ticks_all = [tick_pos[c] for c in chr_unique]
    major_idx = list(range(0, len(chr_unique), max(1, show_every)))
    ax.set_xticks([ticks_all[i] for i in major_idx])
    ax.set_xticklabels([labels_all[i] for i in major_idx])

    ax.set_xlim(d["_x"].min() - gap // 2, d["_x"].max() + gap)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    # --- Annotations (if requested)
    if annotate and len(d) > 0:
        vals = d[y_col].abs()
        top = d.loc[vals.nlargest(min(top_k, len(d))).index].copy()
        top = top.sort_values(by=["_chr_order", pos_col, y_col], ascending=[True, True, False])

        groups: Dict[int, List[Dict]] = {}

        def assign_group(chr_order: int, pos_bp: float):
            if chr_order not in groups:
                groups[chr_order] = []
            for g in groups[chr_order]:
                if abs(pos_bp - g["center"]) <= peak_window:
                    g["count"] += 1
                    g["center"] = (g["center"] * (g["count"] - 1) + pos_bp) / g["count"]
                    return g
            g = {"center": pos_bp, "count": 1, "labels": 0}
            groups[chr_order].append(g)
            return g

        x_offsets = np.array([8e6, -8e6, 0.0, 12e6, -12e6])
        y_offsets = np.array([0.15, 0.2, -0.18, -0.25, 0.22])
        y_range = float(d[y_col].max() - d[y_col].min()) or 1.0

        idx = 0
        for _, r in top.iterrows():
            grp_rec = assign_group(int(r["_chr_order"]), float(r[pos_col]))
            if grp_rec["labels"] >= 3:
                continue
            label = (
                str(r[snp_col])
                if snp_col and snp_col in r and pd.notna(r[snp_col])
                else f'{r["_chr"]}:{int(r[pos_col])}'
            )
            x = float(r["_x"])
            y = float(r[y_col])
            xo = float(x_offsets[idx % len(x_offsets)])
            yo = float(y_offsets[idx % len(y_offsets)] * y_range)
            idx += 1
            ax.annotate(
                label,
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
            grp_rec["labels"] += 1

    return fig, ax


def main(
    data_dir: Path | str,
    task_ids: List[int] | int = list(range(1, 22)),
    abs_values: bool = False,
    log_scale: bool = False,
    annotate_top: bool = False,
    shap_col: str = "shap_values",
    chr_col: str = "chr_name",
    pos_col: str = "chr_position",
    snp_col: str = "snp",
    out_dir_path: Optional[str] = None,
    top_k: int = 10,
    peak_window: int = 1_000_000,
    point_size: float = 6.0,
    alpha: float = 0.8,
    width: float = 12.0,
    height: float = 5.0,
    no_title: bool = False,
    label_size: Optional[float] = None,
    show_every: int = 1,
    only_chr: Optional[Union[str, int]] = None,
):
    """Generate Manhattan plots for one or more task IDs (saves PNG + PDF)."""
    if isinstance(task_ids, int):
        task_ids = [task_ids]

    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError("Provided data path does not exist.")

    for task_id in task_ids:
        phenotype_name, phenotype_id = utils.setup(task_id)
        logger.info(f"Plotting SHAP values for phenotype {phenotype_name} ({phenotype_id})...")

        annotation_df_path = data_dir / f"{phenotype_name}_annotation_df.parquet"
        if not annotation_df_path.is_file():
            logger.warning(f"Missing annotation file for {phenotype_name}")
            continue
        annotation_df = pd.read_parquet(annotation_df_path)

        fig, ax = plot_gwas_from_df(
            annotation_df,
            phenotype_name,
            chr_col=chr_col,
            pos_col=pos_col,
            y_col=shap_col,
            snp_col=snp_col,
            abs_only=abs_values,
            y_log_scale=log_scale,
            annotate=annotate_top,
            top_k=top_k,
            peak_window=peak_window,
            figsize=(width, height),
            point_size=point_size,
            alpha=alpha,
            show_title=not no_title,
            label_size=label_size,
            show_every=show_every,
            only_chr=only_chr,
        )

        out_dir = Path(out_dir_path) if out_dir_path else (data_dir / "plots" / "manhattan")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compose filename suffix for clarity
        suffix_parts = []
        if abs_values:
            suffix_parts.append("abs")
        if log_scale:
            suffix_parts.append("log")
        if annotate_top:
            suffix_parts.append(f"ann{top_k}")
        if only_chr is not None:
            suffix_parts.append(f"chr{only_chr}")
        if show_every > 1:
            suffix_parts.append(f"xevery{show_every}")
        if label_size is not None:
            suffix_parts.append(f"lab{int(label_size)}")
        if no_title:
            suffix_parts.append("notitle")
        suffix_str = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

        out_base = out_dir / f"{phenotype_name}{suffix_str}"
        png_path = out_base.with_suffix(".png")
        pdf_path = out_base.with_suffix(".pdf")

        logger.info(f"Saving plot as {png_path.name} and {pdf_path.name}")
        fig.savefig(png_path, bbox_inches="tight", dpi=200)
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    Fire(main)
