"""LD block computation for the phenotype simulator.

Core training/inference only uses random variant sampling, so LD-block
computation was removed from DataSet. The simulator still needs LD blocks
(to place causal interaction pairs in different/same LD blocks), so this
tool reimplements it as a standalone function operating on a DataSet.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .. import DataSet
from ..data_set import utils as data_set_utils
from ..sources import plink_utils


def compute_ld_blocks(
    data_set: DataSet,
    prune_kb: int = 250,
    prune_step: int = 50,
    prune_r2: float = 0.1,
    tau: float = 0.1,
    ld_window_kb: int = 1000,
    ld_window: int = 50000,
    ram_mb: int = 64_000,
) -> None:
    """Compute LD blocks and merge `block_idx` into `data_set.genotype.annotation_df`.

    Loads from cache if available. Mirrors the LD-block portion of the
    pre-simplification `DataSet._compute_ld()`.
    """
    if data_set._cache_path is None:
        data_set._setup_cache_path()

    sid = data_set_utils.hash_ndarray(data_set.phenotype.sample_idxs)
    vid = data_set_utils.hash_ndarray(data_set.genotype.variant_idxs)
    block_cache = (
        Path(data_set._cache_path)
        / f"blocks_{sid}_{vid}_{prune_kb}_{prune_step}_{prune_r2}_{tau}_{ld_window_kb}_{ld_window}.parquet"
    )

    if "block_idx" in data_set.genotype.annotation_df.columns:
        return

    loaded_from_cache = False
    if block_cache.exists():
        data_set._logger.info("Loading cached LD blocks...")
        cached = pd.read_parquet(block_cache)
        if (
            isinstance(cached, pd.DataFrame)
            and ("block_idx" in cached)
            and len(cached) == len(data_set.genotype.annotation_df)
            and np.array_equal(cached.index.values, data_set.genotype.annotation_df.index.values)
        ):
            data_set.genotype.annotation_df["block_idx"] = cached["block_idx"].values
            loaded_from_cache = True
        else:
            data_set._logger.warning("Cached LD blocks invalid for current genotype; recomputing.")

    if not loaded_from_cache:
        data_set._logger.info("Computing LD blocks...")
        ld_df = plink_utils.compute_ld(
            bed_path=data_set.cfg.paths["bed_path"],
            bim_path=data_set.cfg.paths["bim_path"],
            fam_path=data_set.cfg.paths["fam_path"],
            snp_ids=data_set.genotype.annotation_df["snp"].values,
            prune_kb=prune_kb,
            prune_step=prune_step,
            prune_r2=prune_r2,
            tau=tau,
            ld_window_kb=ld_window_kb,
            ld_window=ld_window,
            include_x=data_set.cfg.include_x_chromosome,
            ram_mb=ram_mb,
        )
        data_set.genotype.annotation_df = data_set.genotype.annotation_df.merge(
            ld_df, on="snp", how="left"
        ).set_index(data_set.genotype.annotation_df.index)
        data_set.genotype.annotation_df["block_idx"] -= 1  # null idx
        data_set.genotype.annotation_df["block_idx"] = (
            data_set.genotype.annotation_df["block_idx"].fillna(-1)
        )
        try:
            pd.DataFrame(
                {"block_idx": data_set.genotype.annotation_df["block_idx"].values},
                index=data_set.genotype.annotation_df.index,
            ).sort_index().to_parquet(block_cache, index=True)
        except Exception as e:
            data_set._logger.warning(f"Could not write block cache {block_cache}: {e}")

    # log LD block statistics
    assigned_mask = data_set.genotype.annotation_df["block_idx"] != -1
    n_blocks = data_set.genotype.annotation_df["block_idx"].max() + 1
    num_not_assigned = (~assigned_mask).sum()
    _, ld_block_len = np.unique(
        data_set.genotype.annotation_df["block_idx"].loc[assigned_mask].values,
        return_counts=True,
    )
    data_set._logger.info(
        f"Mapped SNPs to {int(n_blocks)} LD blocks with an average length of "
        f"{ld_block_len.mean():.2f} (max: {max(ld_block_len)}, min: {min(ld_block_len)}). "
        f"{num_not_assigned} SNPs have not been assigned to any LD block."
    )
