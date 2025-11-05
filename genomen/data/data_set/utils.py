import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer

from ..sources import plink_utils

if TYPE_CHECKING:
    from .config import GWASConfig

logger = logging.getLogger(__name__)


def get_data_paths(file_format: str) -> Dict[str, str]:
    """Get paths to data files based on the specified file format.

    Args:
        file_format: The format of the data files (e.g., "plink")

    Returns:
        Dictionary of file paths

    Raises:
        ValueError: If the file format is not supported
    """
    from dotenv import load_dotenv

    load_dotenv()

    if file_format == "plink":
        return plink_utils.get_plink_paths()
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def check_for_duplicates(array: np.ndarray, array_id: str):
    unique_samples, counts = np.unique(array, return_counts=True)
    if any(counts > 1):
        duplicates = unique_samples[counts > 1]
        logger.warning(
            f"{len(duplicates)} duplicate {array_id} found in the dataset. "
            f"Duplicate {array_id}: {duplicates}"
        )


def adaptive_sampling(
    sample_idxs: npt.NDArray[np.uint32],
    phenotypes: npt.NDArray,
    classification: bool,
    size: int | None = None,
    k: int | None = None,
    strategy: Literal["stratified", "balanced"] = "stratified",
    rng: np.random.Generator = None,
    n_bins: int = 10
) -> npt.NDArray[np.uint32]:
    """Perform adaptive sampling based on binary phenotypes.

    Args:
        sample_idxs: Array of sample indices
        phenotypes: Array of phenotypes corresponding to sample_idxs
        size: Number of samples to return (used for stratified sampling)
        strategy: Sampling strategy - "stratified" for 50:50 ratio, "balanced" for k:1 ratio
        k: Ratio of negative samples to positive samples (used for balanced sampling)
        rng: Random number generator

    Returns:
        Array of sampled indices
    """
    if rng is None:
        rng = np.random.default_rng()

    if classification:
        # Identify cases and controls
        case_mask = phenotypes == 1.0
        control_mask = phenotypes == 0.0

        case_idxs = sample_idxs[case_mask]
        control_idxs = sample_idxs[control_mask]

        if strategy == "stratified":
            # 50:50 stratified sampling
            n_case = size // 2
            n_control = size - n_case

            # Sample with replacement if not enough samples available
            case_samples = rng.choice(
                case_idxs, size=n_case, replace=n_case > len(case_idxs)
            )
            control_samples = rng.choice(
                control_idxs, size=n_control, replace=n_control > len(control_idxs)
            )

            return np.concatenate([case_samples, control_samples])

        elif strategy == "balanced":
            # k:1 balanced sampling (k controls per case)
            if size is not None:
                n_case = min(len(case_idxs), size // 2)
            else: 
                n_case = len(case_idxs)
            n_control_desired = min(len(control_idxs), k * n_case)

            # Use all cases and sample controls
            if n_control_desired > 0 and len(control_idxs) > 0:
                control_samples = rng.choice(
                    control_idxs, size=n_control_desired, replace=False
                )
                return np.concatenate([case_idxs, control_samples])
            else:
                return case_idxs
        else:
            raise ValueError(f"Unknown sampling strategy {strategy} for classification phenotype")
    else: # continuous phenotype
        if np.allclose(phenotypes, phenotypes[0]): # if constant, fallback to random sampling
            return rng.choice(sample_idxs, size=size, replace=len(sample_idxs) < size).astype(np.uint32)

        n_bins = max(2, min(n_bins, len(sample_idxs)))
        kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        y_bins = kbd.fit_transform(phenotypes.reshape(-1, 1)).astype(int).ravel()

        test_size = min(1.0, max(0.0, size / len(sample_idxs)))
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=int(rng.integers(0, 2**32 - 1)),
        )
        _, take = next(sss.split(np.zeros_like(y_bins), y_bins))
        sampled = sample_idxs[take]

        if len(sampled) < size:
            need = size - len(sampled)
            sampled = np.concatenate(
                [sampled, rng.choice(sample_idxs, size=need, replace=len(sample_idxs) < need)]
            )
        elif len(sampled) > size:
            sampled = rng.choice(sampled, size=size, replace=False)

        return sampled


def get_gwas_priors(snps: np.ndarray[str], cfg: "GWASConfig") -> npt.NDArray[np.uint32]:
    try:
        df_gwas = pd.read_csv(cfg.path, sep=cfg.sep, low_memory=False)
    except Exception as e:
        raise ValueError(
            f"variant_sampling=GWAS requires setting the path to a valid GWAS study. Error: {e}"
        )

    assert (cfg.pvalue_column is not None) or (cfg.nlogpvalue_column is not None), "'pvalue_column' or 'nlogpvalue_column' has to specified in GWASConfig"

    value_column = cfg.nlogpvalue_column if cfg.nlogpvalue_column is not None else cfg.pvalue_column
    cols_type = {cfg.snps_column: str, value_column: float}
    cols = list(cols_type.keys())

    if any(col not in df_gwas.columns for col in cols):
        raise ValueError(
            "Invalid GWAS study, specify the correct SNP and P-Value columns"
        )

    df_gwas = df_gwas[cols].dropna()
    for col, type_ in cols_type.items():
        df_gwas[col] = df_gwas[col].astype(type_)

    if cfg.pvalue_aggregation in ["mean", "min"]:
        df_gwas = df_gwas.groupby(cfg.snps_column).aggregate(
            {value_column: cfg.pvalue_aggregation}
        )
    else:
        df_gwas = df_gwas.groupby(cfg.snps_column)[value_column].apply(
            lambda x: scipy.stats.combine_pvalues(x, method=cfg.pvalue_aggregation)[1]
        )

    df = pd.DataFrame({cfg.snps_column: snps})
    print(f"Got {len(df_gwas)} values from GWAS")
    df = pd.merge(
        df, df_gwas, left_on=cfg.snps_column, right_on=cfg.snps_column, how="left"
    )
    if cfg.nlogpvalue_column is not None:
        n_overlap = df[value_column].notna().sum()
        logger.info(f"Found p-values for {n_overlap} variants in GWAS")
        df["weight"] = df[value_column].fillna(cfg.impute_val)
    else:
        n_overlap = df[value_column].notna().sum()
        logger.info(f"Found p-values for {n_overlap} variants in GWAS")
        df = df.fillna(np.exp(-1))
        df["weight"] = -np.log(df[value_column]).fillna(cfg.impute_val)

    return np.nan_to_num(df["weight"].values, nan=cfg.impute_val, posinf=33333333, neginf=cfg.impute_val)


def hash_ndarray(arr: np.ndarray, *, algo="blake2s", chunk_bytes=1 << 20) -> str:
    """
    Stable, fast content hash for a NumPy array.
    Includes dtype and shape. Streams in chunks to avoid copies.
    """
    h = hashlib.new(algo)
    a = np.ascontiguousarray(arr)
    meta = f"{a.dtype.str}|{a.shape}".encode()
    h.update(meta)
    mv = memoryview(a).cast("B")
    for i in range(0, mv.nbytes, chunk_bytes):
        h.update(mv[i : i + chunk_bytes])
    return h.hexdigest()
