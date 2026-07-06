import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import DataSetConfig

_logger = logging.getLogger(__name__)


@dataclass
class DataBatch:
    cfg: DataSetConfig
    X: npt.ArrayLike
    geno_annotation: pd.DataFrame
    y: npt.ArrayLike | None = None
    pheno_annotation: pd.DataFrame | None = None
    residuals: npt.ArrayLike | None = None
    covar_pred: npt.ArrayLike | None = None
    scaler: StandardScaler | None = None
    sample_weights: npt.ArrayLike | None = None
    type: str = "geno"  # "geno" or "covar"

    def __post_init__(self):
        if self.type == "geno":
            maf = self.geno_annotation["MAF"].values.astype(np.float32)
            # A0_FREQ = P(a0) = P(allele bed_reader counts). When a0 is the major allele,
            # P(a0) = 1 - MAF, so using MAF for the mean would be wrong.
            if "A0_FREQ" in self.geno_annotation.columns:
                a0_freq = self.geno_annotation["A0_FREQ"].values.astype(np.float32)
            else:
                _logger.warning(
                    "A0_FREQ missing from annotation; centering with MAF (may be biased)"
                )
                a0_freq = maf
            mean = 2.0 * a0_freq
            std = np.sqrt(2.0 * maf * (1.0 - maf))  # symmetric: MAF*(1-MAF) = a0_freq*(1-a0_freq)
            std[std == 0.0] = 1.0  # monomorphic variants: no-op

            # Detect missing sentinel (-127 from bed_reader) before centering
            if np.issubdtype(self.X.dtype, np.integer):
                missing_mask = self.X == -127
                n_missing = int(missing_mask.sum())
            else:
                missing_mask = None
                n_missing = 0

            X_float = (
                self.X.astype(np.float32) if np.issubdtype(self.X.dtype, np.integer) else self.X
            )
            self.X = (X_float - mean) / std
            if missing_mask is not None and n_missing > 0:
                # Mean imputation: 0.0 in centered space = 2·MAF in raw space
                self.X[missing_mask] = 0.0

            self.type = "prescaled"
        elif self.type == "covar":
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            self.type = "prescaled"
        # type == "prescaled": X already standardized, no-op

    def get_labels(self, use_resids: bool = False) -> npt.ArrayLike:
        """Get labels for training - either original y or residuals."""
        if use_resids:
            if self.residuals is None:
                raise ValueError("No residuals set!")
            return self.residuals
        return self.y

    def get_sample_weights(self) -> npt.ArrayLike | None:
        """Get sample weights based on case/control ratio from original binary labels."""
        unique_labels, _ = np.unique(self.y, return_counts=True)

        if len(unique_labels) == 2:  # Binary case
            weights = compute_sample_weight(class_weight="balanced", y=self.y)

            return weights
        else:
            return None
