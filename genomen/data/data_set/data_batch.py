from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import DataSetConfig


@dataclass
class DataBatch:
    cfg: DataSetConfig
    X: npt.ArrayLike
    geno_annotation: pd.DataFrame
    y: npt.ArrayLike | None = None
    pheno_annotation: pd.DataFrame | None = None
    residuals: npt.ArrayLike | None = None
    scaler: StandardScaler | None = None
    sample_weights: npt.ArrayLike | None = None

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
