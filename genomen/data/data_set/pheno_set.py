import pickle
from typing import Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from .config import CovarConfig, DataSetConfig


class PhenoSet:
    def __init__(
        self,
        annotation_df: pd.DataFrame,
        covar_cfg: CovarConfig,
        residuals: npt.ArrayLike | None = None,
    ):
        self.annotation_df = annotation_df.sort_index()
        self._is_duplicate = self.annotation_df.index.duplicated(keep="first")
        self.covar_cfg = covar_cfg

        self.case_control_ratio = self.y.mean()

        if residuals:
            self.annotation_df["residuals"] = residuals

    @property
    def covar_keys(self) -> list[str]:
        """Get covar keys from the config."""
        return self.covar_cfg.covar_keys if self.covar_cfg.include_covars else []

    @property
    def y(self) -> npt.NDArray:
        """Get phenotype values as numpy array."""
        return self.annotation_df["y"].values

    @property
    def residuals(self) -> npt.NDArray:
        """Get phenotype values as numpy array."""
        if "residuals" not in self.annotation_df:
            return None
        else:
            return self.annotation_df["residuals"].values

    @residuals.setter
    def residuals(self, value: npt.ArrayLike) -> None:
        """Set residuals values in the annotation DataFrame."""
        self.annotation_df["residuals"] = value

    @property
    def sample_idxs(self) -> npt.NDArray:
        """Get sample indices from annotation_df as numpy array."""
        return self.annotation_df.index.values

    def save(self, cache_path: str) -> None:
        data = {
            "annotation_df": self.annotation_df,
            "covar_cfg": self.covar_cfg.__dict__,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_file(cls, cache_path: str) -> "PhenoSet":
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        annotation_df = data["annotation_df"].copy()
        annotation_df.index = annotation_df.index.astype(np.uint32)

        covar_cfg = CovarConfig(**data["covar_cfg"])
        return cls(annotation_df, covar_cfg)

    @classmethod
    def from_plink(cls, cfg: DataSetConfig, df: pd.DataFrame) -> "PhenoSet":
        covar_keys = (
            cfg.covar_config.covar_keys if cfg.covar_config.include_covars else []
        )

        # build phenotype annotation df
        annotation_df = df[["fam_idx", "fid", "iid", "population"] + covar_keys + [cfg.phenotype_id]].copy()  # sample_idx population 
        annotation_df.rename(
            columns={cfg.phenotype_id: "y", "fam_idx": "sample_idx"}, inplace=True
        )
        annotation_df.set_index("sample_idx", drop=True, inplace=True)
        annotation_df.index = annotation_df.index.astype(np.uint32)

        return cls(annotation_df, cfg.covar_config, residuals=None)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.annotation_df)

    def __getitem__(
        self, sample_idxs: int | npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, pd.DataFrame, npt.NDArray | None]:
        unique_annotation_df = self.annotation_df[~self._is_duplicate]
        annotation_df = unique_annotation_df.loc[sample_idxs].copy()
        y = annotation_df["y"].values
        residuals = (
            annotation_df["residuals"].values if self.residuals is not None else None
        )

        return y, annotation_df, residuals

    def get_covars(self) -> npt.NDArray:
        if not self.covar_keys:
            raise ValueError("Phenotype not loaded with covariates!")
        return self.annotation_df[self.covar_keys].to_numpy()
