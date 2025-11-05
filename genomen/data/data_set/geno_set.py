import pickle
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd


class GenoSet:
    def __init__(
        self, pgen_reader: Callable, annotation_df: pd.DataFrame, n_samples: int
    ):
        self.pgen_reader = pgen_reader
        self.annotation_df = annotation_df
        self.n_samples = n_samples

    @property
    def variant_idxs(self) -> npt.NDArray:
        """Get sample indices from annotation_df as numpy array."""
        return self.annotation_df.index.values

    def save(self, cache_path: str) -> None:
        data = {
            "annotation_df": self.annotation_df,
            "n_samples": self.n_samples,
            "bed_path": self.pgen_reader.args[0],
            "raw_sample_ct": self.pgen_reader.keywords.get("raw_sample_ct"),
        }

        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_file(cls, cache_path: str) -> "GenoSet":
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        from functools import partial

        import pgenlib as pg

        if "bed_path" in data and data["bed_path"]:
            pgen_reader = partial(
                pg.PgenReader, data["bed_path"], raw_sample_ct=data["raw_sample_ct"]
            )
        else:
            raise ValueError("Cannot recreate pgen_reader without bed_path")

        annotation_df = data["annotation_df"].copy()
        annotation_df.index = annotation_df.index.astype(np.uint32)

        return cls(pgen_reader, annotation_df, data["n_samples"])

    @classmethod
    def from_plink(
        cls, df: pd.DataFrame, pgen_reader: Callable, n_samples: int
    ) -> "GenoSet":
        # build genotype annotation df
        df.rename(
            columns={
                "chrom": "chr_name",
                "pos": "chr_position",
                "a0": "other_allele",
                "a1": "effect_allele",
            },
            inplace=True,
        )
        df["rsID"] = df.apply(
            lambda x: f"chr{x['chr_name']}:{x['chr_position']}:{x['other_allele']}:{x['effect_allele']}",
            axis=1,
        )
        df.index = df.index.astype(np.uint32)

        return cls(pgen_reader, df, n_samples)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.n_samples

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the dataset."""
        return len(self), len(self.annotation_df)

    def __getitem__(
        self, key: Tuple[int | npt.ArrayLike, int | npt.ArrayLike]
    ) -> Tuple[npt.ArrayLike, pd.DataFrame]:
        sample_idxs, variant_idxs = key

        sample_idxs_array = np.asarray(sample_idxs)
        has_duplicates = (
            len(sample_idxs_array) != len(set(sample_idxs_array))
        )  # not expected, hence check (O(m)) + eventually run unique ((O(m log m))) > run unique ((O(m log m))) every time
        if has_duplicates:
            unique_sample_idxs, inverse_indices = np.unique(
                sample_idxs_array, return_inverse=True
            )
            unique_X = np.zeros(
                (len(unique_sample_idxs), len(variant_idxs)), dtype=np.int8
            )

            reader = self.pgen_reader(sample_subset=unique_sample_idxs)
            reader.read_list(variant_idxs, unique_X, sample_maj=-1)

            X = unique_X[inverse_indices]
        else:
            X = np.zeros((len(sample_idxs), len(variant_idxs)), dtype=np.int8)

            reader = self.pgen_reader(sample_subset=sample_idxs)
            reader.read_list(variant_idxs, X, sample_maj=-1)

        X = np.ascontiguousarray(X)

        # impute missing value to 0
        X[X == -9] = 0

        annotation_df = self.annotation_df.loc[variant_idxs].copy()

        return X, annotation_df
