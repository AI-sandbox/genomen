import logging
import pickle
from typing import Tuple

import bed_reader as br
import numpy as np
import numpy.typing as npt
import pandas as pd


class GenoSet:
    def __init__(
        self,
        bed_path: str | bytes,
        annotation_df: pd.DataFrame,
        n_samples: int,
        n_total_samples: int,
    ):
        self._bed_path = bed_path.decode() if isinstance(bed_path, bytes) else bed_path
        self._bed = br.open_bed(
            self._bed_path,
            iid_count=n_total_samples,
            skip_format_check=True,
        )
        self.n_total_samples = n_total_samples
        self.annotation_df = annotation_df
        self.n_samples = n_samples
        self._preloaded: np.ndarray | None = None
        self._preloaded_sample_idxs: np.ndarray | None = None
        self._preloaded_variant_idxs: np.ndarray | None = None
        self._logger = logging.getLogger(self.__class__.__name__)

    def fork(self, annotation_df: pd.DataFrame, n_samples: int) -> "GenoSet":
        """Create a new GenoSet sharing the same BED file with a different sample count/variant subset."""
        return GenoSet(self._bed_path, annotation_df, n_samples, self.n_total_samples)

    def preload(
        self,
        sample_idxs: npt.ArrayLike,
        chunk_size: int = 10_000,
    ) -> None:
        """Preload all variants for the given samples into memory (int8).

        Eliminates per-batch disk reads. Suitable for fixed sample sets
        like val/test. If sample_idxs is None, uses all samples in the dataset.
        Memory cost: n_samples × n_variants bytes.
        """
        sample_idxs = np.sort(np.asarray(sample_idxs, dtype=np.int64))
        variant_idxs = self.variant_idxs
        n_samples = len(sample_idxs)
        n_variants = len(variant_idxs)
        mem_gb = n_samples * n_variants / 1e9
        self._logger.info(
            f"Preloading {n_samples} samples × {n_variants} variants into memory "
            f"(~{mem_gb:.1f} GB)..."
        )

        buf = np.empty((n_samples, n_variants), dtype=np.int8)
        for start in range(0, n_variants, chunk_size):
            end = min(start + chunk_size, n_variants)
            chunk_vidxs = variant_idxs[start:end].astype(np.int64)
            chunk = self._bed.read(
                index=np.s_[sample_idxs, chunk_vidxs],
                dtype="int8",
                order="C",
            )
            buf[:, start:end] = chunk

        self._preloaded = buf
        self._preloaded_sample_idxs = sample_idxs.astype(np.uint32)
        self._preloaded_variant_idxs = np.asarray(variant_idxs, dtype=np.uint32)
        self._logger.info("Preload complete.")

    @property
    def variant_idxs(self) -> npt.NDArray:
        """Get variant indices from annotation_df as numpy array."""
        return self.annotation_df.index.values

    def save(self, cache_path: str) -> None:
        data = {
            "annotation_df": self.annotation_df,
            "n_samples": self.n_samples,
            "n_total_samples": self.n_total_samples,
            "bed_path": self._bed_path,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_file(cls, cache_path: str, bed_path: str | None = None) -> "GenoSet":
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        resolved_path = bed_path or data.get("bed_path")
        if not resolved_path:
            raise ValueError("Cannot recreate bed reader without bed_path")
        if isinstance(resolved_path, bytes):
            resolved_path = resolved_path.decode()

        # Support old pgenlib cache format (stored raw_sample_ct)
        n_total_samples = data.get("n_total_samples") or data.get("raw_sample_ct")
        if n_total_samples is None:
            raise ValueError("Cannot recreate bed reader: total sample count not found in cache")

        annotation_df = data["annotation_df"].copy()
        annotation_df.index = annotation_df.index.astype(np.uint32)

        return cls(resolved_path, annotation_df, data["n_samples"], int(n_total_samples))

    @classmethod
    def from_plink(
        cls,
        df: pd.DataFrame,
        bed_path: str | bytes,
        n_total_samples: int,
        n_samples: int,
    ) -> "GenoSet":
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

        return cls(bed_path, df, n_samples, n_total_samples)

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

        # fast path: use in-memory preloaded data
        if self._preloaded is not None:
            s_idxs = np.asarray(sample_idxs, dtype=np.uint32)
            if np.all(np.isin(s_idxs, self._preloaded_sample_idxs)):
                v_idxs = np.asarray(variant_idxs, dtype=np.uint32)
                local_s = np.searchsorted(self._preloaded_sample_idxs, s_idxs)
                local_v = np.searchsorted(self._preloaded_variant_idxs, v_idxs)
                X = self._preloaded[np.ix_(local_s, local_v)]
                return np.ascontiguousarray(X), self.annotation_df.loc[v_idxs].copy()

        s_arr = np.asarray(sample_idxs)
        has_duplicates = len(s_arr) != len(set(s_arr.tolist()))

        if has_duplicates:
            unique_s, inverse = np.unique(s_arr, return_inverse=True)
            X = self._bed.read(
                index=np.s_[unique_s.astype(np.int64), np.asarray(variant_idxs, dtype=np.int64)],
                dtype="int8",
                order="C",
            )
            X = X[inverse]
        else:
            X = self._bed.read(
                index=np.s_[s_arr.astype(np.int64), np.asarray(variant_idxs, dtype=np.int64)],
                dtype="int8",
                order="C",
            )

        return np.ascontiguousarray(X), self.annotation_df.loc[variant_idxs].copy()
