"""Class to hold genotype and phenotype data and retrieve batches."""

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from ... import utils
from ..sources import plink_utils
from . import utils as data_set_utils
from .config import DataSetConfig
from .data_batch import DataBatch
from .geno_set import GenoSet
from .pheno_set import PhenoSet


class DataSet:
    def __init__(
        self,
        cfg: DataSetConfig | None = None,
        genotype: GenoSet | None = None,
        phenotype: PhenoSet | None = None,
    ):
        self.cfg: DataSetConfig = cfg or utils.init_class(cls=DataSetConfig)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._cache_path = None

        self.genotype = genotype
        self.phenotype = phenotype

        if (genotype is None) or (phenotype is None):
            self._setup_cache_path()
            self._load_data_set()

        self._logger.debug(
            f"Shape of {'train' if self.cfg.is_train else ''} data set is {self.genotype.shape}"
        )

    def _setup_cache_path(self) -> None:
        """Create a unique cache directory based on input parameters."""
        base_path = os.path.expanduser("~/.genomen")

        # create a unique hash based on input parameters
        covar_str = "_".join(
            self.cfg.covar_config.covar_keys if self.cfg.covar_config.include_covars else []
        )
        uid_parts = [
            str(self.cfg.paths["fam_path"]),
            str(self.cfg.paths["bim_path"]),
            str(self.cfg.paths["master_path"]),
            str(self.cfg.phenotype_id),
            "_".join(self.cfg.populations),
            str(self.cfg.maf_threshold),
            str(self.cfg.include_x_chromosome),
            str(self.cfg.sex if self.cfg.sex is not None else ""),
            covar_str,
        ]

        uid = "".join(uid_parts)
        byte_string = uid.encode()
        hash_object = hashlib.md5(byte_string)
        hash_uid = hash_object.hexdigest()

        self._cache_path = os.path.join(base_path, hash_uid)

        # ensure directories exist
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.exists(self._cache_path):
            os.makedirs(self._cache_path)

    def _load_data_set(self):
        # check if cached data exists
        geno_cache_file = os.path.join(self._cache_path, "data_set.geno")
        pheno_cache_file = os.path.join(self._cache_path, "data_set.pheno")
        data_loaded = False

        self._logger.info("Looking for cached dataset...")
        if os.path.exists(geno_cache_file) and os.path.exists(pheno_cache_file):
            try:
                self._logger.info("Found cached dataset. Proceeding to loading data...")
                self.genotype = GenoSet.from_file(
                    geno_cache_file, bed_path=self.cfg.paths["bed_path"]
                )
                self.phenotype = PhenoSet.from_file(pheno_cache_file)
                data_loaded = True
            except Exception as e:
                self._logger.warning(
                    f"Error loading cached data: {e}. Proceeding to load data from original files..."
                )

        if not data_loaded:
            self._logger.info(
                "Did not find any cached data. Proceeding to load data from original files"
            )

            if self.cfg.file_format == "plink":
                self._load_plink_data()
            else:
                raise ValueError(f"Unsupported file format: {self.cfg.file_format}")

            # Save data to cache
            self._logger.info("Saving data to cache for future use")
            self.genotype.save(geno_cache_file)
            self.phenotype.save(pheno_cache_file)

            data_loaded = True

        # log case/control ratio
        if self.cfg.classification and self.cfg.is_train:
            self._logger.info(
                f"Currently {self.phenotype.case_control_ratio * 100:.2f}% of samples are cases."
            )

    def _load_plink_data(self):
        """Loads PLINK data and returns it as a DataSet object."""
        self._logger.info("Loading .fam file")
        fam_df: pd.DataFrame = plink_utils.load_fam_data(self.cfg.paths["fam_path"])
        n_samples_genotype: int = len(fam_df)
        self._logger.info("Loading .master file")
        covar_keys = (
            self.cfg.covar_config.covar_keys if self.cfg.covar_config.include_covars else []
        )
        master_df_keys = (
            ["IID", "population"]
            + ([] if self.cfg.simulate else [self.cfg.phenotype_id])
            + covar_keys
        )
        master_df: pd.DataFrame = plink_utils.load_master_data(
            self.cfg.paths["master_path"], master_df_keys, self.cfg.sex
        )
        if self.cfg.simulate:
            master_df[self.cfg.phenotype_id] = 0

        self._logger.info("Processing .master file")
        master_df = plink_utils.process_master_df(
            fam_df=fam_df,
            master_df=master_df,
            classification=self.cfg.classification,
            phenotype_id=self.cfg.phenotype_id,
            populations=self.cfg.populations,
        )

        self._logger.info("Loading .bim file")
        bim_df: pd.DataFrame = plink_utils.load_bim_data(
            self.cfg.paths["bim_path"],
            include_x_chromosome=self.cfg.include_x_chromosome,
        )

        # init genotype
        bed_path = self.cfg.paths["bed_path"]
        if isinstance(bed_path, bytes):
            bed_path = bed_path.decode()
        n_samples = len(master_df)
        self.genotype = GenoSet.from_plink(bim_df, bed_path, n_samples_genotype, n_samples)

        # init phenotype
        self.phenotype = PhenoSet.from_plink(self.cfg, master_df)

    def setup(self, skip_maf: bool = False):
        if not skip_maf:
            self._compute_maf()

        self._logger.info("Setting up sampling...")
        self._setup_sampling()

        # logs
        if self.cfg.variant_sampling.max_features > self.genotype.shape[1]:
            self._logger.warning(
                f"max_features ({self.cfg.variant_sampling.max_features}) is larger than the number of variants in train_set. This can lead to issues during sampling..."
            )
        if self.cfg.sample_sampling.max_samples and (
            self.cfg.sample_sampling.max_samples > self.genotype.shape[0]
        ):
            self._logger.warning(
                f"max_samples ({self.cfg.sample_sampling.max_samples}) is larger than the number of samples in train_set ({self.genotype.shape[0]}) samples). This can lead to issues during sampling..."
            )

        if (self.cfg.sample_sampling.strat == "balanced") and self.cfg.classification:
            n_cases = int(np.sum(self.phenotype.y))
            cc_ratio = 100 * n_cases / len(self)
            self._logger.info(
                f"Got {n_cases} cases in the train set ({cc_ratio:.2f} %). Balancing with k={self.cfg.sample_sampling.k} ({self.cfg.sample_sampling.k * n_cases * 2} samples per batch)."
            )

    def _setup_sampling(self):
        n = self.genotype.shape[1]
        self.genotype.annotation_df["sampling_prob"] = np.full(n, 1.0 / n)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.phenotype)

    def preload_to_memory(
        self,
        sample_idxs: npt.ArrayLike | None = None,
        chunk_size: int = 10_000,
    ) -> None:
        """Preload all variants for the given samples into memory (uint8).

        Eliminates per-batch pgenlib disk reads. Suitable for fixed sample sets
        like val/test. If sample_idxs is None, uses all samples in the dataset.
        Memory cost: n_samples × n_variants bytes.
        """
        if sample_idxs is None:
            sample_idxs = self.phenotype.sample_idxs
        self.genotype.preload(sample_idxs, chunk_size=chunk_size)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the dataset."""
        return self.genotype.shape

    @property
    def populations(self) -> List[str]:
        return self.cfg.populations

    @property
    def phenotype_id(self) -> str:
        return self.cfg.phenotype_id

    def _parse_idxs(
        self,
        key: Tuple[int | slice | List | npt.ArrayLike, int | slice | List | npt.ArrayLike],
    ) -> Tuple[int | npt.ArrayLike, int | npt.ArrayLike]:
        samples, variants = key

        # get sample_idxs
        if isinstance(samples, slice):
            sample_idxs = self.phenotype.sample_idxs[samples]
        elif isinstance(samples, (int, list, np.ndarray)):
            if isinstance(samples, int):
                sample_idxs = np.array([samples], dtype=np.uint32)
            else:
                sample_idxs = np.array(samples, dtype=np.uint32)

            if not set(sample_idxs).issubset(self.phenotype.sample_idxs):
                raise IndexError("Not all sample_idxs used for slicing could be found in dataset")
        else:
            raise TypeError(f"Invalid sample index type: {type(samples)}")

        # get variant_idxs
        if isinstance(variants, slice):
            if variants.start is not None or variants.stop is not None or variants.step is not None:
                raise ValueError("Only full slice ':' is supported for variant indices")
            variant_idxs = self.genotype.variant_idxs
        elif isinstance(variants, (int, list, np.ndarray)):
            if isinstance(variants, int):
                variant_idxs = np.array([variants], dtype=np.uint32)
            else:
                variant_idxs = np.array(variants, dtype=np.uint32)

            if not set(variant_idxs).issubset(self.genotype.variant_idxs):
                variant_idxs_not_found = np.setdiff1d(variant_idxs, self.genotype.variant_idxs)
                raise IndexError(
                    f"Could not find {len(variant_idxs_not_found)} variant_idxs in dataset: {variant_idxs_not_found}"
                )
        else:
            raise TypeError(f"Invalid variants index type: {type(variants)}")

        return sample_idxs, variant_idxs

    def __getitem__(
        self,
        key: int | Tuple[int | slice | list | np.ndarray, int | slice | list | np.ndarray],
    ) -> DataBatch | List[DataBatch]:
        if isinstance(key, (int, list, np.ndarray)):
            key = (key, slice(None))
        sample_idxs, variant_idxs = self._parse_idxs(key)
        X, geno_annotation_df = self.genotype[sample_idxs, variant_idxs]
        y, pheno_annotation_df, residuals, covar_pred = self.phenotype[sample_idxs]

        return DataBatch(
            self.cfg,
            X,
            geno_annotation_df,
            y,
            pheno_annotation_df,
            residuals,
            covar_pred=covar_pred,
        )

    def _compute_maf(self):
        """Compute MAF on train samples and merge into annotation_df.

        Loads from cache if available.
        """
        if self._cache_path is None:
            self._setup_cache_path()

        sid = data_set_utils.hash_ndarray(self.phenotype.sample_idxs)
        maf_cache = Path(self._cache_path) / f"maf_{sid}.parquet"
        # Missingness is computed on the MAF-filtered variant set, so its cache key
        # includes the MAF threshold to ensure invalidation when the threshold changes.
        missingness_cache = (
            Path(self._cache_path) / f"missingness_{sid}_{self.cfg.maf_threshold:.6f}.parquet"
        )
        # A0_FREQ is computed on the post-filter variant set, so its cache key includes
        # the MAF and missingness thresholds to ensure invalidation when either changes.
        a0freq_cache = Path(self._cache_path) / (
            f"a0freq_{sid}_{self.cfg.maf_threshold:.6f}_{self.cfg.missingness_threshold:.6f}.parquet"
        )

        # --- MAF (pre-filter) ---
        if maf_cache.exists():
            self._logger.info("Loading cached MAF...")
            cached = pd.read_parquet(maf_cache)
            if (
                isinstance(cached, pd.DataFrame)
                and ("MAF" in cached)
                and np.array_equal(cached.index.values, self.genotype.annotation_df.index.values)
            ):
                self.genotype.annotation_df["MAF"] = cached["MAF"].values
            else:
                self._logger.warning("Cached MAF invalid for current genotype; recomputing.")
                maf_cache.unlink(missing_ok=True)

        if "MAF" not in self.genotype.annotation_df.columns:
            self._logger.info("Computing MAF on train samples")
            keep_iids = self.phenotype.annotation_df[["fid", "iid"]]
            freq_df = plink_utils.calculate_maf(
                bed_path=self.cfg.paths["bed_path"],
                bim_path=self.cfg.paths["bim_path"],
                fam_path=self.cfg.paths["fam_path"],
                keep_iids=keep_iids,
            )
            maf = freq_df.set_index("SNP")["MAF"]
            self.genotype.annotation_df["MAF"] = self.genotype.annotation_df["snp"].map(maf)

            try:
                pd.DataFrame(
                    {"MAF": self.genotype.annotation_df["MAF"].values},
                    index=self.genotype.annotation_df.index,
                ).sort_index().to_parquet(maf_cache, index=True)
            except Exception as e:
                self._logger.warning(f"Could not write MAF cache {maf_cache}: {e}")

        # --- MAF filter ---
        if self.cfg.maf_threshold > 0.0:
            n_before = len(self.genotype.annotation_df)
            self.genotype.annotation_df = self.genotype.annotation_df[
                self.genotype.annotation_df["MAF"] >= self.cfg.maf_threshold
            ]
            self._logger.info(
                f"Filtered out {n_before - len(self.genotype.annotation_df)} variants with MAF < {self.cfg.maf_threshold}"
            )

        # --- Missingness (post-MAF-filter) ---
        if missingness_cache.exists():
            self._logger.info("Loading cached missingness...")
            cached_miss = pd.read_parquet(missingness_cache)
            if (
                isinstance(cached_miss, pd.DataFrame)
                and ("MISSINGNESS" in cached_miss)
                and np.array_equal(
                    cached_miss.index.values, self.genotype.annotation_df.index.values
                )
            ):
                self.genotype.annotation_df["MISSINGNESS"] = cached_miss["MISSINGNESS"].values
            else:
                self._logger.warning(
                    "Cached missingness invalid for current genotype; recomputing."
                )
                missingness_cache.unlink(missing_ok=True)

        if "MISSINGNESS" not in self.genotype.annotation_df.columns:
            self._logger.info(
                "Computing per-variant missingness on train samples (n_variants=%d)...",
                len(self.genotype.annotation_df),
            )
            missingness = plink_utils.calculate_missingness(
                bed_reader=self.genotype._bed,
                variant_idxs=self.genotype.annotation_df.index.values,
                sample_idxs=self.phenotype.sample_idxs,
            )
            self.genotype.annotation_df["MISSINGNESS"] = missingness

            try:
                pd.DataFrame(
                    {"MISSINGNESS": self.genotype.annotation_df["MISSINGNESS"].values},
                    index=self.genotype.annotation_df.index,
                ).sort_index().to_parquet(missingness_cache, index=True)
            except Exception as e:
                self._logger.warning(f"Could not write missingness cache {missingness_cache}: {e}")

        # --- Missingness filter ---
        if self.cfg.missingness_threshold < 1.0:
            n_before = len(self.genotype.annotation_df)
            self.genotype.annotation_df = self.genotype.annotation_df[
                self.genotype.annotation_df["MISSINGNESS"] <= self.cfg.missingness_threshold
            ]
            self._logger.info(
                f"Filtered out {n_before - len(self.genotype.annotation_df)} variants "
                f"with missingness > {self.cfg.missingness_threshold}"
            )

        # --- A0_FREQ (post-filter) ---
        if a0freq_cache.exists():
            self._logger.info("Loading cached A0_FREQ...")
            cached_a0 = pd.read_parquet(a0freq_cache)
            if (
                isinstance(cached_a0, pd.DataFrame)
                and ("A0_FREQ" in cached_a0)
                and np.array_equal(cached_a0.index.values, self.genotype.annotation_df.index.values)
            ):
                self.genotype.annotation_df["A0_FREQ"] = cached_a0["A0_FREQ"].values
            else:
                self._logger.warning("Cached A0_FREQ invalid; recomputing.")
                a0freq_cache.unlink(missing_ok=True)

        if "A0_FREQ" not in self.genotype.annotation_df.columns:
            self._logger.info(
                "Computing empirical A0_FREQ from raw training genotypes "
                "(n_variants=%d, subsample≤5000)...",
                len(self.genotype.annotation_df),
            )
            a0_freq = plink_utils.calculate_a0_freq(
                bed_reader=self.genotype._bed,
                variant_idxs=self.genotype.annotation_df.index.values,
                sample_idxs=self.phenotype.sample_idxs,
            )
            self.genotype.annotation_df["A0_FREQ"] = a0_freq
            ann = self.genotype.annotation_df
            n_flipped = int((ann["A0_FREQ"] > 0.5).sum())
            self._logger.info(
                "A0_FREQ computed: mean=%.4f vs MAF mean=%.4f "
                "(%d/%d variants have a0=major allele, i.e. A0_FREQ>0.5)",
                ann["A0_FREQ"].mean(),
                ann["MAF"].mean(),
                n_flipped,
                len(ann),
            )

            try:
                pd.DataFrame(
                    {"A0_FREQ": self.genotype.annotation_df["A0_FREQ"].values},
                    index=self.genotype.annotation_df.index,
                ).sort_index().to_parquet(a0freq_cache, index=True)
            except Exception as e:
                self._logger.warning(f"Could not write A0_FREQ cache {a0freq_cache}: {e}")

    def sample_sample_idxs(
        self, seed: int | None = None, skip_class_check: bool = False
    ) -> npt.NDArray:
        """Sample a batch of sample indices according to the configured sampling strategy.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Sorted array of sample indices (uint32)
        """
        fix_samples = self.cfg.sample_sampling.fix_balanced_samples
        if fix_samples:
            rng = np.random.default_rng(self.cfg.sample_sampling.split_seed)
        else:
            rng = np.random.default_rng(seed)

        sample_idxs = self.phenotype.sample_idxs
        if self.cfg.sample_sampling.max_samples is None and not (
            self.cfg.classification and self.cfg.sample_sampling.strat in ["stratify", "balanced"]
        ):
            return sample_idxs

        if fix_samples and hasattr(self, "_fixed_batch_sample_idxs"):
            return self._fixed_batch_sample_idxs

        resample_count = 0
        max_resamples = 10
        while True:
            if self.cfg.classification and (
                self.cfg.sample_sampling.strat in ["stratify", "balanced"]
            ):
                strategy = (
                    "stratified" if self.cfg.sample_sampling.strat == "stratify" else "balanced"
                )

                if self.cfg.sample_sampling.balance_pops:
                    batch_sample_idxs = []
                    for population in self.cfg.populations:
                        pop_mask = self.phenotype.annotation_df["population"] == population

                        if strategy == "stratified":
                            samples_per_pop = self.cfg.sample_sampling.max_samples // len(
                                self.cfg.populations
                            )
                            pop_sample_idxs = data_set_utils.adaptive_sampling(
                                sample_idxs=sample_idxs[pop_mask],
                                phenotypes=self.phenotype.y[pop_mask],
                                classification=self.cfg.classification,
                                size=samples_per_pop,
                                strategy=strategy,
                                rng=rng,
                            )
                        else:  # balanced
                            pop_sample_idxs = data_set_utils.adaptive_sampling(
                                sample_idxs=sample_idxs[pop_mask],
                                phenotypes=self.phenotype.y[pop_mask],
                                classification=self.cfg.classification,
                                strategy=strategy,
                                k=self.cfg.sample_sampling.k,
                                rng=rng,
                            )

                        batch_sample_idxs.append(pop_sample_idxs)

                    batch_sample_idxs = np.concatenate(batch_sample_idxs)
                else:
                    batch_sample_idxs = data_set_utils.adaptive_sampling(
                        sample_idxs=sample_idxs,
                        phenotypes=self.phenotype.y,
                        classification=self.cfg.classification,
                        size=self.cfg.sample_sampling.max_samples,
                        strategy=strategy,
                        k=self.cfg.sample_sampling.k if strategy == "balanced" else None,
                        rng=rng,
                    )

            elif self.cfg.sample_sampling.balance_pops:
                samples_per_pop = self.cfg.sample_sampling.max_samples // len(self.cfg.populations)
                batch_sample_idxs = []
                for population in self.cfg.populations:
                    pop_mask = self.phenotype.annotation_df["population"] == population
                    batch_sample_idxs.append(
                        rng.choice(
                            sample_idxs[pop_mask],
                            size=samples_per_pop,
                            replace=True,
                        )
                    )
                batch_sample_idxs = np.concatenate(batch_sample_idxs)
            else:
                batch_sample_idxs = rng.choice(
                    sample_idxs,
                    size=self.cfg.sample_sampling.max_samples,
                    replace=False,
                )

            if self.cfg.classification and not skip_class_check:
                y_batch = self.phenotype.annotation_df.loc[batch_sample_idxs, "y"].values
                if len(np.unique(y_batch)) < 2:
                    logging.warning(
                        f"Resampling attempt {resample_count + 1}: "
                        "Only one class found in the sample. Consider setting balance_classes=True or increasing max_samples."
                    )
                    resample_count += 1
                    if resample_count > max_resamples:
                        raise ValueError(
                            "Could not find samples with more than one class after 10 resamples."
                        )
                    continue

            break

        self._fixed_batch_sample_idxs = np.sort(batch_sample_idxs).astype(np.uint32)
        return self._fixed_batch_sample_idxs

    def sample_variant_idxs(self, batch_idx: int, seed: int | None = None) -> npt.NDArray:
        """Sample a batch of variant indices uniformly at random.

        Args:
            batch_idx: Index of the current batch
            seed: Random seed for reproducibility

        Returns:
            Sorted array of unique variant indices (uint32)
        """
        rng = np.random.default_rng(seed)

        variant_idxs = self.genotype.variant_idxs
        pos_mask = np.arange(self.genotype.shape[1])

        if self.cfg.variant_sampling.max_features > len(pos_mask):
            sampled_pos_mask_idxs = np.arange(len(pos_mask))
        else:
            p = self.genotype.annotation_df["sampling_prob"].values[pos_mask]
            sampled_pos_mask_idxs = rng.choice(
                len(pos_mask),
                size=self.cfg.variant_sampling.max_features,
                replace=False,
                p=p,
            )

        batch_variant_idxs = variant_idxs[pos_mask[sampled_pos_mask_idxs]]

        return np.unique(batch_variant_idxs).astype(np.uint32)

    def sample_batch_idxs(
        self, batch_idx: int, seed: int | None = None
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Sample a batch of samples and variants.

        Args:
            batch_idx: Index of the current batch
            seed: Random seed for reproducibility

        Returns:
            Tuple of (sample_indices, variant_indices)
        """
        batch_sample_idxs = self.sample_sample_idxs(seed)
        batch_variant_idxs = self.sample_variant_idxs(batch_idx, seed)

        return batch_sample_idxs, batch_variant_idxs

    def get_covars(self, sample_idxs: npt.ArrayLike | None = None) -> DataBatch:
        if sample_idxs is not None:
            sample_idxs = np.sort(sample_idxs)
            pheno_annotation_df = self.phenotype.annotation_df.loc[sample_idxs]
            Z = pheno_annotation_df[self.phenotype.covar_keys].to_numpy()
            y = pheno_annotation_df["y"].values
            residuals = (
                pheno_annotation_df["residuals"].values
                if self.phenotype.residuals is not None
                else None
            )
            covar_pred = (
                pheno_annotation_df["covar_pred"].values
                if self.phenotype.covar_pred is not None
                else None
            )
        else:
            Z = self.phenotype.get_covars()
            y = self.phenotype.y
            residuals = self.phenotype.residuals
            covar_pred = self.phenotype.covar_pred
            pheno_annotation_df = self.phenotype.annotation_df

        return DataBatch(
            self.cfg,
            Z,
            self.genotype.annotation_df,
            y,
            pheno_annotation_df,
            residuals,
            covar_pred=covar_pred,
            type="covar",
        )

    def get_labels(self, use_resids: bool = False) -> npt.ArrayLike:
        """Get labels for training - either original y or residuals."""
        if use_resids:
            if self.phenotype.residuals is None:
                raise ValueError("No residuals set!")
            return self.phenotype.residuals
        return self.phenotype.y

    def get_sampling_weight(self) -> npt.ArrayLike | None:
        """Get sampling weights based on case/control ratio from original binary labels."""
        y = self.phenotype.y
        unique_labels, counts = np.unique(y, return_counts=True)

        if len(unique_labels) == 2:  # Binary case
            total_samples = len(y)
            weights = np.zeros_like(y, dtype=float)

            for label, count in zip(unique_labels, counts):
                weight = total_samples / count
                weights[y == label] = weight

            return weights
        else:
            return None

    def get_background_sample_idxs(self, n_samples: int, seed: int) -> npt.NDArray:
        rng = np.random.default_rng(seed)

        sample_idxs = self.phenotype.sample_idxs
        background_sample_idxs = data_set_utils.adaptive_sampling(
            sample_idxs=sample_idxs,
            phenotypes=self.phenotype.y,
            classification=self.cfg.classification,
            size=n_samples,
            strategy="stratified",
            rng=rng,
        )

        return np.sort(background_sample_idxs).astype(np.uint32)
