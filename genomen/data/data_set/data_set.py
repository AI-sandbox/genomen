"""Class to hold genotype and phenotype data and retrieve batches."""

import hashlib
import logging
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import pgenlib as pg
from joblib import Parallel, delayed
from sklearn.feature_selection import chi2, f_regression

from ... import utils
from ..sources import plink_utils
from . import utils as data_set_utils
from .config import DataSetConfig
from .data_batch import DataBatch
from .geno_set import GenoSet
from .pheno_set import PhenoSet

if TYPE_CHECKING:
    from ...model.geno.weak_geno_estimator import WeakGenoEstimator


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

        # logs
        if self.cfg.is_train:
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

        self._logger.debug(f"Shape of data set is {self.genotype.shape}")

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
                self.genotype = GenoSet.from_file(geno_cache_file)
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
        master_df_keys = ["IID", "population", self.cfg.phenotype_id] + covar_keys
        master_df: pd.DataFrame = plink_utils.load_master_data(
            self.cfg.paths["master_path"], master_df_keys, self.cfg.sex
        )

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

        # filter MAF
        if self.cfg.maf_threshold > 0.0:
            self._logger.info(f"Filtering variants with MAF threshold: {self.cfg.maf_threshold}")

            # calculate maf
            freq_df = plink_utils.calculate_maf(
                bed_path=self.cfg.paths["bed_path"],
                bim_path=self.cfg.paths["bim_path"],
                fam_path=self.cfg.paths["fam_path"],
            )
            # merge MAF with bim_df
            bim_df = bim_df.merge(
                freq_df[["SNP", "MAF"]], left_on="snp", right_on="SNP", how="left"
            )
            bim_df.drop(columns=["SNP"], inplace=True)

            # filter based on MAF
            n_variants_before = len(bim_df)
            bim_df = bim_df[bim_df["MAF"] >= self.cfg.maf_threshold]
            self._logger.info(
                f"Filtered out {n_variants_before - len(bim_df)} variants with MAF < {self.cfg.maf_threshold}"
            )

        # init genotype
        pgen_reader = partial(
            pg.PgenReader, self.cfg.paths["bed_path"], raw_sample_ct=n_samples_genotype
        )
        n_samples = len(master_df)
        self.genotype = GenoSet.from_plink(bim_df, pgen_reader, n_samples)

        # init phenotype
        self.phenotype = PhenoSet.from_plink(self.cfg, master_df)

    def _setup_sampling(self):
        if self.cfg.variant_sampling.strat == "window":
            sorted_variant_df = self.genotype.annotation_df.copy()
            sorted_variant_df.rename(
                columns={"chr_name": "chrom", "chr_position": "pos"}, inplace=True
            )
            sorted_variant_df["chrom"] = sorted_variant_df["chrom"].replace(
                {"X": "23", "Y": "24", "M": "25"}
            )
            sorted_variant_df["chrom"] = sorted_variant_df["chrom"].astype(int)
            sorted_variant_df = sorted_variant_df.sort_values(by=["chrom", "pos"])

            self.sorted_variant_idxs = sorted_variant_df.index.values
            self.num_windows = self.shape[1] // self.cfg.variant_sampling.stride
            overlap = self.cfg.variant_sampling.max_features - self.cfg.variant_sampling.stride
            self._logger.info(f"Mapped SNPs to {self.num_windows} windows with {overlap} overlap")
        if self.cfg.variant_sampling.strat == "GWAS":
            self.genotype.annotation_df["sampling_prob"] = data_set_utils.get_gwas_priors(
                self.genotype.annotation_df["snp"].values,
                self.cfg.variant_sampling.gwas_config,
            )
        elif self.cfg.variant_sampling.strat == "MAF":
            self.genotype.annotation_df["sampling_prob"] = (
                self.genotype.annotation_df["MAF"].values / self.genotype.annotation_df["MAF"].sum()
            )
        elif self.cfg.variant_sampling.strat == "LD":
            # setup eps
            self._initial_eps = self.cfg.variant_sampling.ld_config.eps
            self._current_eps = self.cfg.variant_sampling.ld_config.eps
            if self.cfg.is_train and self.cfg.variant_sampling.ld_config.eps_schedule != "constant":
                self._logger.info(
                    f"Initialized epsilon scheduling: {self.cfg.variant_sampling.ld_config.eps_schedule} with initial eps={self._initial_eps}, step_size={self.cfg.variant_sampling.ld_config.eps_step_size}"
                )
        else:
            self.genotype.annotation_df["sampling_prob"] = np.ones(
                (self.genotype.shape[1],), dtype=float
            )

    def _update_eps(self, batch_idx: int):
        if (
            not hasattr(self, "_current_eps")
            or self.cfg.variant_sampling.ld_config.eps_schedule == "constant"
        ):
            return

        if self.cfg.variant_sampling.ld_config.eps_schedule == "step":
            # Step decay: current_eps = current_eps - step_size
            self._current_eps = (
                self._current_eps - self.cfg.variant_sampling.ld_config.eps_step_size
            )

        self._current_eps = max(0.0, self._current_eps)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.phenotype)

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
            if samples.start is not None or samples.stop is not None or samples.step is not None:
                raise ValueError("Only full slice ':' is supported for sample indices")
            sample_idxs = self.phenotype.sample_idxs
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
        y, pheno_annotation_df, residuals = self.phenotype[sample_idxs]

        return DataBatch(self.cfg, X, geno_annotation_df, y, pheno_annotation_df, residuals)

    def score_variants(self, ram_mb: int):
        if self.cfg.variant_sampling.strat != "LD":
            return

        initiated = False

        if self._cache_path is None:
            self._setup_cache_path()

        # load from cache if available
        sid = data_set_utils.hash_ndarray(self.phenotype.sample_idxs)
        vid = data_set_utils.hash_ndarray(self.genotype.variant_idxs)
        prune_kb = self.cfg.variant_sampling.ld_config.prune_kb
        prune_step = self.cfg.variant_sampling.ld_config.prune_step
        prune_r2 = self.cfg.variant_sampling.ld_config.prune_r2
        tau = self.cfg.variant_sampling.ld_config.tau
        ld_window_kb = self.cfg.variant_sampling.ld_config.ld_window_kb
        ld_window = self.cfg.variant_sampling.ld_config.ld_window
        temp = self.cfg.variant_sampling.ld_config.temp
        max_score = self.cfg.variant_sampling.ld_config.max_score
        block_cache = (
            Path(self._cache_path)
            / f"cache_df_{sid}_{vid}_{prune_kb}_{prune_step}_{prune_r2}_{tau}_{ld_window_kb}_{ld_window}_{temp}_{max_score}.parquet"
        )
        if block_cache.exists():
            self._logger.info("Loading cached variant block ids and scores from cache...")
            cached = pd.read_parquet(block_cache)

            if (
                isinstance(cached, pd.DataFrame)
                and ("scores" in cached)
                and ("block_idx" in cached)
                and ("block_id" in cached)
                and len(cached) == len(self.genotype.annotation_df)
                and np.array_equal(cached.index.values, self.genotype.annotation_df.index.values)
            ):
                self.genotype.annotation_df["scores"] = cached["scores"].astype(float).values
                self.genotype.annotation_df["block_idx"] = cached["block_idx"].values
                self.genotype.annotation_df["block_id"] = cached["block_id"].values
                initiated = True
            else:
                self._logger.warning(
                    "Cached blocks and scores invalid for current genotype; recomputing."
                )

        if not initiated:
            self._logger.info("Computing LD blocks...")
            # get ld blocks
            ld_df = plink_utils.compute_ld(
                bed_path=self.cfg.paths["bed_path"],
                bim_path=self.cfg.paths["bim_path"],
                fam_path=self.cfg.paths["fam_path"],
                blocks_max_kb=self.cfg.variant_sampling.blocks_max_kb,
                maf_threshold=self.cfg.maf_threshold,
                prune_kb=prune_kb,
                prune_step=prune_step,
                prune_r2=prune_r2,
                tau=tau,
                ld_window_kb=ld_window_kb,
                ld_window=ld_window,
                include_x=self.cfg.include_x_chromosome,
                ram_mb=ram_mb,
            )

            # merge block_id with genotype annotation_df
            self.genotype.annotation_df = self.genotype.annotation_df.merge(
                ld_df, on="snp", how="left"
            ).set_index(self.genotype.annotation_df.index)

            self.genotype.annotation_df["block_idx"] -= 1  # null idx
            self.genotype.annotation_df["block_idx"] = self.genotype.annotation_df[
                "block_idx"
            ].fillna(-1)

            self._logger.info("Computing scores...")
            assigned_mask = self.genotype.annotation_df["block_idx"] != -1
            y = self.get_labels(use_resids=False)
            scores = pd.Series(1.0, index=self.genotype.annotation_df.index, dtype=float)

            # Score all variants
            def _score_chunk(variant_idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                X, _ = self.genotype[self.phenotype.sample_idxs, variant_idxs]
                if self.cfg.classification:
                    s, _ = chi2(X, y)
                else:
                    s, _ = f_regression(X, y)
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                s = np.clip(s, a_min=0.0, a_max=max_score)
                return variant_idxs, s

            n_jobs = min(12, max(1, (os.cpu_count() // 2)))
            max_bytes = (ram_mb * 1024**2) // n_jobs
            denom = max(1, len(self) * 4)
            variant_idxs = self.genotype.variant_idxs
            v_chunk = max(1, min(len(variant_idxs), max_bytes // denom))
            vidx_chunks = [
                variant_idxs[i : i + v_chunk] for i in range(0, len(variant_idxs), v_chunk)
            ]
            results = Parallel(
                n_jobs=n_jobs,
                prefer="threads",
                batch_size=1,  # one chunk per task (don’t bundle)
                pre_dispatch=n_jobs,
            )(delayed(_score_chunk)(idxs) for idxs in vidx_chunks)

            for idxs, s in results:
                scores.loc[idxs] = s

            self._logger.info("Normalizing scores per block...")
            blocks = self.genotype.annotation_df.groupby("block_idx").apply(
                lambda d: d.index.values, include_groups=False
            )
            for variant_idxs in blocks:
                scores.loc[variant_idxs] = utils.safe_softmax(scores[variant_idxs], temp=temp)

            self.genotype.annotation_df["scores"] = scores.values

            # save to cache
            df = pd.DataFrame(
                {
                    "scores": scores.values,
                    "block_idx": self.genotype.annotation_df["block_idx"].values,
                    "block_id": self.genotype.annotation_df["block_id"].values,
                },
                index=self.genotype.annotation_df.index,
            ).sort_index()
            try:
                df.to_parquet(block_cache, index=True)
            except Exception as e:
                self._logger.warning(f"Could not write scores cache {block_cache}: {e}")

        # log ld block statistics
        assigned_mask = self.genotype.annotation_df["block_idx"] != -1
        n_blocks = self.genotype.annotation_df["block_idx"].max() + 1
        num_not_assigned = (~assigned_mask).sum()
        _, ld_block_len = np.unique(
            self.genotype.annotation_df["block_idx"].loc[assigned_mask].values,
            return_counts=True,
        )
        self._logger.info(
            f"Mapped SNPs to {int(n_blocks)} LD blocks with an average length of {ld_block_len.mean():.2f}. {num_not_assigned} SNPs have not been assigned to any LD block."
        )
        if n_blocks > self.cfg.variant_sampling.max_features:
            self._logger.warning(
                f"Requested max_features={self.cfg.variant_sampling.max_features} < number of LD blocks. Consider increasing max_features"
            )

    def sample_batch_idxs(
        self, batch_idx: int, seed: int | None = None
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Sample a batch of samples and variants.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (sample_indices, variant_indices)
        """
        rng = np.random.default_rng(seed)

        # sample idxs
        sample_idxs = self.phenotype.sample_idxs
        if (self.cfg.sample_sampling.max_samples is not None) or (
            self.cfg.sample_sampling.strat == "balanced" and self.cfg.classification
        ):
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
                    samples_per_pop = self.cfg.sample_sampling.max_samples // len(
                        self.cfg.populations
                    )
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

                if self.cfg.classification:
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
        else:
            batch_sample_idxs = sample_idxs

        self._update_eps(batch_idx)

        if (self.cfg.variant_sampling.strat == "window") and (batch_idx < self.num_windows):
            window_size = self.cfg.variant_sampling.max_features
            start_idx = batch_idx * self.cfg.variant_sampling.stride

            if batch_idx == self.num_windows - 1:
                batch_variant_idxs = self.sorted_variant_idxs[-window_size:]
            else:
                batch_variant_idxs = self.sorted_variant_idxs[start_idx : start_idx + window_size]
        elif self.cfg.variant_sampling.strat == "LD":
            k = self.cfg.variant_sampling.max_features

            assigned_mask = self.genotype.annotation_df["block_idx"] != -1
            N = sum(~assigned_mask)
            blocks = (
                self.genotype.annotation_df.loc[assigned_mask]
                .groupby("block_idx")
                .apply(lambda d: d.index.values, include_groups=False)
            )
            B = len(blocks)

            def sample_from_block(block_indices: np.ndarray, samples_per_block: int) -> int:
                if samples_per_block <= 0:
                    return np.array([], dtype=np.uint32)

                # exploration
                if rng.random() < self.cfg.variant_sampling.ld_config.eps:
                    replace = samples_per_block > len(block_indices)
                    picks = rng.choice(block_indices, size=samples_per_block, replace=replace)
                    return picks.astype(np.uint32)

                # exploitation: sample ~ p
                sampling_probs = self.genotype.annotation_df.loc[
                    block_indices, "sampling_prob"
                ].to_numpy(float)
                replace = samples_per_block > len(block_indices)
                picks = rng.choice(
                    block_indices, size=samples_per_block, p=sampling_probs, replace=replace
                )
                return picks.astype(np.uint32)

            def epsilon_greedy_unassigned(unassigned_idxs: np.ndarray, n_select: int) -> np.ndarray:
                n_greedy = int(n_select * (1 - self.cfg.variant_sampling.ld_config.eps))
                n_random = n_select - n_greedy

                # pick n_greedy variants
                if n_greedy > 0:  # sample from sampling_prob
                    replace = n_greedy > len(unassigned_idxs)
                    if replace:
                        self._logger.warning(
                            f"Requested max_features={n_greedy} > unassigned SNPs ({len(unassigned_idxs)}). Sampling with replacement."
                        )
                    p = self.genotype.annotation_df.loc[unassigned_idxs, "sampling_prob"].to_numpy(
                        float
                    )
                    greedy_idxs = rng.choice(
                        unassigned_idxs,
                        size=n_greedy,
                        p=p,
                        replace=replace,
                    )
                else:
                    greedy_idxs = np.array([], dtype=np.uint32)

                # pick n_random variants randomly
                if n_random > 0:
                    rest_idxs = np.setdiff1d(unassigned_idxs, greedy_idxs, assume_unique=True)
                    replace = n_random > len(rest_idxs)
                    random_idxs = rng.choice(rest_idxs, size=n_random, replace=replace)
                else:
                    random_idxs = np.array([], dtype=np.uint32)

                return np.concatenate([greedy_idxs, random_idxs])

            if B == 0:  # all variants unassigned
                unassigned_variant_idxs = self.genotype.variant_idxs[~assigned_mask]
                if len(unassigned_variant_idxs) == 0:
                    raise ValueError("No LD blocks and no unassigned SNPs available")

                batch_variant_idxs = epsilon_greedy_unassigned(unassigned_variant_idxs, n_select=k)

                batch_variant_idxs = epsilon_greedy_unassigned(unassigned_variant_idxs, n_select=k)
            else:
                if k < B:  # more than k LD blocks
                    n_blocks = int(B / (B + N) * k)
                    n_unassigned = k - n_blocks

                    block_idxs = rng.choice(np.arange(B), size=n_blocks, replace=False)
                    selected_blocks = blocks.iloc[block_idxs]
                    block_picks = [sample_from_block(block, 1) for block in selected_blocks]
                    block_variant_idxs = np.concatenate(block_picks).astype(np.uint32)

                    unassigned_variant_idxs = self.genotype.variant_idxs[~assigned_mask]
                    selected_unassigned_variant_idxs = epsilon_greedy_unassigned(
                        unassigned_variant_idxs, n_select=n_unassigned
                    )
                    batch_variant_idxs = np.concatenate(
                        [block_variant_idxs, selected_unassigned_variant_idxs],
                        dtype=np.uint32,
                    )
                else:  # k >= v_per_block * # LD blocks
                    block_picks = [sample_from_block(block, 1) for block in blocks]
                    block_variant_idxs = np.concatenate(block_picks).astype(np.uint32)
                    left = k - B

                    # fill remainder from unassigned SNPs
                    if left > 0:
                        unassigned_variant_idxs = self.genotype.variant_idxs[~assigned_mask]
                        if len(unassigned_variant_idxs) == 0:
                            self._logger.warning(
                                f"No unassigend SNPs available to fill remanider of {left} variants. Using only per block variants!"
                            )
                            selected_unassigned_variant_idxs = np.array([], dtype=np.uint32)
                        else:
                            selected_unassigned_variant_idxs = epsilon_greedy_unassigned(
                                unassigned_variant_idxs, n_select=left
                            )
                        batch_variant_idxs = np.concatenate(
                            [block_variant_idxs, selected_unassigned_variant_idxs],
                            dtype=np.uint32,
                        )
                    else:
                        batch_variant_idxs = block_variant_idxs

            batch_variant_idxs = np.unique(batch_variant_idxs)
            batch_variant_idxs = np.sort(batch_variant_idxs)
            batch_variant_idxs = np.unique(batch_variant_idxs)
            batch_variant_idxs = np.sort(batch_variant_idxs)
        else:  # random
            if (self.cfg.variant_sampling.strat == "window") and (batch_idx == self.num_windows):
                self._logger.info(
                    "Did not converge on entirety of windows. Proceeding with random sampling!"
                )

            if self.cfg.variant_sampling.strat == "chromosome":
                chrom = self.genotype.annotation_df["chr_name"].values
                unique_chroms = np.unique(chrom)
                batch_chrom = rng.choice(unique_chroms)
                pos_mask = np.where(chrom == batch_chrom)[0]
            else:
                pos_mask = np.arange(self.genotype.shape[1])

            variant_idxs = self.genotype.variant_idxs
            if self.cfg.variant_sampling.max_features > len(pos_mask):
                sampled_pos_mask_idxs = np.arange(len(pos_mask))
            else:
                sampling_probs = self.genotype.annotation_df["sampling_prob"].values[pos_mask]
                p = sampling_probs / np.sum(sampling_probs, dtype=float)
                sampled_pos_mask_idxs = rng.choice(
                    len(pos_mask),
                    size=self.cfg.variant_sampling.max_features,
                    replace=False,
                    p=p,
                )

            batch_variant_idxs = variant_idxs[pos_mask[sampled_pos_mask_idxs]]

        return np.sort(batch_sample_idxs).astype(np.uint32), batch_variant_idxs.astype(np.uint32)

    def get_covars(self) -> DataBatch:
        Z = self.phenotype.get_covars()
        y = self.phenotype.y
        residuals = self.phenotype.residuals

        return DataBatch(
            self.cfg,
            Z,
            self.genotype.annotation_df,
            y,
            self.phenotype.annotation_df,
            residuals,
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
                weight = total_samples / (2 * count)
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
