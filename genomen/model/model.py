import logging
from pathlib import Path
from typing import Literal, Tuple
import pickle

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

import wandb

from .. import global_run_manager, utils
from . import utils as model_utils
from ..data import DataSet
from .configs import GenomenModelConfig, TrainConfig
from .covar import CovarEstimator
from .geno import GenoEstimator


class GenomenModel:
    """"""

    def __init__(self, cfg: GenomenModelConfig | None = None) -> None:
        self.cfg = cfg
        self._logger = logging.getLogger(__name__)
        self.covar_model: CovarEstimator | None = None
        self.geno_model: GenoEstimator | None = None

    def _setup_run(self, train_data: DataSet, mode: Literal["geno", "covar+geno"]):
        geno_model_str = f"GENO{self.cfg.geno_config.model_config.model_name}_"
        covar_model_str = (
            f"COVAR{self.cfg.covar_config.model_config.model_name}_" if mode == "covar+geno" else ""
        )

        run_dir = (
            f"GenomEn_MODE{mode}_"
            + covar_model_str
            + geno_model_str
            + f"{'/'.join(train_data.cfg.populations)}_"
            + f"PHENO{train_data.cfg.phenotype_id}_"
            + f"STRAT{train_data.cfg.variant_sampling.strat}_"
            + f"SAMP{train_data.cfg.sample_sampling.max_samples or len(train_data)}_"
            + f"FEAT{train_data.cfg.variant_sampling.max_features}"
        )
        if global_run_manager.run_path is not None:
            global_run_manager.update_run_dir(run_dir)
        else:
            global_run_manager.init_path(run_dir)

        if self.train_cfg.save_model:
            model_dir = global_run_manager.get_path("model")
            self.cfg.model_dir = model_dir

        if self.train_cfg.log_with_wandb:
            geno_model_str = f"GENO{self.cfg.geno_config.model_config.model_name}_"
            covar_model_str = (
                f"COVAR{self.cfg.covar_config.model_config.model_name}_"
                if mode == "covar+geno"
                else ""
            )

            run_name = (
                f"GenomEn_MODE{mode}_"
                + covar_model_str
                + geno_model_str
                + f"{'/'.join(train_data.cfg.populations)}_"
                + f"PHENO{train_data.cfg.phenotype_id}_"
                + f"STRAT{train_data.cfg.variant_sampling.strat}_"
                + f"SAMP{train_data.cfg.sample_sampling.max_samples or len(train_data)}_"
                + f"FEAT{train_data.cfg.variant_sampling.max_features}"
            )
            wandb.init(
                project="MetaPRS",
                name=run_name,
                config={
                    "train_cfg": self.train_cfg.__dict__,
                    "data_cfg": train_data.cfg.__dict__,
                    "model_cfg": self.cfg.__dict__,
                },
            )

    def fit(
        self,
        train_data: DataSet,
        val_data: DataSet,
        train_cfg: TrainConfig | None = None,
    ):
        self.train_cfg = train_cfg or utils.init_class(
            cls=TrainConfig, classification=train_data.cfg.classification
        )
        self.cfg = self.cfg or utils.init_class(
            cls=GenomenModelConfig,
            classification=train_data.cfg.classification,
            include_covars=train_data.cfg.covar_config.include_covars,
            save_model=self.train_cfg.save_model,
            max_features=train_data.cfg.variant_sampling.max_features,
            backend=self.train_cfg.backend,
            ram_mb=self.train_cfg.ram_mb,
        )

        self._logger.info(
            f"Interaction value computation is set: {self.cfg.geno_config.compute_interactions}"
        )
        if self.cfg.include_covars:
            self._logger.info("Fitting covar model...")
            self._setup_run(train_data, "covar+geno")
            self.covar_model = CovarEstimator(
                self.cfg.covar_config.model_config, self.cfg.eps, self.train_cfg.n_jobs
            )
            # Propagate covar_pred (as logit offset) when any estimator uses use_offset.
            # residualize and use_offset are now independent flags in CovarEstimator.
            use_offset = self.cfg.geno_config.model_config.use_offset
            # When use_offset is active, skip standardization so both linear (residuals)
            # and LGBM (offset) operate in the same raw value / logit space.
            standardize = not use_offset
            # get residuals / covar_pred for train data
            self._logger.info("Fitting covar model on train data...")
            train_data, _ = self.covar_model.cross_val_predict(
                train_data,
                refit=True,
                residualize=self.cfg.geno_config.use_resids,
                use_offset=use_offset,
                standardize=standardize,
            )  # predict on oof + refit
            # get residuals / covar_pred for val data
            if val_data:
                self._logger.info("Applying covar model to val data...")
                val_data, val_covar_preds = self.covar_model.predict(
                    val_data,
                    residualize=self.cfg.geno_config.use_resids,
                    use_offset=use_offset,
                    standardize=standardize,
                )
            if val_data:
                val_covar_score = utils.score(
                    val_data.get_labels(),
                    val_covar_preds,
                    self.train_cfg.scorer,
                    val_data.cfg.classification,
                )
                self._logger.info(f"Validation covar-only score: {val_covar_score:.4f}")
                if self.train_cfg.log_with_wandb:
                    wandb.log({"covar_val_score": val_covar_score})
        else:
            self._setup_run(train_data, "geno")

        self._logger.info("Setting up train data (MAF, scoring, sampling)...")

        # Snapshot MAF before setup() so we can verify it is unchanged afterwards.
        # Compare these logs against [MAF-CHECK SIM] emitted by simulate_data.py.
        _splits_to_check = [("train", train_data), ("val", val_data)]
        _pre_setup_maf: dict[str, "pd.Series"] = {}
        for _split_name, _ds in _splits_to_check:
            if _ds is not None and "MAF" in _ds.genotype.annotation_df.columns:
                _pre_setup_maf[_split_name] = _ds.genotype.annotation_df["MAF"].copy()
                _maf = _pre_setup_maf[_split_name].values
                _vidxs = _ds.genotype.annotation_df.index.values
                self._logger.info(
                    "[MAF-CHECK MODEL pre-setup %s] n_variants=%d, MAF mean=%.5f std=%.5f "
                    "min=%.5f max=%.5f | first5 global_idx=%s MAF=%s",
                    _split_name,
                    len(_maf),
                    _maf.mean(),
                    _maf.std(),
                    _maf.min(),
                    _maf.max(),
                    _vidxs[:5].tolist(),
                    _maf[:5].round(5).tolist(),
                )

        train_data.setup()
        self._train_annotation_df = train_data.genotype.annotation_df
        if val_data is not None:
            val_data.genotype.annotation_df["MAF"] = self._train_annotation_df["MAF"]
            if "A0_FREQ" in self._train_annotation_df.columns:
                val_data.genotype.annotation_df["A0_FREQ"] = self._train_annotation_df["A0_FREQ"]

        # Exact per-variant MAF comparison: pre-setup vs post-setup for each split.
        for _split_name, _ds in _splits_to_check:
            if _ds is None or _split_name not in _pre_setup_maf:
                continue
            _maf_before = _pre_setup_maf[_split_name]
            _maf_after = _ds.genotype.annotation_df["MAF"]
            _shared = _maf_before.index.intersection(_maf_after.index)
            _diff = (_maf_before[_shared] - _maf_after[_shared]).abs()
            _n_mismatch = int((_diff > 1e-6).sum())
            self._logger.info(
                "[MAF-CHECK MODEL post-setup %s] pre_n=%d post_n=%d shared=%d "
                "max_diff=%.2e n_mismatches(>1e-6)=%d",
                _split_name,
                len(_maf_before),
                len(_maf_after),
                len(_shared),
                float(_diff.max()) if len(_diff) else 0.0,
                _n_mismatch,
            )
            if _n_mismatch > 0:
                _bad = _diff[_diff > 1e-6]
                self._logger.warning(
                    "[MAF-CHECK MODEL %s] %d variant(s) changed MAF after setup(). "
                    "First mismatches:\n%s",
                    _split_name,
                    _n_mismatch,
                    _bad.head(10).to_string(),
                )

        self._logger.info("Fitting geno model...")
        # Propagate compute_shap from the data config down to the geno model.
        self.train_cfg.compute_shap = self.train_cfg.compute_shap or train_data.cfg.compute_shap
        self.geno_model = GenoEstimator(self.cfg.geno_config)
        self.geno_model.fit(train_data, val_data, train_cfg=self.train_cfg)
        if val_data is not None:
            self._logger.info("Predicting geno-only on val set...")
            val_geno_preds, val_geno_resid_preds, val_per_type_geno_preds = self.geno_model.predict(
                val_data, return_resids=self.cfg.geno_config.use_resids
            )

            val_geno_score = utils.score(
                val_data.get_labels(),
                val_geno_preds,
                self.train_cfg.scorer,
                val_data.cfg.classification,
            )
            if self.train_cfg.log_with_wandb:
                self._logger.info(f"Geno-only score on val: {val_geno_score:.4f}")
                wandb.log({"geno_val_score": val_geno_score})
            if val_per_type_geno_preds is not None:
                for type_name, type_preds in val_per_type_geno_preds.items():
                    type_score = utils.score(
                        val_data.get_labels(),
                        type_preds,
                        self.train_cfg.scorer,
                        val_data.cfg.classification,
                    )
                    self._logger.info(f"Geno-only val score [{type_name}]: {type_score:.4f}")
                    if self.train_cfg.log_with_wandb:
                        wandb.log({f"geno_val_score_{type_name}": type_score})

        # fuse covar and geno (preferably on val)
        if self.cfg.include_covars and val_data is not None:
            fused_val_preds = self._fuse_predict(
                val_covar_preds,
                val_geno_preds,
                val_geno_resid_preds,
                res_transformer=val_data.phenotype.residual_transformer,
            )
            fused_val_score = utils.score(
                val_data.get_labels(),
                fused_val_preds,
                self.train_cfg.scorer,
                val_data.cfg.classification,
            )
            if self.train_cfg.log_with_wandb:
                wandb.log({"covar_geno_val_score": fused_val_score})

        # safe geno annotation file if needed
        if self.train_cfg.save_annotation:
            annotation_dir = global_run_manager.get_path("annotation")
            self.geno_model.save_annotation_file(annotation_dir)

        # save model if needed
        if self.train_cfg.save_model:
            model_path = Path(self.cfg.model_dir, "genomen_model.pkl")
            joblib.dump(self, model_path)
            self._logger.info(f"Saved strong estimator to: {model_path}")

    def _fuse_predict(
        self,
        covar_preds: npt.NDArray,
        geno_preds: npt.NDArray | None = None,
        geno_resid_preds: npt.NDArray | None = None,
        res_transformer: DataSet | None = None,
    ) -> npt.NDArray:
        if self.cfg.covar_config.covar_strat == "predictive":
            correction = geno_preds
        else:
            if res_transformer is not None:
                # Standardized residuals — apply inverse transform back to value space
                correction = model_utils.safe_inv_transform(res_transformer, geno_resid_preds)
            else:
                # Raw residuals (use_offset=True case) — already in value space, no inverse transform needed
                correction = geno_resid_preds

        fused = covar_preds + correction
        if self.cfg.classification:
            fused = np.clip(fused, self.cfg.eps, 1 - self.cfg.eps)
        return fused

    def predict(
        self, data: DataSet, *, compute_shap: bool = False
    ) -> (
        Tuple[npt.NDArray, npt.NDArray | None, npt.NDArray | None]
        | Tuple[npt.NDArray, npt.NDArray | None, npt.NDArray | None, "pd.DataFrame | None"]
    ):
        if "MAF" not in data.genotype.annotation_df.columns and hasattr(
            self, "_train_annotation_df"
        ):
            data.genotype.annotation_df["MAF"] = self._train_annotation_df["MAF"]
        if (
            "A0_FREQ" not in data.genotype.annotation_df.columns
            and hasattr(self, "_train_annotation_df")
            and "A0_FREQ" in self._train_annotation_df.columns
        ):
            data.genotype.annotation_df["A0_FREQ"] = self._train_annotation_df["A0_FREQ"]
        if self.cfg.include_covars:
            # fit covars
            _use_offset = self.cfg.geno_config.model_config.use_offset
            data, covar_preds = self.covar_model.predict(
                data,
                residualize=self.cfg.geno_config.use_resids,
                use_offset=_use_offset,
                standardize=not _use_offset,
            )
            # fit geno
            geno_preds, geno_resid_preds, per_type_geno_preds = self.geno_model.predict(
                data, return_resids=self.cfg.geno_config.use_resids
            )
            self.per_type_geno_preds = per_type_geno_preds
            # fuse covar and geno
            fused_preds = self._fuse_predict(
                covar_preds,
                geno_preds,
                geno_resid_preds,
                res_transformer=data.phenotype.residual_transformer,
            )

            if compute_shap:
                shap_df = self.geno_model.compute_local_shap(data)
                return geno_preds, covar_preds, fused_preds, shap_df
            return geno_preds, covar_preds, fused_preds
        else:
            # fit geno
            geno_preds, _, per_type_geno_preds = self.geno_model.predict(data)
            self.per_type_geno_preds = per_type_geno_preds

            if compute_shap:
                shap_df = self.geno_model.compute_local_shap(data)
                return geno_preds, None, None, shap_df
            return geno_preds, None, None

    def compute_local_shap(self, data: DataSet) -> npt.NDArray:
        return self.geno_model.compute_local_shap(data)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._logger.info(f"Saved model to: {path}")

    @classmethod
    def load(cls, path: str) -> "GenomenModel":
        with open(path, "rb") as f:
            return pickle.load(f)
