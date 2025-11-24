import logging
import os
import pickle
from pathlib import Path
from typing import Literal, Tuple

import joblib
import numpy as np
import numpy.typing as npt
import statsmodels.api as sm
import wandb

from .. import global_run_manager, utils
from ..data import DataSet
from . import utils as model_utils
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
        if self.train_cfg.save_model:  # setup model dir
            geno_model_str = f"GENO{self.cfg.geno_config.model_config.model_name}_"
            covar_model_str = (
                f"COVAR{self.cfg.covar_config.model_config.model_name}_"
                if mode == "covar+geno"
                else ""
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
            model_dir = global_run_manager.get_path("model")
            annotation_dir = global_run_manager.get_path("annotations")
            vi_dir = global_run_manager.get_path("variant_importance")

            self.cfg.model_dir = model_dir
            self.cfg.annotation_dir = annotation_dir
            self.cfg.variant_importance_dir = vi_dir

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
                project=os.environ.get("WANDB_PROJECT"),
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

        if self.cfg.include_covars:
            self._logger.info("Fitting covar model...")
            self._setup_run(train_data, "covar+geno")
            self.covar_model = CovarEstimator(
                self.cfg.covar_config.model_config, self.cfg.eps, self.train_cfg.n_jobs
            )
            # get residuals for train data
            train_data, train_covar_preds = self.covar_model.cross_val_predict(
                train_data, refit=True, residualize=self.cfg.geno_config.use_resids
            )  # predict on oof + refit
            # get residuals for val data
            if val_data:
                val_data, val_covar_preds = self.covar_model.predict(
                    val_data, residualize=self.cfg.geno_config.use_resids
                )

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

        self._logger.info("Fitting geno model...")
        self.geno_model = GenoEstimator(self.cfg.geno_config)
        self.geno_model.fit(train_data, val_data)
        if val_data is not None:
            val_geno_preds, val_geno_resid_preds = self.geno_model.predict(
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
        else:
            train_geno_preds, train_geno_resid_preds = self.geno_model.predict(
                train_data, return_resids=self.cfg.geno_config.use_resids
            )

        # fuse covar and geno (preferably on val)
        if self.cfg.include_covars:
            covar_fuse_preds = val_covar_preds if val_data is not None else train_covar_preds
            if self.cfg.covar_config.covar_strat == "predictive":
                geno_fuse_preds = val_geno_preds if val_data is not None else train_geno_preds
            else:
                geno_fuse_preds = (
                    val_geno_resid_preds if val_data is not None else train_geno_resid_preds
                )

            if self.cfg.classification:
                covar_fuse_preds = utils.get_logits(covar_fuse_preds, self.cfg.eps)
                if self.cfg.covar_config.covar_strat == "predictive":
                    geno_fuse_preds = utils.get_logits(geno_fuse_preds, self.cfg.eps)
                else:
                    geno_fuse_preds = model_utils.safe_inv_transform(
                        self.covar_model.residual_transformer, geno_fuse_preds
                    )

            # Stack covar and geno predictions as features
            X_fuse = np.column_stack(
                [covar_fuse_preds.reshape(-1, 1), geno_fuse_preds.reshape(-1, 1)]
            )
            X_fuse = sm.add_constant(X_fuse, has_constant="add")
            y = val_data.get_labels() if val_data is not None else train_data.get_labels()

            if self.cfg.classification:
                family = sm.families.Binomial()
            else:
                family = sm.families.Gaussian(sm.families.links.identity())

            self.fusion_model = sm.GLM(y, X_fuse, family=family).fit()

            if val_data is not None:
                fused_val_preds = self._fuse_predict(
                    val_covar_preds, val_geno_preds, val_geno_resid_preds
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
            self.geno_model.save_annotation_file(self.cfg.annotation_dir)

        # save interactions if needed
        self.geno_model.save_interactions(self.cfg.variant_importance_dir)

        # save model if needed
        if self.train_cfg.save_model:
            self.save()

    def _fuse_predict(
        self,
        covar_preds: npt.NDArray,
        geno_preds: npt.NDArray | None = None,
        geno_resid_preds: npt.NDArray | None = None,
    ) -> npt.NDArray:
        geno_preds = (
            geno_preds if self.cfg.covar_config.covar_strat == "predictive" else geno_resid_preds
        )

        if self.cfg.classification:
            covar_preds = utils.get_logits(covar_preds, self.cfg.eps)
            if self.cfg.covar_config.covar_strat == "predictive":
                geno_preds = utils.get_logits(geno_preds, self.cfg.eps)
            else:
                geno_preds = model_utils.safe_inv_transform(
                    self.covar_model.residual_transformer, geno_preds
                )

        X_fuse = np.column_stack([covar_preds.reshape(-1, 1), geno_preds.reshape(-1, 1)])
        X_fuse = sm.add_constant(X_fuse, has_constant="add")

        fused_preds = self.fusion_model.predict(X_fuse)

        return fused_preds

    def predict(self, data: DataSet) -> Tuple[npt.NDArray, npt.NDArray | None, npt.NDArray | None]:
        if self.cfg.include_covars:
            # fit covars
            data, covar_preds = self.covar_model.predict(
                data, residualize=self.cfg.geno_config.use_resids
            )
            # fit geno
            geno_preds, geno_resid_preds = self.geno_model.predict(
                data, return_resids=self.cfg.geno_config.use_resids
            )
            # fuse covar and geno
            fused_preds = self._fuse_predict(covar_preds, geno_preds, geno_resid_preds)

            return geno_preds, covar_preds, fused_preds
        else:
            # fit geno
            geno_preds, _ = self.geno_model.predict(data)

            return geno_preds, None, None

    def save(self, path: str | None = None) -> None:
        if path is None:
            path = Path(self.cfg.model_dir, "genomen_model.pkl")
        joblib.dump(self, path)
        self._logger.info(f"Saved model to: {path}")

    @classmethod
    def load(cls, path: str) -> "GenomenModel":
        with open(path, "rb") as f:
            return pickle.load(f)
