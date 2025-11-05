from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from sklearn.preprocessing import PowerTransformer

from ... import utils
from ...data import DataBatch, DataSet, kfold
from ..configs import ModelConfig
from ..weak_estimator import WeakEstimator


class CovarEstimator(WeakEstimator):
    """A weak covariate estimator that handles covariate data."""

    def __init__(
        self,
        cfg: ModelConfig,
        eps: float,
        n_jobs: int,
        model_init_params: Dict[str, Any] | None = None,
    ):
        """Initialize the weak covariate estimator.

        Args:
            cfg: Model configuration
            model_init_params: Optional parameters for model initialization
        """
        self.covar_keys: List[str] | None = None
        self.residual_transformer: PowerTransformer | None = None
        self.eps = eps
        super().__init__(cfg, model_init_params)
        self.cfg.n_jobs = n_jobs

    def _residualize(self, data: DataSet, preds: npt.NDArray) -> npt.NDArray:
        y = data.get_labels()
        if self.cfg.classification:
            """scale = np.sqrt(preds * (1 - preds)) + self.eps
            resid = (y - preds) / scale"""
            preds_logits = utils.get_logits(preds, self.eps)
            y_logits = utils.get_logits(y, self.eps)
            resid = y_logits - preds_logits
            if self.residual_transformer is None:
                self.residual_transformer = PowerTransformer(
                    method="yeo-johnson", standardize=True
                )
                resid = self.residual_transformer.fit_transform(
                    resid.reshape(-1, 1)
                ).flatten()
            else:
                resid = self.residual_transformer.transform(
                    resid.reshape(-1, 1)
                ).flatten()
        else:
            resid = y - preds

        return resid

    def cross_val_predict(
        self,
        data_set: DataSet,
        cv: int = 5,
        refit: bool = False,
        residualize: bool = False,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        self.covar_keys = data_set.cfg.covar_config.covar_keys

        covar_data = data_set.get_covars()
        cv_folds, oof_idxs = kfold(
            covar_data, cv=cv, shuffle=False, return_oof_idxs=True
        )

        def fit_fold(
            model_cfg: ModelConfig, train_fold: DataBatch, test_fold: DataBatch
        ) -> npt.NDArray:
            """Helper function to fit and predict for a single fold."""
            X_fold, y_fold = train_fold.X, train_fold.get_labels()
            fold_estimator = WeakEstimator(cfg=model_cfg)
            fold_estimator.fit(X_fold, y_fold)

            return fold_estimator.predict(test_fold.X)

        oof_preds = np.zeros(len(covar_data.y))
        if self.cfg.n_jobs is not None:
            fold_results = Parallel(n_jobs=self.cfg.n_jobs)(
                delayed(fit_fold)(self.cfg, train_fold, test_fold)
                for train_fold, test_fold in cv_folds
            )
            for fold_preds, fold_oof_idxs in zip(fold_results, oof_idxs):
                oof_preds[fold_oof_idxs] = fold_preds
        else:
            for (train_fold, test_fold), fold_oof_idxs in zip(cv_folds, oof_idxs):
                fold_preds = fit_fold(self.cfg, train_fold, test_fold)
                oof_preds[fold_oof_idxs] = fold_preds

        if refit:
            estimator = WeakEstimator(cfg=self.cfg)
            estimator.fit(covar_data.X, covar_data.get_labels())
            self.model = estimator.model

        if residualize:
            resids = self._residualize(data_set, oof_preds)
            data_set.phenotype.residuals = resids

            return data_set, oof_preds
        return data_set, oof_preds

    def predict(self, data_set: DataSet, residualize: bool = False) -> npt.NDArray:
        """Make predictions on a covariate data batch.

        Args:
            data_batch: Input data batch containing covariate data

        Returns:
            Predictions as numpy array
        """
        covar_data = data_set.get_covars()
        preds = super().predict(covar_data.X)

        if residualize:
            resids = self._residualize(data_set, preds)
            data_set.phenotype.residuals = resids

        return data_set, preds
