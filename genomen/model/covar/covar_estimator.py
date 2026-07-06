from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from sklearn.preprocessing import PowerTransformer

from .. import utils as model_utils
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
        self.eps = eps
        super().__init__(cfg, model_init_params)
        self.cfg.n_jobs = n_jobs

    def _to_offset(self, preds: npt.NDArray) -> npt.NDArray:
        """Convert covar predictions to offset space.

        For classification: logit(p), matching LGBM init_score and statsmodels GLM offset.
        For regression: predictions are already in value space.
        """
        if self.cfg.classification:
            eps = self.eps
            return np.log(np.clip(preds, eps, 1 - eps) / np.clip(1 - preds, eps, 1 - eps))
        return preds

    def _residualize(self, data: DataSet, preds: npt.NDArray, standardize: bool = True) -> npt.NDArray:
        y = data.get_labels()
        print(f"Got values mean: {preds.mean()}, max: {max(y)}, min: {min(y)} as preds to residualize covariates")
        resid = y - preds
        print(f"Got residuals mean: {resid.mean()}, std: {resid.std()}, max: {max(resid)}, min: {min(resid)} before normalization for task {'cl' if self.cfg.classification else 'reg'}")
        if standardize:
            if self.cfg.classification:
                transformer = model_utils.PearsonResidualTransformer()
                resid = transformer.fit_transform(resid, preds)
            else:
                transformer = PowerTransformer(method="yeo-johnson", standardize=True)
                resid = transformer.fit_transform(resid.reshape(-1, 1)).flatten()
            data.phenotype.residual_transformer = transformer
            print(f"Got residuals mean: {resid.mean()}, std: {resid.std()}, max: {max(resid)}, min: {min(resid)} after normalization for task {'cl' if self.cfg.classification else 'reg'}")
        return resid

    def cross_val_predict(
        self,
        data_set: DataSet,
        cv: int = 5,
        refit: bool = False,
        residualize: bool = False,
        use_offset: bool = False,
        standardize: bool = True,
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
            train_fold_sw = train_fold.get_sample_weights()
            fold_estimator.fit(X_fold, y_fold, sample_weight=train_fold_sw)

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
            covar_data_sw = covar_data.get_sample_weights()
            estimator.fit(covar_data.X, covar_data.get_labels(), sample_weight=covar_data_sw)
            self.model = estimator.model

        if residualize:
            resids = self._residualize(data_set, oof_preds, standardize=standardize)
            data_set.phenotype.residuals = resids
        if use_offset:
            data_set.phenotype.covar_pred = self._to_offset(oof_preds)

        return data_set, oof_preds

    def predict(self, data_set: DataSet, residualize: bool = False, use_offset: bool = False, standardize: bool = True) -> npt.NDArray:
        """Make predictions on a covariate data batch.

        Args:
            data_batch: Input data batch containing covariate data

        Returns:
            Predictions as numpy array
        """
        covar_data = data_set.get_covars()
        preds = super().predict(covar_data.X)

        if residualize:
            resids = self._residualize(data_set, preds, standardize=standardize)
            data_set.phenotype.residuals = resids
        if use_offset:
            data_set.phenotype.covar_pred = self._to_offset(preds)

        return data_set, preds
