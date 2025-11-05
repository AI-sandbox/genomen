import logging
from functools import partial
from typing import Callable, List, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ... import utils
from .. import utils as model_utils
from ..configs import AggregatorConfig
from ..weak_estimator import WeakEstimator


class GenoAggregator:
    def __init__(
        self,
        classification: bool,
        cfg: AggregatorConfig | None = None,
        scorer: Literal["rocauc", "r2"] | None = None,
    ):
        self.cfg = cfg or utils.init_class(cls=AggregatorConfig)
        self.cfg.classification = classification
        self.scorer = scorer
        self._logger = logging.getLogger(__name__)

        self.agg_fn: Callable | None = None
        self.agg_scores: npt.NDArray | None = None
        self.filter_mask: npt.NDArray | None = None
        self.agg_weights: npt.NDArray = np.array([], dtype=float)
        self.eps = 1e-7

    def _filter(self):
        if (self.cfg.filter_strat != "none") and (self.agg_scores is None):
            raise ValueError("Have to provide metrics on validation set to use filter!")

        match self.cfg.filter_strat:
            case "none":
                filter_mask = np.full(len(self.agg_scores), True, dtype=bool)
            case "positive":
                if self.scorer is None:
                    raise ValueError("Need to specify scorer for 'positive' filter!")
                if self.scorer == "rocauc":
                    threshold = 0.5
                elif self.scorer == "r2":
                    threshold = 0.0
                else:
                    raise ValueError(f"Scoer {self.scorer} not supported for 'positive' filter!")
                filter_mask = self.agg_scores >= threshold
            case "geq-average":
                threshold = np.mean(self.agg_scores)
                filter_mask = self.agg_scores >= threshold
            case "top-p-percentile":
                threshold = np.percentile(self.agg_scores, self.cfg.p)
                filter_mask = self.agg_scores >= threshold

        if not np.any(filter_mask):
            self._logger.warning("No models passed the filtering criteria. Skipping filtering...")
            filter_mask = np.full(len(self.agg_scores), True, dtype=bool)

        self.filter_mask = filter_mask

    def _update_loss_weights(self):
        """Update the weights based on current validation results."""
        self.agg_weights = np.zeros(len(self.agg_scores))

        # Get filtered results
        filtered_agg_scores = self.agg_scores[self.filter_mask]

        # calculate weights via softmax (safe implementation)
        self.agg_weights[self.filter_mask] = utils.safe_softmax(filtered_agg_scores, self.cfg.temp)

    def _to2D(self, values: List[npt.NDArray] | npt.NDArray) -> npt.NDArray:
        X = np.asarray(values)  # (n_est, n_values)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # (n_est, 1)

        return X

    def _identity(self, values: List[npt.NDArray] | npt.NDArray, **kwargs):
        """Identity aggregation - returns the first (and presumably only) value."""
        if isinstance(values, list):
            if len(values) != 1:
                raise ValueError(
                    f"Identity aggregation expects exactly 1 estimator, got {len(values)}"
                )
            return values[0]
        else:
            # For 2D array input, return the first row
            if values.shape[0] != 1:
                raise ValueError(
                    f"Identity aggregation expects exactly 1 estimator, got {values.shape[0]}"
                )
            return values[0]

    def _simple_mean(
        self,
        values: List[npt.NDArray] | npt.NDArray,
        filter_mask: npt.NDArray | None = None,
        **kwargs,
    ):
        X = self._to2D(values)

        if filter_mask is not None:
            X = X[filter_mask]

        return model_utils.safe_nanmean(X, axis=0, fill_value=0.0)

    def _rank_mean(
        self,
        values: List[npt.NDArray] | npt.NDArray,
        filter_mask: npt.NDArray | None = None,
        aggregate_coeffs: bool = False,
    ):
        X = self._to2D(values)

        if filter_mask is not None:
            X = X[filter_mask]

        if aggregate_coeffs:  # not supported
            raise ValueError(
                "Cannot aggregate coefficients with rank mean! Use different aggregation strategy."
            )

        ranks = pd.DataFrame(X).rank(axis=0, method="average").to_numpy()  # (n_est, n_values)
        return ranks.mean(axis=0)  # (n_values,)

    def _loss_weighted_avg(
        self,
        values: List[npt.NDArray] | npt.NDArray,
        weights: npt.NDArray,
        filter_mask: npt.NDArray | None = None,
        **kwargs,
    ):
        X = self._to2D(values)

        if filter_mask is not None:
            w = weights[filter_mask]
            X = X[filter_mask]
        else:
            w = weights

        nan_mask = ~np.isnan(X)
        X_imputed = np.nan_to_num(X, nan=0.0)
        num = (X_imputed * w[:, None]).sum(axis=0)
        denom = (nan_mask * w[:, None]).sum(axis=0)
        denom = np.where(denom == 0.0, 1.0, denom)

        return num / denom  # (n_values,)

    def _stack(
        self,
        values: List[npt.NDArray] | npt.NDArray,
        stacking_model: WeakEstimator,
        filter_mask: npt.NDArray | None = None,
        aggregate_coeffs: bool = False,
    ):
        X = self._to2D(values)
        X = np.nan_to_num(X, nan=0.0)

        if aggregate_coeffs:
            coeffs = X[filter_mask] if filter_mask is not None else X
            w = stacking_model.model.coef_.ravel()  # (n_est,)
            w = w[: coeffs.shape[0]]

            return np.tensordot(coeffs.T, w, axes=(1, 0))

        if filter_mask is not None:
            X = X[filter_mask]
        if self.cfg.classification:
            X = utils.get_logits(X, self.eps)

        return stacking_model.predict(X.T)  # (n_values,)

    def calibrate(
        self,
        agg_scores: List[float] | float | None = None,
        agg_preds: List[npt.NDArray] | None = None,
        agg_labels: npt.NDArray | None = None,
        agg_sample_weight: npt.NDArray | None = None,
    ):
        if isinstance(agg_scores, (float, int, np.number)):
            agg_scores = [float(agg_scores)]

        self.agg_scores = np.array(agg_scores, dtype=float)

        # Update filter mask based on current validation results
        if self.cfg.filter_strat != "none":
            self._filter()

        if self.cfg.agg_strat == "loss-weighted-average":
            self._update_loss_weights()

        match self.cfg.agg_strat:
            case "mean":
                self._aggregation_fn = partial(self._simple_mean, filter_mask=self.filter_mask)
            case "rank-mean":
                if not self.cfg.classification:
                    raise ValueError(
                        f"Aggregation strat {self.cfg.agg_strat} is not available for regression data!"
                    )

                self._aggregation_fn = partial(self._rank_mean, filter_mask=self.filter_mask)
            case "loss-weighted-average":
                self._aggregation_fn = partial(
                    self._loss_weighted_avg,
                    weights=self.agg_weights,
                    filter_mask=self.filter_mask,
                )
            case "stacking":
                if (agg_preds is None) or (agg_labels is None):
                    raise ValueError(
                        "Aggregation.agg_strat 'stacking' requires aggregation predictions and labels as input to calibration"
                    )

                X = np.column_stack(agg_preds)
                y = agg_labels

                # apply filter
                if self.filter_mask is not None:
                    X = X[:, self.filter_mask]

                # transform classification to logit space
                if self.cfg.classification:
                    X = utils.get_logits(X, self.eps)

                self.stacking_model = WeakEstimator(self.cfg.model_config)
                self.stacking_model.fit(X, y, sample_weight=agg_sample_weight)

                self._aggregation_fn = partial(
                    self._stack,
                    stacking_model=self.stacking_model,
                    filter_mask=self.filter_mask,
                )

            case _:
                raise ValueError(f"Invalid aggregation strat {self.cfg.agg_strat}")

    def __call__(self, predictions: List[npt.NDArray]) -> npt.NDArray:
        """Apply the aggregation function to predictions.

        Args:
            predictions: List of predictions from each estimator

        Returns:
            Aggregated prediction
        """
        if self._aggregation_fn is None:
            raise ValueError(
                "Aggregation function is None. It seems like the aggregator has not been calibrated yet."
            )

        return self._aggregation_fn(predictions)

    def get_aggregation_fn(self) -> Callable:
        """Returns the initialized aggregation function."""
        if self._aggregation_fn is None:
            raise ValueError(
                "Aggregation function is None. It seems like, the aggregator has not been called yet."
            )

        return self._aggregation_fn

    def aggregate_group(
        self,
        group: pd.DataFrame,
        n_estimators: int,
        columns: List[str],
        aggregate_coeffs: bool = False,
        idx_col: str = "estimator_idx",
    ) -> pd.Series:
        est_idx = group[idx_col].to_numpy(dtype=int)
        X_sparse = group[columns].to_numpy(dtype=float)  # (n_col_est, n_values)
        X = np.full((n_estimators, X_sparse.shape[1]), np.nan)  # (n_est, n_values)
        X[est_idx, :] = X_sparse  # (n_est, n_values)

        agg = self._aggregation_fn(X, aggregate_coeffs=aggregate_coeffs)  # (n_values)
        return pd.Series(agg, index=columns)

    def reset(self):
        """Reset aggregator to identity function for cases where coefficients are already aggregated."""
        self._aggregation_fn = self._identity
        self.filter_mask = None
        self.agg_weights = np.array([1.0])
        self.agg_scores = np.array([1.0])
