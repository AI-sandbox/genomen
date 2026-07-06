import logging
from functools import partial
from typing import Callable, List, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from ... import utils
from .. import utils as model_utils
from ..configs import AggregatorConfig


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

    def _filter(self, group_size: int = 1):
        if (self.cfg.filter_strat != "none") and (self.agg_scores is None):
            raise ValueError("Have to provide metrics on validation set to use filter!")

        # For ensemble models, filter at the group (pair) level so that all members
        # of a logical estimator are kept or discarded together.
        if group_size > 1:
            n_total = len(self.agg_scores)
            n_groups = n_total // group_size
            group_scores = (
                self.agg_scores[: n_groups * group_size].reshape(n_groups, group_size).mean(axis=1)
            )
        else:
            group_scores = self.agg_scores

        match self.cfg.filter_strat:
            case "none":
                group_mask = np.full(len(group_scores), True, dtype=bool)
            case "positive":
                threshold = self.cfg.pos_thresh
                group_mask = group_scores >= threshold
            case "geq-average":
                threshold = np.mean(group_scores)
                group_mask = group_scores >= threshold
            case "top-p-percentile":
                threshold = np.percentile(group_scores, self.cfg.p * 100)
                group_mask = group_scores >= threshold

        if not np.any(group_mask):
            self._logger.warning("No models passed the filtering criteria. Skipping filtering...")
            group_mask = np.full(len(group_scores), True, dtype=bool)

        self.filter_mask = np.repeat(group_mask, group_size) if group_size > 1 else group_mask

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
        aggregate_coeffs: bool = False,
        **kwargs,
    ):
        X = self._to2D(values)

        if filter_mask is not None:
            X = X[filter_mask]

        if aggregate_coeffs:
            # A NaN means this estimator never had this variant — i.e. zero
            # contribution to its own prediction, not missing data to exclude
            # from the average. Zero-fill and divide by the full (filtered)
            # estimator count, so folding coefficients reproduces exactly what
            # averaging each estimator's own complete prediction would give
            # (same zero-fill convention: NaN = zero contribution, not missing).
            return np.nan_to_num(X, nan=0.0).mean(axis=0)

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
        aggregate_coeffs: bool = False,
        **kwargs,
    ):
        X = self._to2D(values)

        if filter_mask is not None:
            w = weights[filter_mask]
            X = X[filter_mask]
        else:
            w = weights

        X_imputed = np.nan_to_num(X, nan=0.0)
        num = (X_imputed * w[:, None]).sum(axis=0)

        if aggregate_coeffs:
            # Same zero-fill convention as _simple_mean: a NaN is zero
            # contribution, not missing data, so divide by the full weight
            # total rather than only the weight of estimators with this variant.
            denom = w.sum()
            denom = denom if denom != 0.0 else 1.0
        else:
            nan_mask = ~np.isnan(X)
            denom = (nan_mask * w[:, None]).sum(axis=0)
            denom = np.where(denom == 0.0, 1.0, denom)

        return num / denom  # (n_values,)

    def calibrate(
        self,
        agg_scores: List[float] | float | None = None,
        agg_preds: List[npt.NDArray] | None = None,
        agg_labels: npt.NDArray | None = None,
        agg_sample_weight: npt.NDArray | None = None,
        group_size: int = 1,
    ):
        if isinstance(agg_scores, (float, int, np.number)):
            agg_scores = [float(agg_scores)]

        self.agg_scores = np.array(agg_scores, dtype=float)

        # Update filter mask based on current validation results
        if self.cfg.filter_strat != "none":
            self._filter(group_size=group_size)

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
        filter_mask: npt.NDArray | None = None,
    ) -> pd.Series:
        est_idx = group[idx_col].to_numpy(dtype=int)
        X_sparse = group[columns].to_numpy(dtype=float)  # (n_col_est, n_values)
        X = np.full((n_estimators, X_sparse.shape[1]), np.nan)  # (n_est, n_values)
        X[est_idx, :] = X_sparse  # (n_est, n_values)

        if filter_mask is not None:
            # Caller supplies a pre-projected mask (AND of global filter + group membership).
            # Unwrap the partial to replace the baked-in global mask with this local one.
            fn = self._aggregation_fn
            if isinstance(fn, partial):
                base_fn = fn.func
                extra_kwargs = {k: v for k, v in fn.keywords.items() if k != "filter_mask"}
                agg = base_fn(
                    X, aggregate_coeffs=aggregate_coeffs, filter_mask=filter_mask, **extra_kwargs
                )
            else:
                agg = fn(X, aggregate_coeffs=aggregate_coeffs, filter_mask=filter_mask)
        else:
            agg = self._aggregation_fn(X, aggregate_coeffs=aggregate_coeffs)  # (n_values)
        return pd.Series(agg, index=columns)

    def reset(self):
        """Reset aggregator to identity function for cases where coefficients are already aggregated."""
        self._aggregation_fn = self._identity
        self.filter_mask = None
        self.agg_weights = np.array([1.0])
        self.agg_scores = np.array([1.0])
