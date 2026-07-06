import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_inv_transform_plot_count = 0
_INV_TRANSFORM_MAX_PLOTS = 3


def yeo_johnson_domain_space(transformer, feature_idx=0, eps=1e-8):
    lam = transformer.lambdas_[feature_idx]

    # Domain in raw Yeo-Johnson transformed space
    lo, hi = -np.inf, np.inf

    # Positive branch creates an upper bound when lambda < 0
    if lam < 0:
        hi = -1.0 / lam - eps

    # Negative branch creates a lower bound when lambda > 2
    if lam > 2:
        lo = 1.0 / (2.0 - lam) + eps  # note: 2-lam < 0, so this is negative

    # If standardization was applied, convert bounds to standardized space
    if getattr(transformer, "standardize", False):
        mean = transformer._scaler.mean_[feature_idx]
        scale = transformer._scaler.scale_[feature_idx]
        if np.isfinite(lo):
            lo = (lo - mean) / scale
        if np.isfinite(hi):
            hi = (hi - mean) / scale

    return lo, hi


class PearsonResidualTransformer:
    """Computes Pearson residuals (y - p) / sqrt(p*(1-p)) with exact inverse.

    fit_transform divides by per-sample scale and stores it for inverse_transform.
    transform accepts optional preds for exact per-sample scaling at inference.
    """

    eps: float = 1e-7

    def __init__(self):
        self.scale_: npt.NDArray = np.array([1.0])

    def fit_transform(self, resids: npt.NDArray, preds: npt.NDArray) -> npt.NDArray:
        self.scale_ = np.sqrt(preds * (1 - preds)) + self.eps
        return resids / self.scale_

    def transform(self, resids: npt.NDArray) -> npt.NDArray:
        assert (
            resids.shape == self.scale_.shape
        ), f"Shape mismatch: resids.shape={resids.shape}, scale_.shape={self.scale_.shape}"
        return resids / self.scale_

    def inverse_transform(self, resids: npt.NDArray) -> npt.NDArray:
        assert (
            resids.shape == self.scale_.shape
        ), f"Shape mismatch: resids.shape={resids.shape}, scale_.shape={self.scale_.shape}"
        return resids * self.scale_


def safe_inv_transform(transformer, x):
    global _inv_transform_plot_count

    if not hasattr(transformer, "lambdas_"):
        # Not a PowerTransformer (e.g. PearsonResidualTransformer) — invert directly
        return transformer.inverse_transform(x)

    lo, hi = yeo_johnson_domain_space(transformer, feature_idx=0)
    x_clipped = np.clip(x, lo, hi)

    n_clipped = np.sum(x != x_clipped)
    if n_clipped > 0:
        logger.warning(
            f"Clipped {n_clipped}/{x.size} inverse-transform inputs to valid "
            f"Yeo-Johnson domain: [{lo:.6f}, {hi:.6f}]"
        )

    out = transformer.inverse_transform(x_clipped.reshape(-1, 1)).reshape(-1)

    nan_mask = np.isnan(out)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:  # impute with median
        finite_vals = out[~nan_mask]
        fill_value = np.nanmedian(finite_vals) if finite_vals.size > 0 else 0.0

        logger.warning(
            f"{nan_count} NaNs ({out.size} total) detected after inverse_transform. Imputing with median {fill_value}"
        )

        if _inv_transform_plot_count < _INV_TRANSFORM_MAX_PLOTS:
            _plot_inv_transform_distributions(x, out, nan_mask, _inv_transform_plot_count)
            _inv_transform_plot_count += 1

        out[nan_mask] = fill_value

    return out


def _plot_inv_transform_distributions(x, out, nan_mask, plot_idx):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"inverse_transform diagnostics (call #{plot_idx + 1}), NaNs={nan_mask.sum()}/{nan_mask.size}"
        )

        # Input distribution
        ax = axes[0]
        ax.hist(x, bins=100, color="steelblue", edgecolor="none")
        nan_inputs = x[nan_mask]
        if len(nan_inputs) > 0:
            ax.axvline(
                nan_inputs.min(),
                color="red",
                linestyle="--",
                label=f"NaN input min={nan_inputs.min():.2f}",
            )
            ax.axvline(
                nan_inputs.max(),
                color="orange",
                linestyle="--",
                label=f"NaN input max={nan_inputs.max():.2f}",
            )
            ax.legend(fontsize=7)
        ax.set_title("Input (transformed residuals)")
        ax.set_xlabel("value")
        ax.set_ylabel("count")

        # Output distribution (finite only)
        ax = axes[1]
        finite_out = out[~nan_mask]
        if len(finite_out) > 0:
            ax.hist(finite_out, bins=100, color="steelblue", edgecolor="none")
        ax.set_title("Output (finite only, logit space)")
        ax.set_xlabel("value")
        ax.set_ylabel("count")

        # Input: NaN-producing vs finite side by side
        ax = axes[2]
        finite_inputs = x[~nan_mask]
        ax.hist(
            finite_inputs,
            bins=80,
            alpha=0.6,
            color="steelblue",
            label="finite output",
            density=True,
        )
        if len(nan_inputs) > 0:
            ax.hist(nan_inputs, bins=80, alpha=0.6, color="red", label="NaN output", density=True)
        ax.set_title("Input split by NaN outcome")
        ax.set_xlabel("value")
        ax.legend(fontsize=8)

        fig.tight_layout()
        path = f"inv_transform_debug_{plot_idx}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info(f"Saved inverse_transform debug plot to {path}")
    except Exception as e:
        logger.warning(f"Failed to generate inverse_transform debug plot: {e}")


def safe_nanmean(X, axis=0, fill_value=0.0):
    X = np.asarray(X)
    nan_mask = ~np.isnan(X)
    num = np.nansum(X, axis=axis)
    denom = nan_mask.sum(axis=axis)

    out = np.divide(num, np.where(denom == 0, 1, denom), dtype=float)
    if np.isscalar(out):
        return out if denom != 0 else fill_value
    out = np.asarray(out)
    out[denom == 0] = fill_value
    return out
