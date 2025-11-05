import logging

import numpy as np

logger = logging.getLogger(__name__)


def safe_inv_transform(transformer, x):
    out = transformer.inverse_transform(x.reshape(-1, 1)).reshape(-1)

    nan_mask = np.isnan(out)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:  # impute with median
        finite_vals = out[~nan_mask]
        fill_value = np.nanmedian(finite_vals) if finite_vals.size > 0 else 0.0

        logger.warning(
            f"{nan_count} NaNs ({out.size} total) detected after inverse_transform. Imputing with median {fill_value}"
        )

        out[nan_mask] = fill_value

    return out

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
