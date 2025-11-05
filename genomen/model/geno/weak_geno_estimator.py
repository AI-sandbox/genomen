import logging
from typing import Any, Dict, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import shap

from ...data import DataBatch
from ...data.data_set import utils as data_utils
from ..configs import ModelConfig
from ..weak_estimator import WeakEstimator


class WeakGenoEstimator(WeakEstimator):
    """A weak genotype estimator that handles genetic variant data."""

    def __init__(
        self, cfg: ModelConfig, model_init_params: Dict[str, Any] | None = None
    ):
        """Initialize the weak genotype estimator.

        Args:
            cfg: Model configuration
            model_init_params: Optional parameters for model initialization
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self.annotation_df: pd.DataFrame | None = None
        self.interactions: npt.NDArray | None = None
        super().__init__(cfg, model_init_params)

    @property
    def variant_idxs(self) -> npt.NDArray:
        """Get sample indices from annotation_df as numpy array."""
        return self.annotation_df.index.values

    def fit(
        self,
        train_batch: DataBatch,
        val_batch: DataBatch | None = None,
        use_resids: bool = False,
        sample_weight: npt.NDArray | None = None,
        compute_shap: bool = False,
        compute_interactions: bool = False,
        background: DataBatch | None = None
    ) -> None:
        """Fit the model to training data with genotype-specific handling.

        Args:
            train_batch: Training data batch containing genotype data
            val_batch: Optional validation data batch
            use_resids: Whether to use residualized labels
            sample_weight: Optional sample weights
            compute_shap: Whether to compute SHAP values after fitting
        """
        X_train, y_train = train_batch.X, train_batch.get_labels(use_resids)
        X_val, y_val = (
            (val_batch.X, val_batch.get_labels(use_resids))
            if val_batch is not None
            else (None, None)
        )
        scaler = train_batch.scaler
        self.annotation_df = train_batch.geno_annotation.copy()

        super().fit(X_train, y_train, X_val, y_val, scaler, sample_weight)

        if compute_shap:
            self.compute_shap_values(
                train_batch, 
                use_resids=use_resids, 
                type="global",
                background=background,
            )
            if (self.cfg.model_name in ["lightgbm", "xgboost", "catboost"]) and compute_interactions:
                self.compute_interaction_values(
                    train_batch, 
                    use_resids=use_resids, 
                    background=background,
                )
                
            

    def predict(
        self,
        data_batch: DataBatch
    ) -> npt.NDArray:
        """Make predictions on a genotype data batch with variant validation.

        Args:
            data_batch: Input data batch containing genotype data

        Returns:
            Predictions as numpy array

        Raises:
            ValueError: If variant features don't match training data
        """
        expected_variants = self.variant_idxs.tolist()
        batch_variants = data_batch.geno_annotation.index.tolist()
        if len(expected_variants) != len(batch_variants):
            raise ValueError(
                f"Feature mismatch: Model expects {len(expected_variants)} variants, "
                f"but input batch has {len(batch_variants)}."
            )

        result = super().predict(data_batch.X)

        return result

    def compute_shap_values(
        self, 
        batch: DataBatch,  
        *,
        use_resids: bool = False,
        background: npt.NDArray | None = None,
        type: Literal["local", "global"] = "global",
        n_samples_shap: int = 2_000
    ) -> npt.NDArray | None:
        """Compute SHAP values for the fitted model and store in annotation_df.

        Args:
            X_train: Training data used to create SHAP explainer background
            X_val: Optional validation data for explanation (defaults to train data)
        """
        # Use validation data if available, otherwise use training data
        X_explain, y_explain = batch.X, batch.get_labels(use_resids)

        if X_explain.shape[0] > n_samples_shap and (type == "global"):
            sample_indices = data_utils.adaptive_sampling(
                np.arange(X_explain.shape[0]),
                y_explain,
                self.cfg.classification,
                size=n_samples_shap,
                k=1,
                strategy="balanced"
            )
            X_explain = X_explain[sample_indices]

        # Create appropriate explainer based on model type
        X_bg = background.X if background is not None else X_explain
        if self.cfg.model_type == "linear":
            masker = shap.maskers.Independent(X_bg)
            explainer = shap.LinearExplainer(self.model, masker=masker)
            shap_values = explainer.shap_values(X_explain)
        elif self.cfg.model_name in ["lightgbm", "xgboost", "catboost"]:
            # For tree-based models, use TreeExplainer
            explainer = shap.TreeExplainer(
                self.model, 
                data=X_bg, 
                feature_perturbation="interventional", 
                model_output="raw"
            ) 
            shap_values = explainer.shap_values(X_explain)
            if self.cfg.classification and isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class SHAP values
        elif self.cfg.model_name == "random_forest":
            # For sklearn random forest, use TreeExplainer
            explainer = shap.TreeExplainer(self.model, data=X_bg, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_explain)
            if self.cfg.classification and isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class SHAP values
        else:
            def model_predict(X):
                return super().predict(X)

            explainer = shap.KernelExplainer(model_predict, X_bg)
            shap_values = explainer.shap_values(X_explain)

        # Compute mean SHAP values across samples for each feature and store
        if type == "global": 
            mean_shap = np.mean(shap_values, axis=0)
            self.annotation_df["shap_values"] = mean_shap
        else:
            return shap_values

    def compute_interaction_values(
        self,
        batch: DataBatch,
        *,
        use_resids: bool = False,
        background: DataBatch | None = None,
        ram_mb: float | int | None = None,
        n_samples_shap: int = 1_000,
        chunk_size: int = 50,
        eps: float = 1e-10
    ):
        pd.set_option('future.no_silent_downcasting', True)
        assert np.array_equal(batch.geno_annotation.index.values, self.annotation_df.index.values), "Batch and estimator variant_idxs need to be identical"

        X_explain, y_explain = batch.X, batch.get_labels(use_resids)
        X_bg = background.X if background is not None else X_explain

        if X_explain.shape[0] > n_samples_shap:
            sample_indices = data_utils.adaptive_sampling(
                np.arange(X_explain.shape[0]),
                y_explain,
                self.cfg.classification,
                size=n_samples_shap,
                k=1,
                strategy="balanced"
            )
            X_explain = X_explain[sample_indices]

        if self.cfg.backend == "gpu":
            try:
                chunk_explainer = shap.GPUTreeExplainer(
                    self.model,
                    #data=X_bg,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                )
            except ImportError:
                self._logger.warning(
                    "Failed importing GPUTreeExplainer. To use GPUTreeExplainer, shap must be installed from source. Using TreeExplainer instead..."
                )
                chunk_explainer = shap.TreeExplainer(
                    self.model,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                )
        else:
            chunk_explainer = shap.TreeExplainer(
                self.model,
                #data=X_bg,
                feature_perturbation="tree_path_dependent",
                model_output="raw",
            )
            

        n_samples = min(X_explain.shape[0], n_samples_shap)
        n_features = X_explain.shape[1]
        interact_sum = np.zeros((n_features, n_features), dtype=np.float64)
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_X_explain = X_explain[start_idx:end_idx]
            chunk_interact_values = chunk_explainer.shap_interaction_values(chunk_X_explain)

            interact_sum += chunk_interact_values.sum(axis=0)

        interact_values = interact_sum / np.maximum(n_samples, 1)
        
        self.interactions = interact_values