import inspect
import logging
import os
import uuid
from typing import Any, Dict, Literal

import joblib
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .configs import ModelConfig


class WeakEstimator:
    """Base class for weak meta models that train on subsets of data."""

    def __init__(self, cfg: ModelConfig, model_init_params: Dict[str, Any] | None = None):
        """Initialize the weak meta model.

        Args:
            cfg: Model configuration
            model_init_params: Optional parameters for model initialization
        """
        self.cfg = cfg
        self._logger = logging.getLogger(self.__class__.__name__)
        self.model: Any | None = None
        self.scaler: StandardScaler | None = None
        self._init_model(model_init_params)

    def _init_model(self, model_init_params: Dict[str, Any] | None) -> None:
        """Initialize the underlying model based on configuration."""
        if self.cfg.model_type == "linear":
            self._init_linear_model(model_init_params)
        else:
            self._init_nonlinear_model()

    def _init_linear_model(self, model_init_params: Dict[str, Any] | None) -> None:
        """Initialize linear models."""
        if self.cfg.model_name.startswith("sgd"):
            penalty: Literal["l2", "l1", "elasticnet"] | None = None
            if self.cfg.model_name != "sgd":
                penalty_str = self.cfg.model_name.split("_")[-1]
                if penalty_str in ["l1", "l2", "elasticnet"]:
                    penalty = penalty_str

            if self.cfg.classification:
                from sklearn.linear_model import SGDClassifier

                model = SGDClassifier(
                    random_state=self.cfg.seed,
                    loss="log_loss",
                    penalty=penalty,
                    class_weight="balanced" if self.cfg.balance_classes else None,
                    n_jobs=self.cfg.n_jobs,
                    max_iter=2000,
                    **self.cfg.hyperparameters,
                )
            else:
                from sklearn.linear_model import SGDRegressor

                model = SGDRegressor(
                    random_state=self.cfg.seed,
                    loss="squared_error",
                    penalty=penalty,
                    n_jobs=self.cfg.n_jobs,
                    **self.cfg.hyperparameters,
                )
        elif self.cfg.model_name == "linear":
            # Use LogisticRegression for classification tasks
            if self.cfg.backend == "gpu":
                try:
                    if self.cfg.classification:
                        from cuml.linear_model import LogisticRegression
                    else:
                        from cuml.linear_model import LinearRegression
                    from cuml import set_global_output_type

                    set_global_output_type("numpy")
                except ImportError:
                    raise ImportError(
                        "Could not find cuml in your dependencies. Please make sure you have downloaded the optional dependencies 'gpu' to use cuML for sklearn models."
                    )
            else:
                if self.cfg.classification:
                    from sklearn.linear_model import LogisticRegression
                else:
                    from sklearn.linear_model import LinearRegression

            if self.cfg.classification:
                model = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced" if self.cfg.balance_classes else None,
                    **self.cfg.hyperparameters,
                )
            else:
                model = LinearRegression(**self.cfg.hyperparameters)
        elif self.cfg.model_name == "linear_l1":
            # Use LogisticRegression for classification tasks
            if self.cfg.backend == "gpu":
                try:
                    if self.cfg.classification:
                        from cuml.linear_model import LogisticRegression
                    else:
                        from cuml.linear_model import Lasso
                    from cuml import set_global_output_type

                    set_global_output_type("numpy")
                except ImportError:
                    raise ImportError(
                        "Could not find cuml in your dependencies. Please make sure you have downloaded the optional dependencies 'gpu' to use cuML for sklearn models."
                    )
            else:
                if self.cfg.classification:
                    from sklearn.linear_model import LogisticRegression
                else:
                    from sklearn.linear_model import Lasso

            if self.cfg.classification:
                if ("alpha" in self.cfg.hyperparameters) and ("C" not in self.cfg.hyperparameters):
                    self.cfg.hyperparameters["C"] = (
                        1.0 / self.cfg.hyperparameters["alpha"]
                        if self.cfg.hyperparameters["alpha"] > 0
                        else 1.0
                    )
                    self.cfg.hyperparameters.pop("alpha")
                model = LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    class_weight="balanced" if self.cfg.balance_classes else None,
                    max_iter=2000,
                    **self.cfg.hyperparameters,
                )
            else:
                if ("C" in self.cfg.hyperparameters) and ("alpha" not in self.cfg.hyperparameters):
                    self.cfg.hyperparameters["alpha"] = (
                        1.0 / self.cfg.hyperparameters["C"]
                        if self.cfg.hyperparameters["C"] > 0
                        else 1.0
                    )
                    self.cfg.hyperparameters.pop("C")
                model = Lasso(**self.cfg.hyperparameters)

        elif self.cfg.model_name == "linear_l2":
            # Use LogisticRegression for classification tasks
            if self.cfg.backend == "gpu":
                try:
                    if self.cfg.classification:
                        from cuml.linear_model import LogisticRegression
                    else:
                        from cuml.linear_model import Ridge
                    from cuml import set_global_output_type

                    set_global_output_type("numpy")
                except ImportError:
                    raise ImportError(
                        "Could not find cuml in your dependencies. Please make sure you have downloaded the optional dependencies 'gpu' to use cuML for sklearn models."
                    )
            else:
                if self.cfg.classification:
                    from sklearn.linear_model import LogisticRegression
                else:
                    from sklearn.linear_model import Ridge

            if self.cfg.classification:
                if ("alpha" in self.cfg.hyperparameters) and ("C" not in self.cfg.hyperparameters):
                    self.cfg.hyperparameters["C"] = (
                        1.0 / self.cfg.hyperparameters["alpha"]
                        if self.cfg.hyperparameters["alpha"] > 0
                        else 1.0
                    )
                    self.cfg.hyperparameters.pop("alpha")
                model = LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced" if self.cfg.balance_classes else None,
                    max_iter=2000,
                    **self.cfg.hyperparameters,
                )
            else:
                if ("C" in self.cfg.hyperparameters) and ("alpha" not in self.cfg.hyperparameters):
                    self.cfg.hyperparameters["alpha"] = (
                        1.0 / self.cfg.hyperparameters["C"]
                        if self.cfg.hyperparameters["C"] > 0
                        else 1.0
                    )
                    self.cfg.hyperparameters.pop("C")
                model = Ridge(solver="svd", **self.cfg.hyperparameters)

        elif self.cfg.model_name == "elasticnet":
            # Use LogisticRegression for classification tasks
            if self.cfg.backend == "gpu":
                try:
                    if self.cfg.classification:
                        from cuml.linear_model import LogisticRegression
                    else:
                        from cuml.linear_model import ElasticNet
                    from cuml import set_global_output_type

                    set_global_output_type("numpy")
                except ImportError:
                    raise ImportError(
                        "Could not find cuml in your dependencies. Please make sure you have downloaded the optional dependencies 'gpu' to use cuML for sklearn models."
                    )
            else:
                if self.cfg.classification:
                    from sklearn.linear_model import LogisticRegression
                else:
                    from sklearn.linear_model import ElasticNet

            if self.cfg.classification:
                if ("alpha" in self.cfg.hyperparameters) and ("C" not in self.cfg.hyperparameters):
                    self.cfg.hyperparameters["C"] = (
                        1.0 / self.cfg.hyperparameters["alpha"]
                        if self.cfg.hyperparameters["alpha"] > 0
                        else 1.0
                    )
                    self.cfg.hyperparameters.pop("alpha")
                model = LogisticRegression(
                    penalty="elasticnet",
                    solver="liblinear",
                    class_weight="balanced" if self.cfg.balance_classes else None,
                    max_iter=2000,
                    **self.cfg.hyperparameters,
                )
            else:
                if ("C" in self.cfg.hyperparameters) and ("alpha" not in self.cfg.hyperparameters):
                    self.cfg.hyperparameters["alpha"] = (
                        1.0 / self.cfg.hyperparameters["C"]
                        if self.cfg.hyperparameters["C"] > 0
                        else 1.0
                    )
                    self.cfg.hyperparameters.pop("C")
                model = ElasticNet(**self.cfg.hyperparameters)

        elif self.cfg.model_name == "bayesian":
            if self.cfg.backend == "gpu":
                try:
                    from cuml.linear_model import BayesianRidge
                    from cuml import set_global_output_type

                    set_global_output_type("numpy")
                except ImportError:
                    raise ImportError(
                        "Could not find cuml in your dependencies. Please make sure you have downloaded the optional dependencies 'gpu' to use cuML for sklearn models."
                    )
            else:
                from sklearn.linear_model import BayesianRidge
            model = BayesianRidge(max_iter=2000, **self.cfg.hyperparameters)
        else:
            raise ValueError(f"Linear model {self.cfg.model_name} is not supported.")

        if model_init_params:
            for param in self.cfg.init_params:
                if param in model_init_params:
                    # Skip 'classes_' parameter for regression tasks in the combined 'linear' model
                    if (
                        (self.cfg.model_name in ["linear", "linear_l1"])
                        and (param == "classes_")
                        and (not self.cfg.classification)
                    ):
                        continue
                    setattr(model, param, model_init_params[param])

                else:
                    raise ValueError(f"Required parameter '{param}' not found in model_init_params")

        self.model = model

    def _init_nonlinear_model(self) -> None:
        """Initialize non-linear models."""
        if self.cfg.model_name == "lightgbm":
            try:
                if self.cfg.classification:
                    from lightgbm import LGBMClassifier
                else:
                    from lightgbm import LGBMRegressor
            except ImportError:
                raise ImportError(
                    "Could not find lightgbm in your dependencies. Please make sure you have downloaded the optional dependencies 'lightgbm' or 'all_models' to use this model."
                )

            if self.cfg.classification:
                model = LGBMClassifier(
                    random_state=self.cfg.seed,
                    device_type="gpu" if self.cfg.backend == "gpu" else "cpu",
                    verbose=-1,
                    n_jobs=self.cfg.n_jobs,
                    objective="binary",
                    class_weight="balanced" if self.cfg.balance_classes else None,
                    metric="auc",
                    **self.cfg.hyperparameters,
                )
            else:
                model = LGBMRegressor(
                    random_state=self.cfg.seed,
                    device_type="gpu" if self.cfg.backend == "gpu" else "cpu",
                    verbose=-1,
                    n_jobs=self.cfg.n_jobs,
                    **self.cfg.hyperparameters,
                )
        elif self.cfg.model_name == "catboost":
            try:
                if self.cfg.classification:
                    from catboost import CatBoostClassifier
                else:
                    from catboost import CatBoostRegressor
            except ImportError:
                raise ImportError(
                    "Could not find catboost in your dependencies. Please make sure you have downloaded the optional dependencies 'catboost' or 'all_models' to use this model."
                )
            if self.cfg.backend == "gpu":
                self.cfg.hyperparameters.setdefault("devices", "0")
                self.cfg.hyperparameters.setdefault("task_type", "GPU")

            if self.cfg.seed is not None:
                self.cfg.hyperparameters["seed"] = self.cfg.seed
            if self.cfg.classification:
                model = CatBoostClassifier(
                    thread_count=self.cfg.n_jobs,
                    logging_level="Silent",
                    **self.cfg.hyperparameters,
                )
            else:
                model = CatBoostRegressor(
                    thread_count=self.cfg.n_jobs,
                    logging_level="Silent",
                    **self.cfg.hyperparameters,
                )
        elif self.cfg.model_name == "xgboost":
            try:
                if self.cfg.classification:
                    from xgboost import XGBClassifier
                else:
                    from xgboost import XGBRegressor
            except ImportError:
                raise ImportError(
                    "Could not find xgboost in your dependencies. Please make sure you have downloaded the optional dependencies 'xgboost' or 'all_models' to use this model."
                )
            if self.cfg.backend == "gpu":
                # Ensure GPU params are set
                self.cfg.hyperparameters.setdefault("predictor", "gpu_predictor")
                self.cfg.hyperparameters.setdefault("device", "cuda:0")

            if self.cfg.seed is not None:
                self.cfg.hyperparameters["seed"] = self.cfg.seed
            if self.cfg.classification:
                model = XGBClassifier(
                    objective="binary:logistic",
                    n_jobs=self.cfg.n_jobs,
                    eval_metric="auc",
                    early_stopping_rounds=20,
                    verbosity=0,
                    **self.cfg.hyperparameters,
                )
            else:
                model = XGBRegressor(
                    objective="reg:squarederror",
                    n_jobs=self.cfg.n_jobs,
                    early_stopping_rounds=20,
                    verbosity=0,
                    **self.cfg.hyperparameters,
                )

        elif self.cfg.model_name == "random_forest":
            if self.cfg.classification:
                from sklearn.ensemble import RandomForestClassifier

                model = RandomForestClassifier(
                    random_state=self.cfg.seed,
                    class_weight="balanced" if self.cfg.balance_classes else None,
                    n_jobs=self.cfg.n_jobs,
                    **self.cfg.hyperparameters,
                )
            else:
                from sklearn.ensemble import RandomForestRegressor

                model = RandomForestRegressor(
                    random_state=self.cfg.seed,
                    n_jobs=self.cfg.n_jobs,
                    **self.cfg.hyperparameters,
                )
        elif self.cfg.model_name == "simple_mlp":
            from .custom import DNNModel

            model = DNNModel(self.cfg, model_name="simple_mlp")
        else:
            raise ValueError(f"Non-linear model {self.cfg.model_name} is not supported.")

        self.model = model

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        weak_estimator_dir = os.path.join(self.cfg.model_dir, "weak_estimator")
        os.makedirs(weak_estimator_dir, exist_ok=True)

        model_path = os.path.join(weak_estimator_dir, f"weak_estimator_{self.cfg.model_id}.pkl")

        if self.cfg.model_type != "dnn":
            joblib.dump(self.model, model_path)
        else:
            self.model.save_state_dict(model_path)

    def get_base_model(self) -> Any:
        """Get the underlying model instance."""
        return self.model

    def fit(
        self,
        X_train: npt.NDArray,
        y_train: npt.NDArray,
        X_val: npt.NDArray | None = None,
        y_val: npt.NDArray | None = None,
        scaler: StandardScaler | None = None,
        sample_weight: npt.NDArray | None = None,
    ) -> None:
        """Fit the model to training data using numpy arrays.

        Args:
            X_train: Training features array
            y_train: Training labels array
            X_val: Optional validation features array
            y_val: Optional validation labels array
            scaler: Optional scaler for label standardization
        """
        self.scaler = scaler

        if self.scaler and (not self.cfg.classification):
            y_train = self.scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
            if y_val is not None:
                y_val = self.scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

        if hasattr(self.model, "fit"):
            # heck if the fit method accepts validation parameters
            fit_signature = inspect.signature(self.model.fit)
            fit_params = {}

            has_val = (X_val is not None) and (y_val is not None)
            if self.cfg.model_name == "lightgbm" and has_val:
                import lightgbm as lgb

                fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
                fit_params["eval_metric"] = "auc" if self.cfg.classification else "rmse"
            if self.cfg.model_name == "xgboost":
                fit_params["verbose"] = False

            # Generic sklearn-style models
            if has_val and "eval_set" in inspect.signature(self.model.fit).parameters:
                fit_params["eval_set"] = [(X_val, y_val)]

            # add sample_weight if the model supports it and weights are provided
            if "sample_weight" in fit_signature.parameters and sample_weight is not None:
                fit_params["sample_weight"] = sample_weight

            if self.cfg.backend == "gpu" and self.cfg.model_type == "linear":
                # cuML models return fitted model from fit()
                fitted_model = self.model.fit(X_train, y_train, **fit_params)
                self.model = fitted_model
            else:
                self.model.fit(X_train, y_train, **fit_params)
        else:
            raise NotImplementedError("Model does not have fit method!")

        if self.cfg.save_model:
            self._save_model()

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        """Make predictions using a numpy array X as input.

        Args:
            X: A numpy array with shape (n_samples, n_features) where n_features matches
               the number of features used to train the model. The ordering of the
               features must match the ordering used to train the model.

        Returns:
            Predictions as numpy array
        """
        if self.cfg.classification and hasattr(self.model, "predict_proba"):
            prediction = self.model.predict_proba(X)[:, 1]
        else:
            prediction = self.model.predict(X)

        if (not self.cfg.classification) and (self.scaler is not None):
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)

        return prediction
