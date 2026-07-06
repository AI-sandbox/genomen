import json
import os
from copy import deepcopy
from dataclasses import asdict, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic.dataclasses import dataclass

from ...base_config import BaseConfig


@dataclass
class ModelConfig:
    model_name: str = field(default="lightgbm", metadata={"help": "Name of the model to be used"})
    hyperparameters: Dict[str, Any] | Dict[str, Dict[str, Any]] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of hyperparameters for the model"},
    )
    balance_classes: bool = field(
        default=False,
        metadata={"help": "Whether to balance classes in classification tasks"},
    )
    ensemble_estimator_names: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of weak estimator names to use for ensemble. If larger than batch_size, models are randomly selected."
        },
    )
    # Helper, will be filled automatically
    model_type: Literal["linear", "tree", "ensemble", "other"] | None = field(
        default=None, metadata={"help": "Type of the model"}
    )
    linear: Optional[bool] = field(
        default=None, metadata={"help": "Whether the model is linear model or not"}
    )
    use_offset: bool = field(
        default=False,
        metadata={
            "help": "Use covar predictions as LGBM init_score instead of fitting on residualized labels (lightgbm only)."
        },
    )
    classification: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether the model is intended for classification or regression. Can be 'classification', 'regression', or 'hybrid'."
        },
    )
    init_params: List[str] = field(
        default_factory=list, metadata={"help": "List of parameters linear model has"}
    )
    seed: Optional[int] = field(default=None, metadata={"help": "Random seed for reproducibility"})
    model_id: int = field(
        default=-1, metadata={"help": "Int identifying model in strong estimator"}
    )
    save_model: bool = field(
        default=False, metadata={"help": "Whether to save the trained model to disk"}
    )
    n_jobs: int = field(
        default=1, metadata={"help": "Number of parallel jobs to run during training"}
    )
    backend: Literal["cpu", "gpu"] = field(
        default="cpu", metadata={"help": "Computing backend to use for training"}
    )
    max_features: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of features to use for training"},
    )
    ram_mb: int = field(default=16_000, metadata={"help": "Total available RAM in MB"})

    def __post_init__(self):
        self._initialize_from_catalog()

    def _initialize_from_catalog(self, overwrite: bool = False):
        models_json_path = Path(__file__).parent.parent / "models.json"
        with open(models_json_path) as f:
            models_info = json.load(f)["models"]

        mi = models_info.get(self.model_name)
        if not mi:
            raise ValueError(f"Unknown model '{self.model_name}' in {models_json_path}")

        # Validate task if known
        if (self.classification is not None) or overwrite:
            task = "classification" if self.classification else "regression"
            supp = mi.get("supp_tasks", [])
            if task not in supp:
                raise ValueError(
                    f"Model {self.model_name} does not support {task}. Supported: {supp}"
                )

        # Fill inferred defaults
        if (self.model_type is None) or overwrite:
            self.model_type = mi["model_type"]
        if (not self.init_params) or overwrite:
            self.init_params = deepcopy(mi.get("init_params", []))
        self.linear = "linear" in self.model_type


@dataclass
class CovarConfig:
    covar_strat: Literal["residualization", "predictive"] = field(
        default="residualization", metadata={"help": "Covar integration strategy"}
    )
    model_config: Optional[ModelConfig] = field(
        default_factory=ModelConfig, metadata={"help": "Covar model config"}
    )


@dataclass
class FeatureSelectionConfig:
    method: Literal["none", "k_best", "percentile", "variance_threshold", "mutual_info", "rfe"] = (
        field(default="none", metadata={"help": "Feature selection method to use"})
    )
    k: int = field(
        default=1000,
        metadata={"help": "Number of features to select (for k_best and rfe methods)"},
    )
    percentile: float = field(
        default=10.0,
        metadata={"help": "Percentile of features to select (for percentile method)"},
    )
    variance_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minimum variance threshold for features (for variance_threshold method)"
        },
    )
    score_func: Literal["f_classif", "f_regression", "r_regression", "chi2"] | None = field(
        default=None,
        metadata={"help": "Score function for univariate feature selection"},
    )


@dataclass
class GenoPreprocessingConfig:
    z_score_thresh: float = field(
        default=0.0,
        metadata={
            "help": "Standard-score to use for outlier filtering. 0.0 means no outlier filtering."
        },
    )
    standard_labels: bool = field(
        default=False, metadata={"help": "Whether to standardize numerical features"}
    )
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig,
        metadata={"help": "Feature selection config"},
    )


@dataclass
class AggregatorConfig:
    filter_strat: Literal["none", "positive", "geq-average", "top-p-percentile"] = field(
        default="none",
        metadata={"help": "Method to filter models before aggregation"},
    )
    agg_strat: Literal["mean", "rank-mean", "loss-weighted-average"] = field(
        default="mean", metadata={"help": "Method to aggregate predictions"}
    )
    p: float = field(
        default=0.75,
        metadata={
            "help": "Percentile p of models based on val score to use for aggregation. Has to be between 0.0 and 1.0"
        },
    )
    temp: float = field(
        default=0.5,
        metadata={
            "help": "Temperature of softmax to compute estimator weights when mode 'loss-weighted-average' is selected."
        },
    )
    pos_thresh: float | None = field(
        default=None,
        metadata={
            "help": "Minimum score above which an estimator is classified as positively contributing to predictive performance."
        },
    )
    # Helper, will be filled automatically
    hold_out_neeeded: bool = field(
        default=False,
        metadata={"help": "Whether extra hold-out set is needed for aggregation."},
    )

    def __post_init__(self):
        if self.filter_strat == "positive" and self.p is None:
            raise ValueError("Must provide a value for p when using positive filter")

        self.hold_out_neeeded = (
            self.filter_strat != "none" or self.agg_strat == "loss-weighted-average"
        )

        if self.agg_strat == "loss-weighted-average" and self.temp is None:
            raise ValueError(
                "Must provide a value for temperature when using mode 'loss-weighted-average'"
            )


@dataclass
class GenoConfig:
    n_estimators: int = field(
        default=128, metadata={"help": "Number of estimators in the ensemble"}
    )
    compute_interactions: bool = field(
        default=False,
        metadata={
            "help": "Whether to include compue interactions when computing global shap values."
        },
    )
    preprocessing_config: Optional[GenoPreprocessingConfig] = field(
        default_factory=GenoPreprocessingConfig,
        metadata={"help": "Genotype preprocessing config"},
    )
    model_config: Optional[ModelConfig] = field(
        default_factory=ModelConfig, metadata={"help": "Geno model config"}
    )
    aggregator_config: Optional[AggregatorConfig] = field(
        default_factory=AggregatorConfig,
        metadata={"help": "Config for aggregating weak estimators"},
    )
    # Helper, will be filled automatically
    preprocessing_needed: bool | None = field(
        default=None, metadata={"help": "Whether preprocessing is needed."}
    )
    use_resids: bool = field(
        default=False,
        metadata={"help": "Whether geno model has to be fitted on residualized labels."},
    )
    is_ensemble: bool = field(
        default=False, metadata={"help": "Whether genotype model is an ensemble"}
    )


@dataclass
class GenomenModelConfig(BaseConfig):
    classification: bool = field(metadata={"help": "Whether phenotype is binary or continuous"})
    covar_config: Optional[CovarConfig] = field(
        default_factory=CovarConfig, metadata={"help": "Config for fitting covariates"}
    )
    geno_config: Optional[GenoConfig] = field(
        default_factory=GenoConfig, metadata={"help": "Config for fitting genotype"}
    )
    # Helper, will be filled automatically
    eps: float = field(
        default=1e-7,
        metadata={"help": "Epsilon for clipping probabilities to compute logits."},
    )
    include_covars: bool = field(
        default=True, metadata={"help": "Whether to include or regress out covariates"}
    )
    save_model: bool = field(default=False, metadata={"help": "Whether to save model."})
    max_features: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of features to use for training"},
    )
    backend: Literal["cpu", "gpu"] = field(
        default="cpu", metadata={"help": "Computing backend to use for training"}
    )
    model_dir: Optional[Path | str] | None = field(
        default=None, metadata={"help": "Path to save the mode under."}
    )
    ram_mb: int = field(default=16_000, metadata={"help": "Total available RAM in MB"})

    def __post_init__(self):
        # set use_resids and use_offset
        self.geno_config.use_resids = self.include_covars and (
            self.covar_config.covar_strat == "residualization"
        )

        # set model max_features to k (will not affect sampling)
        if self.max_features:
            if self.geno_config.preprocessing_config.feature_selection.method == "k_best":
                self.max_features = self.geno_config.preprocessing_config.feature_selection.k
            self.geno_config.model_config.max_features = self.max_features

        # init classification parameter based on on_resid
        self.covar_config.model_config.classification = self.classification
        # Models using use_offset train on original labels, so they retain the classification flag.
        # Models in residualization mode train on continuous residuals (regression task).
        self.geno_config.model_config.classification = self.classification and (
            not self.geno_config.use_resids or self.geno_config.model_config.use_offset
        )

        # init pos_thresh
        if self.geno_config.aggregator_config.pos_thresh is None:
            self.geno_config.aggregator_config.pos_thresh = 0.5 if self.classification else 0.0

        # pass down backend parameter to model_configs
        self.covar_config.model_config.backend = self.backend
        self.covar_config.model_config.ram_mb = self.ram_mb
        self.geno_config.model_config.backend = self.backend
        self.geno_config.model_config.ram_mb = self.ram_mb

        self.geno_config.preprocessing_needed = (
            self.geno_config.preprocessing_config.standard_labels
            or (
                self.geno_config.preprocessing_config.z_score_thresh > 0.0
                and (not self.geno_config.model_config.classification)
            )
            or self.geno_config.preprocessing_config.feature_selection.method != "none"
        )

        self.geno_config.is_ensemble = self.geno_config.model_config.model_name == "ensemble"

    def save(self, path: str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path where to save the config
        """
        # Create directory if it doesn't exist
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

        # Create config dict in the expected format
        config_dict = {"MetaModelConfig": asdict(self)}

        # Save config as YAML
        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
