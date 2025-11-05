from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Literal

import joblib

from ...base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    classification: bool = field(
        metadata={"help": "Whether phenotype is binary or continuous"}
    )
    batch_size: int = field(
        default=1, metadata={"help": "Number of mini batches per batch"}
    )
    n_jobs: int = field(
        default=-1,
        metadata={
            "help": "Number of jobs used to run training. If set to -1, n_jobs is set via 'os.cpu_count()'."
        },
    )
    backend: Literal["cpu", "gpu"] = field(
        default="cpu",
        metadata={
            "help": "Backend to use for training. For DNN models, GPU is automatically selected."
        },
    )
    ram_mb: int = field(default=16_000, metadata={"help": "Total available RAM in MB"})
    scorer: Literal["r2", "rocauc", "pearson_corr"] | None = field(
        default=None,
        metadata={"help": "Scorer to use for model aggregation and early stopping"},
    )
    patience: int = field(
        default=0,
        metadata={
            "help": "Number of batches to wait for improvement before early stopping"
        },
    )
    seed: int = field(
        default=22, metadata={"help": "Seed for constant evaluation conditions"}
    )
    log_with_wandb: bool = field(
        default=False, metadata={"help": "Whether to log metrics with Weights & Biases"}
    )
    save_annotation: bool = field(
        default=False, metadata={"help": "Path to save annotation file to."}
    )
    save_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to save the model to the model_path defiend in .env."
        },
    )
    compute_shap: bool = field(
        default=False, metadata={"help": "Whether to compute shap values during fit"}
    )

    def __post_init__(self):
        if self.scorer is None:
            self.scorer = "rocauc" if self.classification else "r2"

        if self.n_jobs == -1:
            self.n_jobs = joblib.cpu_count()
