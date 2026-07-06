import logging
from dataclasses import field
from typing import Dict, List, Literal

from pydantic.dataclasses import dataclass

from ...base_config import BaseConfig
from . import utils

logger = logging.getLogger(__name__)


@dataclass
class CovarConfig:
    include_covars: bool = field(
        default=True, metadata={"help": "Whether to include or regress out covariates"}
    )
    covar_keys: List[str] = field(
        default_factory=lambda: [
            "age",
            "sex",
            "Global_PC1",
            "Global_PC2",
            "Global_PC3",
            "Global_PC4",
            "Global_PC5",
            "Global_PC6",
            "Global_PC7",
            "Global_PC8",
            "Global_PC9",
            "Global_PC10",
        ],
        metadata={
            "help": "List of covar keys to use. Keys must be present in master file"
        },
    )


@dataclass
class SampleSamplingConfig:
    strat: Literal["random", "stratify", "balanced"] = field(
        default="random",
        metadata={
            "help": "Strategy used for sampling samples from the data. 'balanced' uses k:1 ratio of controls to cases."
        },
    )
    max_samples: int | None = field(
        default=100_000,
        metadata={"help": "Maximum number of samples to use for training"},
    )
    balance_pops: bool = field(
        default=False,
        metadata={"help": "Whether to balance populations while sampling"},
    )
    k: int = field(
        default=3,
        metadata={
            "help": "Ratio of negative (control) samples to positive (case) samples for balanced sampling strategy"
        },
    )
    split_seed: int = field(
        default=0,
        metadata={
            "help": "Fixed seed used for sample splitting, independent of the training seed, so the split is reproducible across sweeps."
        },
    )
    fix_balanced_samples: bool = field(
        default=True,
        metadata={
            "help": "If True, the balanced/stratified sample draw is fixed across all patches (same samples every estimator). "
                    "If False, each patch draws a fresh balanced sample using its own batch seed, increasing ensemble diversity."
        },
    )


@dataclass
class VariantSamplingConfig:
    strat: Literal["random"] = field(
        default="random",
        metadata={"help": "Strategy used for sampling features from the data."},
    )
    max_features: int = field(
        default=10_000,
        metadata={"help": "Maximum number of features to use for training"},
    )


@dataclass
class DataSetConfig(BaseConfig):
    phenotype_id: str = field(
        metadata={
            "help": "ID of the phenotype to analyze. Must be column in 'master' file."
        }
    )
    classification: bool = field(
        metadata={"help": "Whether phenotype is binary or continuous"}
    )
    file_format: Literal["plink"] = field(
        default="plink",
        metadata={"help": "File format of input. Only accepts 'plink' for now."},
    )
    populations: str | List[str] = field(
        default="white_british",
        metadata={"help": "Population(s) to include in the analysis"},
    )
    include_x_chromosome: bool = field(
        default=False, metadata={"help": "Whether to include X chromosome variants"}
    )
    maf_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minor allele frequency threshold for variants. If 0.0, variants are not filtered wrt to MAF."
        },
    )
    missingness_threshold: float = field(
        default=0.05,
        metadata={
            "help": "Maximum per-variant missing call rate allowed. Variants with a higher missing "
                    "rate on the train samples are dropped. If 1.0, variants are not filtered wrt to missingness."
        },
    )
    sex: Literal["m", "w"] | None = field(
        default=None,
        metadata={
            "help": "Wether to filter for a specific sex. If None, no filter is set."
        }
    )
    covar_config: CovarConfig = field(
        default_factory=CovarConfig, metadata={"help": "Covar configuration"}
    )
    sample_sampling: SampleSamplingConfig = field(
        default_factory=SampleSamplingConfig,
        metadata={"help": "Sample sampling config"},
    )
    variant_sampling: VariantSamplingConfig = field(
        default_factory=VariantSamplingConfig,
        metadata={"help": "Variant sampling config"},
    )
    # Helper, will be filled automatically
    paths: Dict[str, str] | None = field(
        default_factory=dict, metadata={"help": "Paths to the input data"}
    )
    is_train: bool = field(
        default=False, metadata={"help": "If dataset is train set (used for logging)."}
    )
    compute_shap: bool = field(
        default=False, metadata={"help": "Whether train needs to compute shap values."}
    )
    simulate: bool = field(
        default=False,
        metadata={"help": "Whether phenotype will be simulated. If True, phenotype_id does not need to exist in the master file."},
    )

    def __post_init__(self):
        # Load paths based on file format if not already provided
        if not self.paths:
            self.paths = utils.get_data_paths(self.file_format)

        if isinstance(self.populations, str):
            self.populations = [self.populations]
        self.populations: List[str] = sorted(self.populations)
