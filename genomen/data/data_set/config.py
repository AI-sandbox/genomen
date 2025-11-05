import logging
from dataclasses import field
from typing import Dict, List, Literal

from pydantic.dataclasses import dataclass

from ...base_config import BaseConfig
from . import utils

logger = logging.getLogger(__name__)


@dataclass
class GWASConfig:
    path: str | None = field(
        default=None,
        metadata={
            "help": "Path to GWAS study (https://www.ebi.ac.uk/gwas/), used to sample variants using p-values."
        },
    )
    snps_column: str = field(
        default="SNPS",
        metadata={"help": "Name of SNPs column in GWAS data"},
    )
    pvalue_column: str | None = field(
        default=None,
        metadata={"help": "Name of p-value column in GWAS data"},
    )
    nlogpvalue_column: str | None = field(
        default=None,
        metadata={"help": "Name of negative log of p-value column if any in GWAS data"},
    )
    sep: str = field(
        default=r"\s+",
        metadata={"help": "Seperator to read GWAS into df."},
    )
    impute_val: float = field(
        default=1.0,
        metadata={"help": "Default value with which variants not found in GWAS will be imputed."},
    )
    pvalue_aggregation: Literal[
        "mean", "min", "fisher", "pearson", "tippett", "stouffer", "mudholkar_george"
    ] = field(default="fisher", metadata={"help": "Method to aggregate p-values."})


@dataclass
class CovarConfig:
    include_covars: bool = field(
        default=True, metadata={"help": "Whether to include or regress out covariates"}
    )
    covar_keys: List[str] = field(
        default_factory=[
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
        metadata={"help": "List of covar keys to use. Keys must be present in master file"},
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


@dataclass
class LDSamplingConfig:
    prune_kb: int = field(
        default=250,
        metadata={
            "help": "Window size in kilobases for LD pruning (--indep-pairwise window size)."
        },
    )
    prune_step: int = field(
        default=50,
        metadata={"help": "Step size in SNPs for LD pruning (--indep-pairwise step size)."},
    )
    prune_r2: float = field(
        default=0.1,
        metadata={"help": "r^2 threshold for LD pruning (--indep-pairwise r^2 threshold)."},
    )
    tau: float = field(
        default=0.1, metadata={"help": "Minimum r^2 threshold for assigning SNPs to LD blocks."}
    )
    ld_window_kb: int = field(
        default=1000,
        metadata={"help": "Window size in kilobases for LD calculation (--ld-window-kb)."},
    )
    ld_window: int = field(
        default=50000, metadata={"help": "Maximum number of SNPs in LD window (--ld-window)."}
    )
    max_score: float = field(
        default=10.0, metadata={"help": "Maximum number of SNPs in LD window (--ld-window)."}
    )
    eps: float = field(
        default=0.0,
        metadata={"help": "Hyperparamer controlling epsilon-greedy sampling."},
    )
    eps_schedule: Literal["constant", "step"] = field(
        default="constant",
        metadata={"help": "Hyperparamer controlling epsilon-greedy sampling."},
    )
    eps_step_size: float = field(
        default=0.0,
        metadata={"help": "Hyperparamer controlling epsilon-greedy sampling."},
    )
    temp: float = field(
        default=1.0,
        metadata={
            "help": "Temperature used for softmax to create categorical probability distribution over variants."
        },
    )


@dataclass
class VariantSamplingConfig:
    strat: Literal["random", "chromosome", "window", "LD", "GWAS"] = field(
        default="random",
        metadata={"help": "Strategy used for sampling features from the data."},
    )
    max_features: int = field(
        default=10_000,
        metadata={"help": "Maximum number of features to use for training"},
    )
    gwas_config: GWASConfig | None = field(
        default_factory=GWASConfig,
        metadata={"help": "GWAS study config"},
    )
    window_overlap_ratio: float = field(
        default=0.5,
        metadata={
            "help": "Percentage of window size (max_samples) shared between subsequent windows. Has to be between 0.0 and 1.0"
        },
    )
    ld_config: LDSamplingConfig = field(
        default_factory=LDSamplingConfig,
        metadata={"help": "Dataclass encompassing hyperparameters for LD sampling"},
    )
    blocks_max_kb: int = field(
        default=400,
        metadata={
            "help": "Maximum block size in kilobases for LD block definition (PLINK 1.9 --blocks-max-kb)."
        },
    )
    blocks_strong_lowci: float = field(
        default=0.8,
        metadata={
            "help": "Lower confidence interval for strong LD in block definition (PLINK 1.9 --blocks-strong-lowci)."
        },
    )
    # Helper, will be filled automatically
    stride: int | None = field(
        default=None,
        metadata={"help": "Stride used for window sampling. Will be computed automatically."},
    )


@dataclass
class DataSetConfig(BaseConfig):
    phenotype_id: str = field(
        metadata={"help": "ID of the phenotype to analyze. Must be column in 'master' file."}
    )
    classification: bool = field(metadata={"help": "Whether phenotype is binary or continuous"})
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
    sex: Literal["m", "w"] | None = field(
        default=None,
        metadata={"help": "Wether to filter for a specific sex. If None, no filter is set."},
    )
    sex: Literal["m", "w"] | None = field(
        default=None,
        metadata={"help": "Wether to filter for a specific sex. If None, no filter is set."},
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

    def __post_init__(self):
        # Load paths based on file format if not already provided
        if not self.paths:
            self.paths = utils.get_data_paths(self.file_format)

        self.compute_shap = self.variant_sampling.strat == "MAB"

        if self.variant_sampling.strat == "window":
            self.variant_sampling.stride = int(
                self.variant_sampling.max_features
                * (1 - self.variant_sampling.window_overlap_ratio)
            )

        if isinstance(self.populations, str):
            self.populations = [self.populations]
        self.populations: List[str] = sorted(self.populations)
