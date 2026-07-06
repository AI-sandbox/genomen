"""Functions to split a DataSet into subsets."""

import logging
from typing import Tuple

import numpy as np

from ..data_set.data_set import DataSet
from ..data_set.geno_set import GenoSet
from ..data_set.pheno_set import PhenoSet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def bootstrap(
    data_set: DataSet,
    seed: int | None = None,
) -> Tuple[DataSet, ...]:
    """ """
    # Split all samples directly
    all_sample_idxs = data_set.phenotype.sample_idxs
    bootstrap_sample_idxs = np.random.choice(
        all_sample_idxs, size=len(all_sample_idxs), replace=True
    )

    # Create train set
    bootstrapped_pheno_annotation_df = data_set.phenotype.annotation_df.loc[
        bootstrap_sample_idxs
    ].copy()

    train_phenotype = PhenoSet(
        annotation_df=bootstrapped_pheno_annotation_df,
        covar_cfg=data_set.cfg.covar_config,
    )

    train_genotype = data_set.genotype.fork(
        annotation_df=data_set.genotype.annotation_df.copy(),
        n_samples=len(bootstrapped_pheno_annotation_df),
    )

    bootstrapped_set = DataSet(cfg=data_set.cfg, genotype=train_genotype, phenotype=train_phenotype)

    return bootstrapped_set
