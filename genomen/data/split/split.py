"""Functions to split a DataSet into subsets."""

import copy
import logging
from typing import Dict, Tuple

from genomen import data
import numpy as np
import pandas as pd

from ..data_set import utils
from ..data_set.data_set import DataSet
from ..data_set.geno_set import GenoSet
from ..data_set.pheno_set import PhenoSet
from ..sources import plink_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split(
    data_set: DataSet,
    test_size: float | None = 0.2,
    shuffle: bool = False,
    seed: int | None = None,
    population_aware_split: bool = False,
    split_by_col: Tuple[str, Tuple[str, ...]] | None = None,
    split_idxs: list | None = None,
) -> Tuple[DataSet, ...]:
    """
    Splits the dataset into subsets based on provided parameters.

    Args:
        dataset (DataSet): The dataset to split.
        test_size (float): Proportion of the dataset to include in the test split.
        shuffle (bool): Whether to shuffle the data before splitting.
        seed (int): Random seed for reproducibility.
        split_by_col (Tuple[str, Tuple[str, ...]]): Column name and tuple of labels.
                                                  At least two labels are required.

    Returns:
        Tuple[DataSet, ...]: A tuple of DataSet objects, one for each specified label.
                            If split_by_col is None, returns (set1, set2).
    """
    if split_idxs is not None:
        result = []
        for i, idxs in enumerate(split_idxs):
            idxs = np.asarray(idxs, dtype=np.uint32)
            pheno_ann = data_set.phenotype.annotation_df.loc[idxs].copy()
            phenotype = PhenoSet(
                annotation_df=pheno_ann,
                covar_cfg=data_set.cfg.covar_config,
            )
            genotype = data_set.genotype.fork(
                annotation_df=data_set.genotype.annotation_df.copy(),
                n_samples=len(pheno_ann),
            )
            cfg = copy.deepcopy(data_set.cfg)
            cfg.is_train = i == 0
            result.append(DataSet(cfg=cfg, genotype=genotype, phenotype=phenotype))
        return tuple(result)

    if split_by_col is not None:
        column_name, labels = split_by_col

        # Ensure we have at least two labels
        if len(labels) < 2:
            raise ValueError("At least two labels are required for splitting")

        paths: Dict[str, str] = utils.get_data_paths(data_set.cfg.file_format)
        try:
            master_df: pd.DataFrame = plink_utils.load_master_data(
                paths["master_path"], columns=["IID", "population", column_name]
            )
            master_df[column_name] = master_df[column_name].astype(str)

            # Create fam_idx mapping using data_set.iids
            iid_to_idx = dict(
                zip(
                    data_set.phenotype.annotation_df["iid"],
                    data_set.phenotype.annotation_df.index,
                )
            )
            master_df["fam_idx"] = master_df["IID"].map(iid_to_idx)
            master_df = master_df.dropna(subset=["fam_idx"])  # Remove any IIDs not in data_set
            master_df["fam_idx"] = master_df["fam_idx"].astype(np.uint32)

        except Exception as e:
            raise ValueError(f"Error loading master data: {e}")

        # Check that all required labels exist in the data
        missing_labels = set(labels) - set(master_df[column_name].unique())
        if missing_labels:
            raise ValueError(f"Missing required labels in split column: {missing_labels}")

        data_sets = []

        for label in labels:
            # Get sample_idxs for this label by population
            label_sample_idxs = master_df[master_df[column_name] == label]["fam_idx"]
            label_pheno_annotation_df = data_set.phenotype.annotation_df.loc[
                label_sample_idxs
            ].copy()

            # Get IIDs for this label
            label_phenotype = PhenoSet(
                annotation_df=label_pheno_annotation_df,
                covar_cfg=data_set.cfg.covar_config,
            )
            label_genotype = data_set.genotype.fork(
                annotation_df=data_set.genotype.annotation_df.copy(),
                n_samples=len(label_pheno_annotation_df),
            )

            label_cfg = copy.deepcopy(data_set.cfg)
            if label == "train":
                label_cfg.is_train = True
            else:
                label_cfg.is_train = False

            label_data_set = DataSet(label_cfg, label_genotype, label_phenotype)

            data_sets.append(label_data_set)

        train_set = data_sets[0]
        nottrain_set = data_sets[1:]
    else:
        # Regular splitting without column-based split - only train/test
        rng = np.random.RandomState(seed)

        if population_aware_split:
            # Split samples by population
            pop_train_idxs = []
            pop_test_idxs = []

            for population in data_set.cfg.populations:
                pop_mask = data_set.phenotype.annotation_df["population"] == population
                pop_sample_idxs = data_set.phenotype.annotation_df.iloc[pop_mask].index.values

                # Split indices for this population
                pop_idxs = np.arange(len(pop_sample_idxs))
                if shuffle:
                    rng.shuffle(pop_idxs)

                split_idx = int(len(pop_idxs) * (1 - test_size))
                pop_train_idxs.extend(pop_sample_idxs[pop_idxs[:split_idx]])
                pop_test_idxs.extend(pop_sample_idxs[pop_idxs[split_idx:]])

            train_sample_idxs = np.array(pop_train_idxs, dtype=np.uint32)
            test_sample_idxs = np.array(pop_test_idxs, dtype=np.uint32)
        else:
            # Split all samples directly
            all_sample_idxs = data_set.phenotype.sample_idxs
            sample_idxs = np.arange(len(all_sample_idxs))

            if shuffle:
                rng.shuffle(sample_idxs)

            split_idx = int(len(sample_idxs) * (1 - test_size))
            train_sample_idxs = all_sample_idxs[sample_idxs[:split_idx]]
            test_sample_idxs = all_sample_idxs[sample_idxs[split_idx:]]

        # Create train set
        train_pheno_annotation_df = data_set.phenotype.annotation_df.loc[train_sample_idxs].copy()

        train_phenotype = PhenoSet(
            annotation_df=train_pheno_annotation_df,
            covar_cfg=data_set.cfg.covar_config,
        )

        train_genotype = data_set.genotype.fork(
            annotation_df=data_set.genotype.annotation_df.copy(),
            n_samples=len(train_pheno_annotation_df),
        )

        train_set = DataSet(cfg=data_set.cfg, genotype=train_genotype, phenotype=train_phenotype)

        # Create test set
        test_pheno_annotation_df = data_set.phenotype.annotation_df.loc[test_sample_idxs].copy()

        test_phenotype = PhenoSet(
            annotation_df=test_pheno_annotation_df, covar_cfg=data_set.cfg.covar_config
        )

        test_genotype = data_set.genotype.fork(
            annotation_df=data_set.genotype.annotation_df.copy(),
            n_samples=len(test_pheno_annotation_df),
        )

        test_cfg = copy.deepcopy(data_set.cfg)
        test_cfg.is_train = False
        test_set = DataSet(cfg=test_cfg, genotype=test_genotype, phenotype=test_phenotype)
        nottrain_set = [test_set]

    return (train_set, *nottrain_set)
