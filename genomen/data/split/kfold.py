from typing import List, Optional

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import KFold, StratifiedKFold

from ..data_set.data_set import DataBatch, DataSet
from ..data_set.geno_set import GenoSet
from ..data_set.pheno_set import PhenoSet


def kfold(
    data_set: DataSet | DataBatch,
    cv: int = 5,
    shuffle: bool = False,
    seed: int | None = None,
    stratified: bool = False,
    return_oof_idxs=False,
) -> List[DataSet | DataBatch] | Optional[List[npt.NDArray]]:
    if stratified:
        kf = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=seed)
    else:
        kf = KFold(n_splits=cv, shuffle=shuffle, random_state=seed)

    data_splits: List[DataSet | DataBatch] = []
    oof_idxs: List[npt.NDArray] = []
    if isinstance(data_set, DataSet):
        # Create an array of indices corresponding to rows
        sample_indices = np.arange(len(data_set.phenotype.annotation_df))
        split_idxs = kf.split(sample_indices, data_set.phenotype.y)

        for train_idxs, test_idxs in split_idxs:
            # Create train set - use the actual indices directly
            train_pheno_annotation_df = data_set.phenotype.annotation_df.iloc[train_idxs].copy()

            train_phenotype = PhenoSet(
                annotation_df=train_pheno_annotation_df,
                covar_cfg=data_set.cfg.covar_config,
            )

            train_genotype = data_set.genotype.fork(
                annotation_df=data_set.genotype.annotation_df,
                n_samples=len(train_pheno_annotation_df),
            )

            train_set = DataSet(
                cfg=data_set.cfg, genotype=train_genotype, phenotype=train_phenotype
            )

            # Create test set - use the actual indices directly
            test_pheno_annotation_df = data_set.phenotype.annotation_df.iloc[test_idxs].copy()

            test_phenotype = PhenoSet(
                annotation_df=test_pheno_annotation_df,
                covar_cfg=data_set.cfg.covar_config,
            )

            test_genotype = data_set.genotype.fork(
                annotation_df=data_set.genotype.annotation_df,
                n_samples=len(test_pheno_annotation_df),
            )

            test_set = DataSet(cfg=data_set.cfg, genotype=test_genotype, phenotype=test_phenotype)

            data_splits.append((train_set, test_set))
            oof_idxs.append(test_idxs)
    elif isinstance(data_set, DataBatch):
        # Create an array of indices corresponding to rows
        sample_indices = np.arange(len(data_set.pheno_annotation))
        split_idxs = kf.split(sample_indices, data_set.y)

        for train_idxs, test_idxs in split_idxs:
            # Create train set - use the actual indices directly
            train_pheno_annotation_df = data_set.pheno_annotation.iloc[train_idxs].copy()
            y_train = data_set.y[train_idxs]
            resid_train = data_set.residuals[train_idxs] if data_set.residuals is not None else None
            X_train = data_set.X[train_idxs]

            train_batch = DataBatch(
                cfg=data_set.cfg,
                X=X_train,
                geno_annotation=data_set.geno_annotation,
                y=y_train,
                pheno_annotation=train_pheno_annotation_df,
                residuals=resid_train,
                scaler=data_set.scaler,
                type=data_set.type,
            )

            # Create train set - use the actual indices directly
            test_pheno_annotation_df = data_set.pheno_annotation.iloc[test_idxs].copy()
            y_test = data_set.y[test_idxs]
            resid_test = data_set.residuals[test_idxs] if data_set.residuals is not None else None
            X_test = data_set.X[test_idxs]

            test_batch = DataBatch(
                cfg=data_set.cfg,
                X=X_test,
                geno_annotation=data_set.geno_annotation,
                y=y_test,
                pheno_annotation=test_pheno_annotation_df,
                residuals=resid_test,
                scaler=data_set.scaler,
                type=data_set.type,
            )

            data_splits.append((train_batch, test_batch))
            oof_idxs.append(test_idxs)
    else:
        raise TypeError("kfold only supports DataBatch and DataSet as input")

    if return_oof_idxs:
        return data_splits, oof_idxs
    else:
        return data_splits
