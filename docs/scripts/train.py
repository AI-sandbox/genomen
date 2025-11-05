import logging

import pandas as pd
from fire import Fire

import genomen.utils as utils
from genomen.data import DataSet, split
from genomen.model import GenomenModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(cfg_path: str = "config.yml"):
    utils.set_config_path(cfg_path)

    logger.info("Initiate data set...")
    dataset = DataSet()
    train_set, test_set, val_set = split(dataset, split_by_col=("split", ("train", "test", "val")))
    print(f"Got {len(train_set)} samples in train set")

    logger.info("Train Model...")
    model = GenomenModel()
    model.fit(train_set, val_set)

    logger.info("Done with training. Predicting on test set...")
    geno_preds, covar_preds, preds = model.predict(test_set)

    geno_score = utils.score(
        test_set.get_labels(),
        geno_preds,
        model.train_cfg.scorer,
        test_set.cfg.classification,
    )
    logger.info(f"Geno-only score on test set: {geno_score:.4f}")
    if train_set.cfg.covar_config.include_covars:
        covar_score = utils.score(
            test_set.get_labels(),
            covar_preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        logger.info(f"Covar score on test set: {covar_score:.4f}")
        combined_score = utils.score(
            test_set.get_labels(),
            preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        logger.info(
            f"Combined ({model.cfg.covar_config.covar_strat}) score on test set: {combined_score:.4f}"
        )

    if test_set.cfg.classification:
        logger.info("Computing local shap values for all cases")
        # train
        train_local_shap_df = model.geno_model.compute_local_shap(
            train_set, sample_idxs=train_set.phenotype.sample_idxs[train_set.get_labels() == 1]
        )
        # val
        val_local_shap_df = model.geno_model.compute_local_shap(
            val_set, sample_idxs=val_set.phenotype.sample_idxs[val_set.get_labels() == 1]
        )
        # test
        test_local_shap_df = model.geno_model.compute_local_shap(
            test_set, sample_idxs=test_set.phenotype.sample_idxs[test_set.get_labels() == 1]
        )
        local_shap_df = pd.concat(
            [train_local_shap_df, val_local_shap_df, test_local_shap_df], axis=0
        )

        local_shap_path = f"local_shap_{train_set.cfg.phenotype_id}.parquet"
        logger.info(f"Saving local shap values to: {local_shap_path}")
        local_shap_df.to_parquet(local_shap_path)


if __name__ == "__main__":
    Fire(main)
