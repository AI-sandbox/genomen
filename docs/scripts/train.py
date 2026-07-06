import logging
from pathlib import Path

import pandas as pd
from fire import Fire

import genomen.utils as utils
from genomen.data import DataSet, split
from genomen.model import GenomenModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(cfg_path: str = "config.yml", per_type_scores: bool = True):
    utils.set_config_path(cfg_path)

    logger.info("Initiate data set...")
    dataset = DataSet()
    train_set, test_set, val_set = split(dataset, split_by_col=("split", ("train", "test", "val")))
    logger.info(
        f"Got {len(train_set)} samples in train set and {len(val_set)} samples in validation set."
    )

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
    if per_type_scores and model.per_type_geno_preds is not None:
        for type_name, type_preds in model.per_type_geno_preds.items():
            type_score = utils.score(
                test_set.get_labels(),
                type_preds,
                model.train_cfg.scorer,
                test_set.cfg.classification,
            )
            logger.info(f"Geno-only score [{type_name}] on test set: {type_score:.4f}")

    if train_set.cfg.covar_config.include_covars:
        covar_score = utils.score(
            test_set.get_labels(),
            covar_preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        logger.info(f"Covar-only score on test set: {covar_score:.4f}")
        combined_score = utils.score(
            test_set.get_labels(),
            preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        logger.info(
            f"Combined ({model.cfg.covar_config.covar_strat}) score on test set: {combined_score:.4f}"
        )


if __name__ == "__main__":
    Fire(main)
