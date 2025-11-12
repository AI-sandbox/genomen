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


def main(cfg_path: str = "config.yml"):
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

        logger.info(
            "Computing local shap values"
        )  # cases (up to 2k) for binary, subset of 2k samples for cont.
        test_local_shap_df = model.geno_model.compute_local_shap(test_set, n_samples=2_000)

        local_shap_path = Path(model.cfg.variant_importance_dir, "test_local_shap.parquet")
        test_local_shap_df.to_parquet(local_shap_path)
        logger.info(f"Saved local shap values computed on cases in test set to: {local_shap_path}")


if __name__ == "__main__":
    Fire(main)
