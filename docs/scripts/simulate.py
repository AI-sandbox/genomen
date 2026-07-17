import json
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
from fire import Fire

import genomen.utils as utils
from genomen.data import DataSet
from genomen.data.simulations import simulate
from genomen.model import GenomenModel
from genomen.model.configs import TrainConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    cfg_path: str = "docs/configs/sim_binary.yml",
    task: Literal["cls", "reg"] = "cls",
    prevalence: float = 0.1,
    n_samples: int = 10_000,
    h_cov: float = 0.3,
    h_add: float = 0.3,
    h_epi: float = 0.3,
    frac_add_causal: float = 0.0001,
    n_epi_causal: float = 100,
    overlap_add_epi: float = 1.0,
    interaction_order: int = 2,
    max_interactions_per_snp: int = 2,
    seed: int = 0,
    epi_both_add: bool = True,
    require_diff_block: bool = False,
    require_non_add_block: bool = False,
    sim_seed: int | None = None,
    out_path: str = "artifacts/simulations",
    compute_shap: bool = False,
):
    utils.set_config_path(cfg_path)

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    sim_seed = sim_seed if sim_seed is not None else seed

    logger.info("Initiate data set...")
    dataset = DataSet()

    logger.info("Simulating phenotype...")
    train_set, val_set, test_set, sim_meta = simulate(
        dataset,
        task=task,
        prevalence=prevalence,
        n_samples=n_samples,
        h_cov=h_cov,
        h_add=h_add,
        h_epi=h_epi,
        frac_add_causal=frac_add_causal,
        n_epi_causal=n_epi_causal,
        overlap_add_epi=overlap_add_epi,
        interaction_order=interaction_order,
        epi_both_add=epi_both_add,
        require_diff_block=require_diff_block,
        require_non_add_block=require_non_add_block,
        max_interactions_per_snp=max_interactions_per_snp,
        seed=sim_seed,
    )

    logger.info("Training model on simulated data...")
    train_cfg = utils.init_class(
        cls=TrainConfig, classification=train_set.cfg.classification, seed=seed
    )
    model = GenomenModel()
    model.fit(train_set, val_set, train_cfg=train_cfg)

    logger.info("Done with training. Predicting on test set...")
    pred_result = model.predict(test_set, compute_shap=compute_shap)
    geno_preds, covar_preds, preds = pred_result[:3]
    shap_df = pred_result[3] if compute_shap else None

    geno_score = utils.score(
        test_set.get_labels(),
        geno_preds,
        model.train_cfg.scorer,
        test_set.cfg.classification,
    )
    logger.info(f"Geno-only score on test set: {geno_score:.4f}")
    scores = {"geno_test_score": geno_score}
    if model.per_type_geno_preds is not None:
        for type_name, type_preds in model.per_type_geno_preds.items():
            type_score = utils.score(
                test_set.get_labels(),
                type_preds,
                model.train_cfg.scorer,
                test_set.cfg.classification,
            )
            logger.info(f"Geno-only score [{type_name}] on test set: {type_score:.4f}")
            scores[f"geno_test_score_{type_name}"] = type_score

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
        scores["covar_test_score"] = covar_score
        scores["covar_geno_test_score"] = combined_score

    # Save test labels and predictions to out_path
    test_labels = test_set.get_labels()
    results = {
        "test_labels": test_labels,
        "geno_preds": geno_preds,
        "covar_preds": covar_preds,
        "preds": preds,
    }
    if model.per_type_geno_preds is not None:
        for type_name, type_preds in model.per_type_geno_preds.items():
            results[f"geno_preds_{type_name}"] = type_preds

    pred_path = out_path / "test_preds.npz"
    np.savez(pred_path, **results)
    logger.info(f"Saved test labels and predictions to: {pred_path}")

    if compute_shap and shap_df is not None:
        shap_path = out_path / "test_shap.parquet"
        shap_df.to_parquet(shap_path)
        logger.info(f"Saved SHAP matrix to: {shap_path}")

    meta_path = out_path / "sim_meta.pkl"
    with meta_path.open("wb") as f:
        pickle.dump(sim_meta, f)
    logger.info(f"Saved simulation metadata to: {meta_path}")

    scores_path = out_path / "scores.json"
    with scores_path.open("w") as f:
        json.dump(scores, f)
    logger.info(f"Saved scores to: {scores_path}")


if __name__ == "__main__":
    Fire(main)
