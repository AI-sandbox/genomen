"""Script to evaluate the performance of Genomen on phenotypes."""

import logging
import os
import tempfile
from typing import Literal

import fire
import numpy as np
import yaml

import genomen.utils as utils
from genomen.data import DataSet, bootstrap, split
from genomen.model import GenomenModel
from genomen.tools.multi_phenotype.phenotype_config import PHENOTYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BOOTSTRAP_N = 50


def run_genomen(configs, phenotype, compute_local_shap):
    """Run Genomen for a specific phenotype configuration."""
    logger.info(f"Running Genomen for phenotype: {phenotype['id']}")
    # Update configs with parameters
    configs[0]["DataSetConfig"].update(
        {
            "phenotype_id": phenotype["id"],
            "classification": phenotype["classification"],
        }
    )

    use_resid = (
        configs[0]["DataSetConfig"]["covar_config"]["include_covars"]
        and configs[1]["GenomenModelConfig"]["covar_config"]["covar_strat"] == "residualization"
    )
    configs[1]["GenomenModelConfig"]["geno_config"]["preprocessing_config"][
        "feature_selection"
    ].update(
        {"score_func": "chi2" if phenotype["classification"] and not use_resid else "f_regression"}
    )

    configs[2]["TrainConfig"].update({"scorer": "rocauc" if phenotype["classification"] else "r2"})

    logger.info(configs)

    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as temp_config:
        yaml.safe_dump_all(configs, temp_config)
        temp_config_path = temp_config.name

    # Set config path
    utils.set_config_path(temp_config_path)

    # Initialize dataloader
    logger.info("Initiate data set...")
    dataset = DataSet()
    train_set, test_set, val_set = split(dataset, split_by_col=("split", ("train", "test", "val")))

    geno_scores = []
    covar_scores = []
    combined_scores = []
    for i in range(BOOTSTRAP_N):
        bootstrapped_set = bootstrap(train_set)
        logger.info(f"Training model {i+1} of {BOOTSTRAP_N}...")
        model = GenomenModel()
        model.fit(bootstrapped_set, val_set)

        logger.info("Done with training. Predicting on test set...")
        geno_preds, covar_preds, preds = model.predict(test_set)

        geno_score = utils.score(
            test_set.get_labels(),
            geno_preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        geno_scores.append(geno_score)
        logger.info(f"Geno-only score on test set: {geno_score:.4f}")

        if train_set.cfg.covar_config.include_covars:
            covar_score = utils.score(
                test_set.get_labels(),
                covar_preds,
                model.train_cfg.scorer,
                test_set.cfg.classification,
            )
            covar_scores.append(covar_score)
            logger.info(
                f"covar ({model.cfg.covar_config.covar_strat}) score on test set: {covar_score:.4f}"
            )

            combined_score = utils.score(
                test_set.get_labels(),
                preds,
                model.train_cfg.scorer,
                test_set.cfg.classification,
            )
            logger.info(
                f"Combined ({model.cfg.covar_config.covar_strat}) score on test set: {combined_score:.4f}"
            )
            combined_scores.append(combined_score)
        logger.info(
            f"Geno-only mean score on test set: {np.mean(geno_scores):.4f} ± {np.std(geno_scores):.4f}"
        )
        logger.info(
            f"Covar-only mean score on test set: {np.mean(covar_scores):.4f} ± {np.std(covar_scores):.4f}"
        )
        logger.info(
            f"Combined mean score on test set: {np.mean(combined_scores):.4f} ± {np.std(combined_scores):.4f}"
        )

    # Clean up temporary file
    os.unlink(temp_config_path)


def main(
    task_id: int,
    use_phenotype_config: bool = False,
    covar_strat: Literal["residualization", "predictive"] = "residualization",
    variant_sampling_strat: Literal["random", "LD", "MAB", "gene", "GWAS"] | None = "random",
    sample_sampling_strat: Literal["random", "stratify", "balanced"] | None = None,
    n_estimators: int | None = 32,
    max_samples: Literal["all"] | int | None = None,
    max_features: int | None = None,
    feature_selection: Literal["k_best", "mutual_info", "none"] | None = None,
    k: int | None = None,
    balance_k: int | None = None,
    batch_size: int | None = 2,
    maf: float | None = None,
    eps: float | None = None,
    eps_schedule: Literal["step", "constant"] | None = None,
    filter_strat: str | None = None,
    agg_strat_bin: str | None = None,
    agg_strat_cont: str | None = None,
    n_per_block: int | None = None,
    use_ss: bool | None = None,
    agg_model: Literal["linear", "bayesian", "linear_l2"] | None = None,
    include_x: bool | None = False,
    linear_geno_model: bool = False,
    compute_global_shap: bool | None = True,
    compute_local_shap: bool = False,
    compute_interactions: bool | None = False,
    n_jobs: int | None = 32,
    backend: Literal["gpu", "cpu"] = "cpu",
):
    # Get phenotype id
    with open("genomen/tools/multi_phenotype/train/configs/config.yml", "r") as f:
        configs = list(yaml.load_all(f, Loader=yaml.SafeLoader))

    benchmark_config = configs[3]["BenchmarkConfig"]

    # Get phenotype from config based on task_id
    phenotype_name = benchmark_config["phenotypes"][task_id - 1]  # Convert to 0-based index
    phenotype = PHENOTYPES[phenotype_name]

    # Load config
    if use_phenotype_config:
        config_path = f"genomen/tools/multi_phenotype/train/configs/{phenotype_name}_config.yml"
        with open(config_path, "r") as f:
            configs = list(yaml.load_all(f, Loader=yaml.SafeLoader))

        # Overwrite config values with provided parameters
        # covar_strat
        if covar_strat is not None:
            configs[1]["GenomenModelConfig"]["covar_config"]["covar_strat"] = covar_strat
        # variant_sampling_strat
        if variant_sampling_strat is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["strat"] = variant_sampling_strat
        # sample_sampling_strat
        if sample_sampling_strat is not None:
            configs[0]["DataSetConfig"]["sample_sampling"]["strat"] = sample_sampling_strat
        # balance k
        if balance_k is not None:
            configs[0]["DataSetConfig"]["sample_sampling"]["k"] = balance_k
        # eps
        if eps is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["ld_config"]["eps"] = eps
        # eps_schedule
        if eps_schedule is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["ld_config"][
                "eps_schedule"
            ] = eps_schedule
        # n_per_block
        if n_per_block is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["n_per_block"] = n_per_block
        # n_estimators
        if n_estimators is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["n_estimators"] = n_estimators
        # max_samples
        if max_samples is not None:
            if max_samples == "all":
                configs[0]["DataSetConfig"]["sample_sampling"]["max_samples"] = None
            configs[0]["DataSetConfig"]["sample_sampling"]["max_samples"] = max_samples
        # max_features
        if max_features is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["max_features"] = max_features
        # feature seleciton
        if feature_selection is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["preprocessing_config"][
                "feature_selection"
            ]["method"] = feature_selection
        if k is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["preprocessing_config"][
                "feature_selection"
            ]["k"] = k
        # batch_size
        if batch_size is not None:
            configs[2]["TrainConfig"]["batch_size"] = batch_size
        # backend
        if backend is not None:
            configs[2]["TrainConfig"]["backend"] = backend
        # n_jobs
        if backend is not None:
            configs[2]["TrainConfig"]["n_jobs"] = n_jobs
        # backend
        if compute_global_shap is not None:
            configs[2]["TrainConfig"]["compute_shap"] = compute_global_shap
            configs[2]["TrainConfig"]["save_annotation"] = True
            if compute_interactions is not None:
                configs[1]["GenomenModelConfig"]["geno_config"][
                    "compute_interactions"
                ] = compute_interactions
        # maf
        if maf is not None:
            configs[0]["DataSetConfig"]["maf_threshold"] = maf
        # include x
        if include_x is not None:
            configs[0]["DataSetConfig"]["include_x_chromosome"] = include_x
        # agg_model
        if agg_model is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"]["model_config"][
                "model_name"
            ] = agg_model
        # use_ss
        if use_ss is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"][
                "use_summary_stats"
            ] = use_ss
        # filter strat
        if filter_strat is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"][
                "filter_strat"
            ] = filter_strat
        # agg_strat_bin/agg_strat_cont
        if phenotype["classification"]:
            if agg_strat_bin is not None:
                configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"][
                    "agg_strat"
                ] = agg_strat_bin
        else:
            if agg_strat_cont is not None:
                configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"][
                    "agg_strat"
                ] = agg_strat_cont
        if linear_geno_model:
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"][
                "model_name"
            ] = "linear_l1"
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["hyperparameters"] = {
                "alpha": 0.01
            }

    logger.info(f"Processing phenotype: {phenotype_name}")

    # Run Genomen
    run_genomen(configs=configs, phenotype=phenotype, compute_local_shap=compute_local_shap)


if __name__ == "__main__":
    logger.info("Starting Genomen evaluation...")
    fire.Fire(main)
    logger.info("Genomen evaluation complete.")
