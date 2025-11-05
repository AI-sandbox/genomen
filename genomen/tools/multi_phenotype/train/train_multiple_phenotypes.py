"""Script to evaluate the performance of Genomen on phenotypes."""

import logging
import os
import tempfile
import uuid
from typing import Literal

import fire
import numpy as np
import pandas as pd
import yaml

import genomen.utils as utils
import wandb
from genomen.data import DataSet, split
from genomen.model import GenomenModel
from genomen.tools.multi_phenotype.phenotype_config import PHENOTYPES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _sample_idxs(all_sample_idxs, n=2000, seed=42):
    all_sample_idxs = np.asarray(all_sample_idxs)
    if all_sample_idxs.size <= n:
        return all_sample_idxs
    rng = np.random.default_rng(seed)
    return rng.choice(all_sample_idxs, size=n, replace=False)


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
        and configs[1]["GenomenModelConfig"]["covar_config"]["covar_strat"]
        == "residualization"
    )
    configs[1]["GenomenModelConfig"]["geno_config"]["preprocessing_config"][
        "feature_selection"
    ].update(
        {
            "score_func": "chi2"
            if phenotype["classification"] and not use_resid
            else "f_regression"
        }
    )

    configs[2]["TrainConfig"].update(
        {"scorer": "rocauc" if phenotype["classification"] else "r2"}
    )

    logger.info(configs)

    # Create temporary config file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as temp_config:
        yaml.safe_dump_all(configs, temp_config)
        temp_config_path = temp_config.name

    # Set config path
    utils.set_config_path(temp_config_path)

    # Initialize dataloader
    logger.info("Initiate data set...")
    dataset = DataSet()
    train_set, test_set, val_set = split(
        dataset, split_by_col=("split", ("train", "test", "val"))
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
    wandb.log({"geno_test_score": geno_score})

    if train_set.cfg.covar_config.include_covars:
        covar_score = utils.score(
            test_set.get_labels(),
            covar_preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        logger.info(
            f"covar ({model.cfg.covar_config.covar_strat}) score on test set: {covar_score:.4f}"
        )
        wandb.log({"covar_test_score": covar_score})

        combined_score = utils.score(
            test_set.get_labels(),
            preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        logger.info(
            f"Combined ({model.cfg.covar_config.covar_strat}) score on test set: {combined_score:.4f}"
        )
        wandb.log({"covar_geno_test_score": combined_score})

    if compute_local_shap:
        if test_set.cfg.classification:
            logger.info("Computing local shap values for all positive cases (classification)")
            # test
            test_local_shap_df = model.geno_model.compute_local_shap(
                test_set, 
                sample_idxs=test_set.phenotype.sample_idxs[test_set.get_labels() == 1]
            )
            test_local_shap_df["pred"] = geno_preds[test_set.get_labels() == 1]
            test_local_shap_df["split"] = "test"
        else:
            test_all = test_set.phenotype.sample_idxs
            test_sample = _sample_idxs(test_all, n=2000, seed=42)
            test_local_shap_df = model.geno_model.compute_local_shap(test_set, sample_idxs=test_sample)
            test_preds, _, _ = model.predict(test_set)
            test_preds_s = pd.Series(test_preds, index=test_all)
            test_local_shap_df["pred"] = test_preds_s.reindex(test_local_shap_df.index).values
            test_local_shap_df["split"] = "test"

        uid = str(uuid.uuid4())[:8]
        out_path = f"local_shap_{train_set.cfg.phenotype_id}_{uid}.parquet"
        test_local_shap_df.to_parquet(out_path)
        logger.info(f"Saved SHAP values to: {out_path}")

    # Clean up temporary file
    os.unlink(temp_config_path)


def main(
    task_id: int,
    use_phenotype_config: bool = False,
    covar_strat: Literal["residualization", "predictive"] = "residualization",
    variant_sampling_strat: Literal["random", "LD", "MAB", "gene", "GWAS"] | None = "random",
    sample_sampling_strat: Literal["random", "stratify", "balanced"] | None = None,
    n_estimators: int | None = 32,
    max_samples: Literal["all"] | int | None = 50_000,
    max_features: int | None = None,
    feature_selection: Literal["k_best", "mutual_info", "none"] | None = None,
    k: int | None = None,
    balance_k: int | None = None,
    batch_size: int | None = 2, 
    impute_val: float | None = None,
    maf: float | None = None,
    eps: float | None = None,
    sex: Literal["m", "w"] | None = None,
    eps_schedule: Literal["step", "constant"] | None = None,
    filter_strat: str | None = None,
    agg_strat_bin: str | None = None,
    agg_strat_cont: str | None = None,
    n_per_block: int | None = None,
    use_ss: bool | None = None,
    window_overlap_ratio: float | None = None,
    agg_model: Literal["linear", "bayesian", "linear_l2"] | None = None,
    include_x: bool | None = False,
    geno_model: Literal["linear", "xgb", "lgbm", "mlp"] | None = "mlp",
    compute_global_shap: bool | None = False,
    compute_local_shap: bool = False,
    compute_interactions: bool | None = False,
    n_jobs: int | None = 32,
    backend: Literal["gpu", "cpu"] = "cpu"
):
    # Get phenotype id
    with open("genomen/tools/multi_phenotype/train/configs/config.yml", "r") as f:
        configs = list(yaml.load_all(f, Loader=yaml.SafeLoader))

    benchmark_config = configs[3]["BenchmarkConfig"]

    # Get phenotype from config based on task_id
    phenotype_name = benchmark_config["phenotypes"][
        task_id - 1
    ]  # Convert to 0-based index
    phenotype = PHENOTYPES[phenotype_name]

    # Load config
    if use_phenotype_config:
        config_path = (
            f"genomen/tools/multi_phenotype/train/configs/{phenotype_name}_config.yml"
        )
        with open(config_path, "r") as f:
            configs = list(yaml.load_all(f, Loader=yaml.SafeLoader))

        # Overwrite config values with provided parameters
        # covar_strat
        if covar_strat is not None:
            configs[1]["GenomenModelConfig"]["covar_config"]["covar_strat"] = covar_strat
        # variant_sampling_strat
        if variant_sampling_strat is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["strat"] = variant_sampling_strat
            if variant_sampling_strat == "window":
                configs[0]["DataSetConfig"]["variant_sampling"]["window_overlap_ratio"] = window_overlap_ratio
            if (variant_sampling_strat == "GWAS") and (impute_val is not None):
                configs[0]["DataSetConfig"]["variant_sampling"]["gwas_config"]["impute_val"] = impute_val
        # sample_sampling_strat
        if sample_sampling_strat is not None:
            configs[0]["DataSetConfig"]["sample_sampling"]["strat"] = sample_sampling_strat
        if sex is not None:
            configs[0]["DataSetConfig"]["sex"] = sex
        # balance k
        if balance_k is not None:
            configs[0]["DataSetConfig"]["sample_sampling"]["k"] = balance_k
        # eps
        if eps is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["ld_config"]["eps"] = eps
        # eps_schedule
        if eps_schedule is not None:
            configs[0]["DataSetConfig"]["variant_sampling"]["ld_config"]["eps_schedule"] = eps_schedule
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
            configs[1]["GenomenModelConfig"]["geno_config"]["preprocessing_config"]["feature_selection"]["method"] = feature_selection
        if k is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["preprocessing_config"]["feature_selection"]["k"] = k
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
                configs[1]["GenomenModelConfig"]["geno_config"]["compute_interactions"] = compute_interactions
        # maf
        if maf is not None:
            configs[0]["DataSetConfig"]["maf_threshold"] = maf
        # include x
        if include_x is not None:
            configs[0]["DataSetConfig"]["include_x_chromosome"] = include_x
        # agg_model
        if agg_model is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"]["model_config"]["model_name"] = agg_model
        # use_ss
        if use_ss is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"]["use_summary_stats"] = use_ss
        # filter strat
        if filter_strat is not None:
            configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"]["filter_strat"] = filter_strat
        # agg_strat_bin/agg_strat_cont
        if phenotype["classification"]:
            if agg_strat_bin is not None:
                configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"]["agg_strat"] = agg_strat_bin
        else:
            if agg_strat_cont is not None:
                configs[1]["GenomenModelConfig"]["geno_config"]["aggregator_config"]["agg_strat"] = agg_strat_cont
        if geno_model == "linear":
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["model_name"] = "linear_l1"
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["hyperparameters"] = {"alpha": 0.01}
        elif geno_model == "xgb":
            xgb_hyperparams = configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["hyperparameters"].get("xgboost", None) or {          
                "booster": "gbtree",
                "colsample_bytree": 0.8946650565969123,
                "learning_rate": 0.060620701769366465,
                "max_depth": 9,
                "n_estimators": 1660,
                "reg_alpha": 0.6656256452959488,
                "reg_lambda": 0.12183231460398992,
                "subsample": 0.8841736709630683,
                "tree_method": "hist"
            }
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["model_name"] = "xgboost"
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["hyperparameters"] = xgb_hyperparams
        elif geno_model == "lgbm":
            lgb_hyperparams = configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["hyperparameters"].get("lightgbm", None) or {          
                "bagging_fraction": 0.6232546393535136,
                "bagging_freq": 1,
                "feature_fraction": 0.7568699255535379,
                "learning_rate": 0.020426221815062907,
                "max_bin": 289,
                "max_depth": 14,
                "min_child_samples": 13,
                "learning_rate": 0.020426221815062907,
                "max_bin": 289,
                "max_depth": 14,
                "min_child_samples": 13,
                "min_child_weight": 5.174997310706651,
                "min_data_in_leaf": 286,
                "min_gain_to_split": 1.7960224741869382,
                "n_estimators": 1500,
                "num_leaves": 180,
                "path_smooth": 40.095909854952,
            }
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["model_name"] = "lightgbm"
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["hyperparameters"] = lgb_hyperparams
        elif geno_model == "mlp":
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["model_name"] = "simple_mlp"
            configs[1]["GenomenModelConfig"]["geno_config"]["model_config"]["hyperparameters"] = {}

    logger.info(f"Processing phenotype: {phenotype_name}")

    # Run Genomen
    run_genomen(configs=configs, phenotype=phenotype, compute_local_shap=compute_local_shap)


if __name__ == "__main__":
    logger.info("Starting Genomen evaluation...")
    fire.Fire(main)
    logger.info("Genomen evaluation complete.")
