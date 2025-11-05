import logging
import os
import tempfile

import fire
import yaml

import genomen.utils as utils
import wandb
from genomen.data import DataSet, split
from genomen.model import GenomenModel
from genomen.tools.multi_phenotype.phenotype_config import PHENOTYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def update_config_with_wandb_params(wandb_config, task_id):
    """Update the config file with wandb parameters."""
    base_config_path = "genomen/tools/multi_phenotype/sweeps/sweeps.yml"
    with open(base_config_path, "r") as f:
        config = yaml.safe_load_all(f)
        config = list(config)

    phenotype_name = config[3]["BenchmarkConfig"]["phenotypes"][
        task_id - 1
    ]  # Convert to 0-based index
    phenotype = PHENOTYPES[phenotype_name]

    # Update DataSetConfig
    max_features = (
        wandb_config.max_features
        if wandb_config.feature_selection_method == "k_best"
        else wandb_config.max_features
    )

    data_set_config = config[0]["DataSetConfig"]
    data_set_config.update(
        {
            "phenotype_id": phenotype["id"],
            "classification": phenotype["classification"],
        }
    )
    data_set_config["sample_sampling"].update(
        {
            "strat": wandb_config.sample_sampling_strategy,
            "max_samples": wandb_config.max_samples,
            "k": wandb_config.k_balance,
        }
    )
    data_set_config["variant_sampling"].update(
        {"strat": wandb_config.variant_sampling_strategy, "max_features": max_features}
    )
    data_set_config["variant_sampling"]["ld_config"].update(
        {
            "eps": wandb_config.eps,
            "eps_schedule": wandb_config.eps_schedule,
        }
    )

    # Update GenomenModelConfig
    genomen_model_config = config[1]["GenomenModelConfig"]
    genomen_model_config["covar_config"]["model_config"][
        "model_name"
    ] = wandb_config.covar_model_name
    genomen_model_config["geno_config"]["model_config"]["model_name"] = wandb_config.geno_model_name
    if "ensemble_estimator_names" in wandb_config:
        genomen_model_config["geno_config"]["model_config"][
            "ensemble_estimator_names"
        ] = wandb_config.ensemble_estimator_names

    if wandb_config.geno_model_name == "ensemble":
        geno_model_hyperparams = {}

        # Process hyperparameters for each model in the ensemble
        for model_name in wandb_config.ensemble_estimator_names:
            geno_model_hyperparams[model_name] = wandb_config.geno_model_hyperparameters.get(
                model_name, {}
            )
    else:
        # For single models
        geno_model_hyperparams = wandb_config.geno_model_hyperparameters.get(
            wandb_config.geno_model_name, {}
        )
    genomen_model_config["geno_config"]["model_config"]["hyperparameters"] = geno_model_hyperparams

    genomen_model_config["geno_config"]["aggregator_config"].update(
        {
            "agg_strat": wandb_config.agg_mode,
            "use_summary_stats": wandb_config.include_ss,
            "filter_strat": wandb_config.filter_strat,
            "p": wandb_config.p,
        }
    )
    genomen_model_config["geno_config"]["aggregator_config"]["model_config"][
        "model_name"
    ] = wandb_config.aggregation_model_name

    # Add aggregator hyperparameters if available
    aggregator_hyperparams = {}
    if hasattr(wandb_config, "aggregation_model_hyperparameters") and hasattr(
        wandb_config.aggregation_model_hyperparameters,
        wandb_config.aggregation_model_name,
    ):
        aggregator_hyperparams = getattr(
            wandb_config.aggregation_model_hyperparameters,
            wandb_config.aggregation_model_name,
        )
        genomen_model_config["geno_config"]["aggregator_config"]["model_config"][
            "hyperparameters"
        ] = aggregator_hyperparams

    use_resid = (
        data_set_config["covar_config"]["include_covars"]
        and genomen_model_config["covar_config"]["covar_strat"] == "residualization"
    )
    genomen_model_config["geno_config"]["preprocessing_config"]["feature_selection"].update(
        {
            "method": wandb_config.feature_selection_method,
            "k": wandb_config.k_fs,
            "score_func": (
                "chi2" if phenotype["classification"] and not use_resid else "f_regression"
            ),
        }
    )

    # Update TrainConfig
    train_config = config[2]["TrainConfig"]
    train_config.update(
        {
            "scorer": "rocauc" if phenotype["classification"] else "r2",
            "backend": wandb_config.backend if hasattr(wandb_config, "backend") else "cpu",
        }
    )

    # Overwrite config with changes
    config = [
        {"DataSetConfig": data_set_config},
        {"GenomenModelConfig": genomen_model_config},
        {"TrainConfig": train_config},
    ]

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yml")
    yaml.safe_dump_all(config, temp_file)
    temp_file.close()

    return temp_file.name, phenotype_name


def train(task_id: int | None, model_type: str, project_name: str = "MetaPRS", **kwargs):
    """Main training function for sweep.

    Parameters can come from either direct arguments or wandb.config
    """
    # Initialize wandb
    run = wandb.init(project=project_name)

    # Extract task_id from wandb config if not provided as argument
    if task_id is None:
        task_id = getattr(wandb.config, "task_id", None)
        if task_id is None:
            # Use SLURM_ARRAY_TASK_ID if available
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID", 1)
        task_id = int(task_id)

    # Log task_id and model_type
    wandb.config.update({"task_id": task_id, "model_type": model_type}, allow_val_change=True)

    # Update config with wandb parameters and phenotype
    config_path, phenotype_name = update_config_with_wandb_params(wandb.config, task_id)

    # Set the config path
    utils.set_config_path(config_path)

    logger.info(
        f"Running sweep for task_id: {task_id} (phenotype: {phenotype_name}) with {model_type} model"
    )

    # Add phenotype to wandb config
    wandb.config.update(
        {"phenotype": phenotype_name, "model_type": model_type}, allow_val_change=True
    )

    # Initialize dataset
    logger.info("Initiating DataSet...")
    dataset = DataSet()
    train_set, test_set, val_set = split(dataset, split_by_col=("split", ("train", "test", "val")))

    # Initialize model
    logger.info("Training model...")
    model = GenomenModel()

    # Train model
    model.fit(train_set, val_set)

    # Evaluate and log results
    logger.info("Evaluating model...")
    geno_preds, covar_preds, preds = model.predict(test_set)

    # Score genotype-only predictions
    geno_score = utils.score(
        test_set.get_labels(),
        geno_preds,
        model.train_cfg.scorer,
        test_set.cfg.classification,
    )

    # Log to wandb
    run.log({"test_geno_score": geno_score})
    logger.info(f"Geno-only score on test set: {geno_score}")

    # Score combined predictions if covariates are included
    if train_set.cfg.covar_config.include_covars:
        covar_score = utils.score(
            test_set.get_labels(),
            covar_preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        run.log({"test_covar_score": covar_score})
        logger.info(f"Covar score on test set: {covar_score}")

        combined_score = utils.score(
            test_set.get_labels(),
            preds,
            model.train_cfg.scorer,
            test_set.cfg.classification,
        )
        run.log({"test_combined_score": combined_score})
        logger.info(f"Combined score on test set: {combined_score}")

        # Use combined score as primary metric for sweeps
        run.log({"test_pred_metric": combined_score})
    else:
        # Use geno score as primary metric if no covariates
        run.log({"test_pred_metric": geno_score})

    # Cleanup
    os.remove(config_path)
    run.finish()


if __name__ == "__main__":
    fire.Fire(train)
