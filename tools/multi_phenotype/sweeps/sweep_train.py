import logging
import os
import tempfile

import fire
import wandb
import yaml

import genomen.utils as utils
from genomen.data import DataSet, split
from genomen.model import GenomenModel
from tools.multi_phenotype.phenotype_config import PHENOTYPES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def update_config_with_wandb_params(wandb_config, task_id):
    """Update the base sweep config with wandb-sampled hyperparameters."""
    base_config_path = "tools/multi_phenotype/sweeps/sweeps.yml"
    with open(base_config_path, "r") as f:
        config = list(yaml.safe_load_all(f))

    phenotype_name = config[3]["BenchmarkConfig"]["phenotypes"][
        task_id - 1
    ]  # Convert to 0-based index
    phenotype = PHENOTYPES[phenotype_name]

    # --- DataSetConfig ---
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
    data_set_config["variant_sampling"] = {
        "strat": wandb_config.variant_sampling_strategy,
        "max_features": wandb_config.max_features,
    }

    # --- GenomenModelConfig ---
    genomen_model_config = config[1]["GenomenModelConfig"]
    genomen_model_config["covar_config"]["model_config"]["model_name"] = (
        wandb_config.covar_model_name
    )
    genomen_model_config["geno_config"]["model_config"]["model_name"] = (
        wandb_config.geno_model_name
    )

    if wandb_config.geno_model_name == "ensemble":
        ensemble_estimator_names = wandb_config.ensemble_estimator_names
        genomen_model_config["geno_config"]["model_config"]["ensemble_estimator_names"] = (
            ensemble_estimator_names
        )
        geno_model_hyperparams = {
            model_name: wandb_config.geno_model_hyperparameters.get(model_name, {})
            for model_name in ensemble_estimator_names
        }
    else:
        # For single models
        geno_model_hyperparams = wandb_config.geno_model_hyperparameters.get(
            wandb_config.geno_model_name, {}
        )
    genomen_model_config["geno_config"]["model_config"]["hyperparameters"] = geno_model_hyperparams

    genomen_model_config["geno_config"]["aggregator_config"] = {
        "filter_strat": wandb_config.filter_strat,
        "agg_strat": wandb_config.agg_strat,
        "p": wandb_config.p,
    }

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

    # --- TrainConfig ---
    train_config = config[2]["TrainConfig"]
    train_config.update(
        {
            "scorer": "rocauc" if phenotype["classification"] else "r2",
            "backend": getattr(wandb_config, "backend", "cpu"),
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

    classification = test_set.cfg.classification
    include_covars = train_set.cfg.covar_config.include_covars

    def score_split(name: str, data_set) -> float:
        """Score genotype-only, covar-only, and combined predictions on a split.

        Returns the primary prediction metric for that split (combined score if
        covariates are included, geno-only score otherwise).
        """
        geno_preds, covar_preds, preds = model.predict(data_set)

        geno_score = utils.score(
            data_set.get_labels(), geno_preds, model.train_cfg.scorer, classification
        )
        run.log({f"{name}_geno_score": geno_score})
        logger.info(f"Geno-only score on {name} set: {geno_score}")

        if include_covars:
            covar_score = utils.score(
                data_set.get_labels(), covar_preds, model.train_cfg.scorer, classification
            )
            combined_score = utils.score(
                data_set.get_labels(), preds, model.train_cfg.scorer, classification
            )
            run.log({f"{name}_covar_score": covar_score, f"{name}_combined_score": combined_score})
            logger.info(f"Covar score on {name} set: {covar_score}")
            logger.info(f"Combined score on {name} set: {combined_score}")
            pred_metric = combined_score
        else:
            pred_metric = geno_score

        run.log({f"{name}_pred_metric": pred_metric})
        return pred_metric

    # The sweep's objective (val_pred_metric) is computed on the validation split, never on
    # test -- test scores are logged purely for held-out reporting and must not feed the search.
    logger.info("Evaluating model on validation set...")
    score_split("val", val_set)

    logger.info("Evaluating model on test set...")
    score_split("test", test_set)

    # Cleanup
    os.remove(config_path)
    run.finish()


if __name__ == "__main__":
    fire.Fire(train)
