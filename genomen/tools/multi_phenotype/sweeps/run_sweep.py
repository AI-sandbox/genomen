import logging
import os

import fire
import wandb
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


def get_phenotype_name(task_id):
    """Get phenotype name from task_id using the sweeps.yml config."""
    base_config_path = "genomen/tools/multi_phenotype/sweeps/sweeps.yml"
    with open(base_config_path, "r") as f:
        config = yaml.safe_load_all(f)
        config = list(config)

    phenotype_name = config[3]["BenchmarkConfig"]["phenotypes"][task_id - 1]
    return phenotype_name


def run_sweep(task_id=int | None, model_type: str = "lightgbm", project_name: str = "MetaPRS"):
    """Create and run a sweep with phenotype name in the sweep name."""
    # Get task_id from environment if not provided
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))

    # Get phenotype name from task_id using the sweeps.yml config
    phenotype_name = get_phenotype_name(task_id)

    # Choose config file based on model type
    config_file = f"genomen/tools/multi_phenotype/sweeps/sweep_configs/{model_type}.yml"
    with open(config_file, "r") as f:
        sweep_config = yaml.safe_load(f)

    # Update sweep name to include phenotype
    sweep_config["name"] = f"genomen_{model_type}_{phenotype_name}_sweep"
    logger.info(f"Creating sweep for phenotype: {phenotype_name} with model type: {model_type}")

    # Initialize wandb and create sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    logger.info(f"Created sweep with ID: {sweep_id}")

    # Run the agent
    wandb.agent(sweep_id=sweep_id, project=project_name, count=50)
    logger.info("Sweep completed")


if __name__ == "__main__":
    fire.Fire(run_sweep)
