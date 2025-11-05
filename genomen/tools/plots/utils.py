import logging
import tempfile

import yaml

import genomen.utils as utils
from genomen.tools.multi_phenotype.phenotype_config import PHENOTYPES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup(task_id: int, dummy: bool = False):
    with open("genomen/tools/multi_phenotype/train/configs/config.yml", "r") as f:
        configs = list(yaml.load_all(f, Loader=yaml.SafeLoader))

    benchmark_config = configs[3]["BenchmarkConfig"]

    # Get phenotype from config based on task_id
    phenotype_name = benchmark_config["phenotypes"][
        task_id - 1
    ]

    config_path = (
        f"genomen/tools/multi_phenotype/train/configs/{phenotype_name}_config.yml"
    )

    if dummy:
        phenotype_name = "standing_height"
        config_path = "config.yml.template"

    with open(config_path, "r") as f:
        configs = list(yaml.load_all(f, Loader=yaml.SafeLoader))

    phenotype = PHENOTYPES[phenotype_name]
    phenotype_id = phenotype["id"]
    configs[0]["DataSetConfig"].update(
        {
            "phenotype_id": phenotype_id,
            "classification": phenotype["classification"],
        }
    )

    configs[2]["TrainConfig"]["backend"] = "cpu"

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

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as temp_config:
        yaml.safe_dump_all(configs, temp_config)
        temp_config_path = temp_config.name

    utils.set_config_path(temp_config_path)

    return phenotype_name, phenotype_id