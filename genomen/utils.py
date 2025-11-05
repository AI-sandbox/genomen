import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Literal, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import yaml
from dacite import Config as DaciteConfig
from dacite import from_dict
from dotenv import load_dotenv

T = TypeVar("T")

logger = logging.getLogger(__name__)

_CONFIG_PATH: str = "config.yml"


def set_config_path(path: str) -> None:
    """Set the global configuration file path.

    Args:
        path: Path to the configuration file
    """
    logger.info(f"Set config path to {path}")
    global _CONFIG_PATH
    _CONFIG_PATH = str(Path(path))


def init_class(
    cls: Type[T],
    config_id: str | None = None,
    config_file_path: str | None = None,
    **override_args,
) -> T:
    """Create a dataclass instance from yaml config file."""
    path = config_file_path if config_file_path is not None else _CONFIG_PATH
    id = config_id if config_id is not None else cls.__name__
    try:
        with open(path, "r") as f:
            config = yaml.safe_load_all(f)
            for doc in config:
                if id in doc:
                    # get config dict
                    config_args = doc[id]
                    # Update with any override arguments
                    config_args.update(override_args)

                    return from_dict(
                        data_class=cls,
                        data=config_args,
                        config=DaciteConfig(
                            cast=[Path],  # auto-cast str -> Path
                            strict=False,  # catches missing/extra fields
                        ),
                    )
    except FileNotFoundError:
        logger.warning(f"Config file not found at {path}")
    except yaml.YAMLError as e:
        logger.warning(f"Error parsing YAML config: {e}")
    except Exception as e:
        logger.warning(f"Failed to instantiate {cls.__name__} from config: {e}")

    # If no config found in YAML or there were errors, initialize with just the override args
    logger.info(f"No config found for {id}, initializing with provided arguments")
    return cls(**override_args)


def time_taken(func):
    """
    A decorator to measure the execution time of a function.

    Args:
        func: The target function.

    Returns:
        A wrapper function that measures and prints the execution time.
    """

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function {func.__name__!r} took {execution_time:.4f} seconds to execute.")
        return result

    return wrapper


def init_result_path(run_dir: str | None = None):
    load_dotenv()

    if "RESULT_PATH" not in os.environ:
        raise EnvironmentError(
            "RESULT_PATH environment variable is not set. "
            "Please add RESULT_PATH to your .env file or environment variables."
        )
    run_id = str(uuid.uuid4())[:8]
    if run_dir is None:
        run_dir = run_id
    else:
        run_dir += run_id

    # Get base result path
    base_path = os.environ["RESULT_PATH"]
    run_path = os.path.join(base_path, "metaprs", run_dir)
    os.makedirs(run_path, exist_ok=True)

    # Create the directory if it doesn't exist
    os.environ["METAPRS_RUN_RESULT_PATH"] = run_path

    logger.info(f"Initialized result path for run: {run_path}")
    return run_path


def get_safe_path(sub_dir, filename):
    if "METAPRS_RUN_RESULT_PATH" not in os.environ:
        init_result_path()
    result_base_path = os.environ["METAPRS_RUN_RESULT_PATH"]

    # Create the subdirectory path
    sub_dir_path = os.path.join(result_base_path, sub_dir)
    os.makedirs(sub_dir_path, exist_ok=True)

    # Return the complete file path
    return os.path.join(sub_dir_path, filename)


def score(
    labels: npt.NDArray | List[npt.NDArray],
    predictions: npt.ArrayLike | List[npt.ArrayLike],
    scorer: Literal["rocauc", "r2", "pearson_corr"],
    classification: bool,
    aggregate: bool | None = False,
) -> float | List[float]:
    """Computes the evaluation metric based on the task type."""
    from scipy.stats import pearsonr
    from sklearn.metrics import roc_auc_score

    if isinstance(labels, np.ndarray):
        labels = [labels]
    if isinstance(predictions, np.ndarray):
        predictions = [predictions]

    if len(labels) != len(predictions):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of predictions ({len(predictions)})"
        )

    # TODO: parallelize if possible
    scores = []
    for y_true, y_pred in zip(labels, predictions):
        match scorer:
            case "pearson_corr":
                try:
                    score = pearsonr(y_pred, y_true).statistic
                except ValueError as e:
                    logger.warning(
                        f"Failed to calculate correlation matrix: {e}. Returning 0.0 as a fallback."
                    )
                    return 0.0
            case "rocauc":
                if classification:
                    try:
                        score = roc_auc_score(y_true, y_pred)
                    except ValueError as e:
                        logger.error(
                            f"An error occured: {e}. Returning ROC AUC of 0.5 as a fallback."
                        )
                        score = 0.5
                else:
                    raise ValueError(f"Metric {scorer} not supported for regression.")
            case "r2":
                if not classification:
                    score = pearsonr(y_true, y_pred).statistic ** 2
                else:
                    raise ValueError(f"Metric {scorer} not supported for classification.")
        scores.append(score)

    if aggregate:
        return sum(scores) / len(scores)
    else:
        if len(scores) == 1:
            return scores[0]
        else:
            return scores


def get_logits(probs: npt.NDArray, eps: float) -> npt.NDArray:
    probs_clipped = np.clip(probs, eps, 1 - eps)  # clip to eps offset to prevent large numbers
    logits = np.log(probs_clipped / (1 - probs_clipped))

    return logits


def sigmoid(x: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-x))


def safe_softmax(x: npt.NDArray, temp: float = 1.0) -> npt.NDArray:
    if temp <= 0.0:
        raise ValueError(f"Temp must be > 0.0. Got temp: {temp}")
    m = np.max(x)
    z = np.exp((x - m) / temp)

    return z / z.sum()


def plot_hist(data: np.array, title: str = "data"):
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {title}")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"hist_{title}.png")

    # Optional: Print some basic statistics
    print(f"Mean: {np.mean(data):.4f}")
    print(f"Std: {np.std(data):.4f}")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print(f"Shape: {data.shape}")


def plot_importance(annotation_df, batch_id: int):
    # Configure matplotlib for large datasets
    import matplotlib

    matplotlib.rcParams["agg.path.chunksize"] = 10000
    matplotlib.rcParams["path.simplify_threshold"] = 0.1

    plt.figure(figsize=(15, 6))

    # Get unique chromosomes and sort them
    chromosomes = annotation_df["chr_name"].unique()
    chrom_order = sorted(chromosomes, key=lambda x: (int(x) if str(x).isdigit() else 100, str(x)))

    # Color palette for chromosomes
    colors = plt.cm.Set1(np.linspace(0, 1, len(chrom_order)))

    x_offset = 0
    x_ticks = []
    x_labels = []

    for i, chrom in enumerate(chrom_order):
        # Get data for this chromosome
        chrom_data = annotation_df[annotation_df["chr_name"] == chrom].copy()
        chrom_data = chrom_data.sort_values("chr_position")  # Sort by chr_positionition

        # Create x-coordinates (continuous across genome)
        chrom_x = chrom_data["chr_position"].values + x_offset
        chrom_y = chrom_data["importance"].values

        # Plot chromosome
        plt.scatter(
            chrom_x,
            chrom_y,
            c=[colors[i % len(colors)]],
            s=0.3,
            alpha=0.5,
            label=f"Chr {chrom}",
        )

        # Track chromosome centers for x-axis labels
        x_ticks.append(x_offset + chrom_data["chr_position"].max() // 2)
        x_labels.append(str(chrom))

        # Update offset for next chromosome
        x_offset += (
            chrom_data["chr_position"].max() + chrom_data["chr_position"].max() * 0.02
        )  # 2% gap

    # Customize plot
    plt.xticks(x_ticks, x_labels, rotation=0)
    plt.xlabel("Chromosome")
    plt.ylabel("Importance (Mean of posterior)")
    plt.title(f"Variant improtance across genome - Batch {batch_id}")
    plt.grid(True, alpha=0.3)

    # Optional: Add legend only if few chromosomes
    if len(chrom_order) <= 12:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=5)

    # Save plot
    filename = f"importance_manhattan_{batch_id}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Manhattan plot saved as {filename}")
