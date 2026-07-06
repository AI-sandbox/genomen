import logging
import multiprocessing
import warnings

from .run_manager import RunManager

warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
    category=FutureWarning,
)

logger = logging.getLogger(__name__)
if multiprocessing.current_process().name == "MainProcess":
    print(
        f"{'#' * 10} Welcome to Genomic Ensembling (GenomEn) - Polygenic risk and association beyond linearity {'#' * 10}"
    )

global_run_manager = RunManager()
