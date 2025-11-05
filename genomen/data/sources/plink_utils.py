import concurrent.futures
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import gzip

import numpy as np
import pandas as pd
import pgenlib as pg

logger = logging.getLogger(__name__)


def get_plink_paths() -> Dict[str, str]:
    """Get paths to PLINK files from environment variables."""
    try:
        fam_path = os.environ["FAM_PATH"]
        bed_path = os.environ["BED_PATH"]
        master_path = os.environ["MASTER_PATH"]
        bim_path = os.environ["BIM_PATH"]

        if bed_path.startswith('b"') or bed_path.startswith("b'"):
            logger.error(
                "Error: Bed path should be provided as plain strings, not with 'b' in .env file."
            )
            raise ValueError(
                "Error: Bed path should be provided as plain strings, not with 'b' in .env file."
            )

        bed_path = bed_path.encode("utf-8") if bed_path else None

    except KeyError as e:
        logger.error(f"Error loading paths from .env file: {e}. Check if all paths are set.")
        raise

    return dict(
        fam_path=fam_path,
        bed_path=bed_path,
        master_path=master_path,
        bim_path=bim_path,
    )


def load_fam_data(path: str) -> pd.DataFrame:
    """
    Load .fam file data into a DataFrame and count the number of samples.
    The .fam file describes the individuals in the study sample and their
    associated family and/or phenotype data.

    Parameters:
        path (str): The file path to the .fam file.

    Returns:
        Tuple[pd.DataFrame, int]: A tuple containing the DataFrame of the
        .fam file and the number of samples present in the file.
    """
    fam_column_names = ["fid", "iid", "father", "mother", "gender", "trait"]
    fam_df = pd.read_csv(path, sep="\t", names=fam_column_names)
    fam_df["iid"] = fam_df["iid"].apply(lambda x: str(x))

    return fam_df


def load_master_data(
    path: str, columns: List[str] | None = None, sex: Literal["m", "w"] | None = None
) -> pd.DataFrame:
    """
    Load master file data into a DataFrame. The master file contains phenotype
    many continuous and binary phenotypes information for every individual.
    It also includes the Family ID (FID), Individual ID (IID), populations, and age.

    Parameters:
        path (str): The file path to the master file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the master file.
    """
    if (sex is not None) and (sex not in columns):
        columns.append("sex")

    master_df = pd.read_csv(path, sep="\t", usecols=columns)
    master_df["IID"] = master_df["IID"].astype(int).astype(str)

    if sex is not None:
        logger.info(f"Filtering samples by sex: {sex}")
        sex_code = 1 if sex == "m" else 0
        before_count = len(master_df)
        master_df = master_df[master_df["sex"] == sex_code]
        after_count = len(master_df)
        logger.info(f"Dropped {before_count - after_count} samples after filtering by sex.")

    return master_df


def load_bim_data(
    path: str,
    include_x_chromosome: bool = False,
) -> pd.DataFrame:
    """
    Load .bim file data into a DataFrame. The bim file describes the genetic markers
    used in the study. It contains information such as the chromosome number,
    genetic position, and allele names for each marker.

    Parameters:
        path (str): The file path to the .bim file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the .bim file.
    """
    bim_column_names = ["chrom", "snp", "cm", "pos", "a0", "a1"]
    bim_df = pd.read_csv(path, sep="\t", names=bim_column_names, low_memory=False)

    valid_chromosomes = [str(i) for i in range(1, 23)]
    if include_x_chromosome:
        valid_chromosomes += ["X"]

    bim_df = bim_df[bim_df["chrom"].isin(valid_chromosomes)]

    return bim_df


def process_master_df(
    fam_df: pd.DataFrame,
    master_df: pd.DataFrame,
    classification: bool,
    phenotype_id: str,
    populations: List[str],
) -> pd.DataFrame:
    """ """
    master_df = master_df[master_df["population"].isin(populations)]
    master_df = (
        master_df.drop_duplicates(subset=["IID"])
        .replace([-9, -9.0, "-9", "-9.0"], np.nan)
        .dropna(subset=["IID", phenotype_id])
    )

    unique_labels = np.unique(master_df[phenotype_id])
    if (
        len(unique_labels) == 2
        and classification
        and not np.array_equal(unique_labels, np.array([0, 1]))
    ):
        logger.warning("Phenotype is binary but not 0/1. Converting to 0/1.")
        master_df[phenotype_id] = (
            master_df[phenotype_id] > np.median(master_df[phenotype_id])
        ).astype(np.uint32)

    # Order and filter master_df with fam_df
    master_df = pd.merge(fam_df, master_df, left_on="iid", right_on="IID", how="left").dropna(
        subset=["IID"]
    )
    master_df["fam_idx"] = master_df.index
    master_df = master_df.reset_index(drop=True)

    return master_df


def load_pgen_reader(bed_path: bytes, n_samples: int, idxs: np.ndarray) -> pg.PgenReader:
    """
    Load a PgenReader object containing the genomic data for specific
    sample IDs.

    Parameters:
        path (btyes): The file path to the .bed file.
        n_samples (int): The total number of samples in the .bed file.
        idxs (np.ndarray): An array of indices specifying the samples to load.

    Returns:
        Tuple[pg.PgenReader, int]:
            - The PgenReader object.
            - The number of variants (features) found in the .pgen file.
    """
    pgen_reader = pg.PgenReader(bed_path, raw_sample_ct=n_samples, sample_subset=idxs)

    return pgen_reader


def calculate_maf(bed_path: bytes, bim_path: str, fam_path: str) -> pd.DataFrame:
    """Calculate minor allele frequencies using PLINK.

    Args:
        bed_path: Path to .bed file
        bim_path: Path to .bim file
        fam_path: Path to .fam file

    Returns:
        DataFrame containing SNP IDs and their MAF values

    Raises:
        ValueError: If the input files don't share the same prefix and directory
        ValueError: If plink_executable is not configured
    """
    plink_path = get_plink_path()

    bed_path_str = bed_path.decode()
    paths = {
        "bed": os.path.split(bed_path_str),
        "bim": os.path.split(bim_path),
        "fam": os.path.split(fam_path),
    }

    # Check if all files are in the same directory
    if len(set(p[0] for p in paths.values())) > 1:
        raise ValueError("All PLINK files must be in the same directory")

    # Check if all files share the same prefix
    prefixes = {name: p[1].rsplit(".", 1)[0] for name, p in paths.items()}
    if len(set(prefixes.values())) > 1:
        raise ValueError("All PLINK files must share the same prefix")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_prefix = os.path.join(tmp_dir, "freq")
        input_prefix = bed_path_str.rsplit(".", 1)[0]

        cmd = [
            plink_path,
            "--bfile",
            input_prefix,
            "--freq",
            "--out",
            output_prefix,
            "--silent",
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise ValueError(f"MAF with PLINK2 failed: {e}")

        expected_file = output_prefix + ".frq"

        freq_df = pd.read_csv(expected_file, sep="\\s+", low_memory=False)
        freq_df = freq_df.rename(columns={"ID": "SNP"})

    return freq_df


def get_plink_path():
    plink_path = Path(__file__).resolve().parents[3] / "assets" / "plink"
    # make plink executable if not already
    if not (os.path.isfile(plink_path) and os.access(plink_path, os.X_OK)):
        logger.info("Making plink file executable!")
        cmd = ["chmod", "+x", plink_path]
        subprocess.run(cmd, check=True)

    return plink_path


def get_repr_per_block(
    chr_num: int,
    plink_path: str | Path,
    bfile_prefix: str,
    maf_threshold: int,
    prune_kb: int,
    prune_step: int,
    prune_r2: float,
    out_dir: Path,
):
    # build a per-chr output prefix
    chr_prefix = out_dir / f"prn_chr{chr_num}"
    cmd = [
        plink_path,
        "--bfile",
        bfile_prefix,
        "--chr",
        str(chr_num),
        "--maf",
        str(maf_threshold),
        "--indep-pairwise",
        str(prune_kb),
        str(prune_step),
        str(prune_r2),
        "--out",
        str(chr_prefix),
    ]
    try:
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"PLINK command failed for chromosome {chr_num}")
        logger.error(f"Command: {' '.join(str(x) for x in cmd)}")
        logger.error(f"Return code: {e.returncode}")
        if e.stderr:
            logger.error(f"PLINK stderr output: {e.stderr}")
        raise

    return chr_prefix.with_suffix(".prune.in")


def compute_ld(
    bed_path: bytes,
    bim_path: str,
    fam_path: str,
    blocks_max_kb: int,
    maf_threshold: float,
    prune_kb: int,
    prune_step: int,
    prune_r2: float,
    tau: float,
    ld_window_kb: int,
    ld_window: int,
    include_x: bool,
    ram_mb: int,
) -> pd.DataFrame:
    plink_path = get_plink_path()

    bed_path_str = bed_path.decode()
    paths = {
        "bed": os.path.split(bed_path_str),
        "bim": os.path.split(bim_path),
        "fam": os.path.split(fam_path),
    }

    # Check if all files are in the same directory
    if len(set(p[0] for p in paths.values())) > 1:
        raise ValueError("All PLINK files must be in the same directory")

    # Check if all files share the same prefix
    prefixes = {name: p[1].rsplit(".", 1)[0] for name, p in paths.items()}
    if len(set(prefixes.values())) > 1:
        raise ValueError("All PLINK files must share the same prefix")

    cpu_count = os.cpu_count() or 1
    workers = max(1, cpu_count - 1)  # leave 1 core for os
    workers = min(workers, 22)  # never more than 22 workers

    with tempfile.TemporaryDirectory() as tmp_dir:
        bfile_prefix = bed_path_str.rsplit(".", 1)[0]
        tmp_dir = Path(tmp_dir)

        # compute representatives per region
        repr_files: list[Path] = []
        # autosomes
        out_auto = tmp_dir / "prn_auto"
        out_auto.mkdir(exist_ok=True)

        autosomes = list(range(1, 23))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
            futures = [
                exe.submit(
                    get_repr_per_block,
                    chr,
                    plink_path,
                    bfile_prefix,
                    maf_threshold,
                    prune_kb,
                    prune_step,
                    prune_r2,
                    out_auto,
                )
                for chr in autosomes
            ]
            for future in concurrent.futures.as_completed(futures):
                rf = future.result()
                if rf.exists():
                    repr_files.append(rf)
        if include_x:  # sex chromosome
            out_x = tmp_dir / "prn_X"
            out_x.mkdir(exist_ok=True)
            x_cmd = [
                plink_path,
                "--bfile",
                bfile_prefix,
                "--chr",
                "X",
                "--maf",
                str(maf_threshold),
                "--indep-pairwise",
                str(prune_kb),
                str(prune_step),
                str(prune_r2),
                "--ld-xchr",
                "3",
                "--out",
                str(out_x),
                "--threads",
                str(workers),
            ]
            subprocess.run(
                x_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            if (out_x.with_suffix(".prune.in")).exists():
                repr_files.append(out_x.with_suffix(".prune.in"))

        reprs: List[str] = []
        reprs_path = tmp_dir / "reps.in"
        for repr in repr_files:
            with open(repr, "r") as fh:
                reprs.extend(s.strip() for s in fh if s.strip())
        reprs = sorted(set(reprs))
        with open(reprs_path, "w") as fh:
            fh.write("\n".join(reprs) + "\n")

        # LD map from reps to all SNPs
        ld_out = tmp_dir / "ldmap"
        ld_cmd = [
            plink_path,
            "--bfile",
            bfile_prefix,
            "--r2",
            "gz",
            "--ld-window-kb",
            str(ld_window_kb),
            "--ld-window",
            str(ld_window),
            "--ld-window-r2",
            str(tau),
            "--ld-snp-list",
            str(reprs_path),
            "--out",
            str(ld_out),
            "--threads",
            str(workers),
            "--memory",
            str(ram_mb),
        ]
        subprocess.run(
            ld_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )

        best: dict[str, Tuple[str, float]] = {}
        ld_gz = ld_out.with_suffix(".ld.gz")
        if ld_gz.exists():
            with gzip.open(ld_gz, "rt") as fh:
                header = fh.readline().strip().split()
                iA, iB, iR2 = header.index("SNP_A"), header.index("SNP_B"), header.index("R2")
                for ln in fh:
                    parts = ln.strip().split()
                    rep, snp = parts[iA], parts[iB]
                    r2 = float(parts[iR2])
                    if r2 < tau:
                        continue
                    cur = best.get(snp)
                    if (cur is None) or (r2 > cur[1]):
                        best[snp] = (rep, r2)

        snp_list = list(best.keys())
        block_ids = [f"REPR:{best[s][0]}" for s in snp_list]
        df = pd.DataFrame({"snp": snp_list, "block_id": block_ids})
        uniq = {bid: i + 1 for i, bid in enumerate(sorted(set(df["block_id"])))}
        df["block_idx"] = df["block_id"].map(uniq)

        return df
