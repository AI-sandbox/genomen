"""Generate a small, fully-synthetic PLINK1 + master-file dataset for demoing GenomEn.

Produces (under --out_dir):
    demo.bed / demo.bim / demo.fam   PLINK1 binary genotype files
    demo_master.phe                  tab-delimited phenotype/covariate/split file

The data has no biological meaning -- genotypes are drawn independently per
variant from a random minor allele frequency, and the phenotype is a binary
trait with injected genetic (h_geno) and covariate (h_cov) signal, scaled to
the requested share of variance explained, so a demo model can show
non-trivial (but not perfect) predictive performance.

Usage:
    python docs/scripts/generate_demo_data.py --out_dir=data/demo
"""

import logging

import numpy as np
import pandas as pd
from bed_reader import to_bed
from fire import Fire

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main(
    out_dir: str = "data/demo",
    n_samples: int = 1_000,
    n_variants: int = 5_000,
    n_causal: int = 30,
    prevalence: float = 0.15,
    missing_rate: float = 0.02,
    h_geno: float = 0.3,
    h_cov: float = 0.3,
    seed: int = 0,
):
    from pathlib import Path

    if h_geno + h_cov >= 1.0:
        raise ValueError("Need h_geno + h_cov < 1 so residual variance stays positive.")

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- samples: plain numeric IIDs, since genomen casts master IID via int() ---
    iids = np.arange(1_000_001, 1_000_001 + n_samples)
    fids = iids.copy()
    sex = rng.integers(0, 2, size=n_samples)  # 0/1 covariate convention used by genomen

    # --- variants: spread across autosomes 1-22 ---
    logger.info(f"Simulating {n_variants} variants across chromosomes 1-22...")
    chrom_pool = [str(c) for c in range(1, 23)]
    chroms = np.sort(rng.choice(np.arange(1, 23), size=n_variants)).astype(str)
    positions = np.empty(n_variants, dtype=np.int64)
    for c in chrom_pool:
        mask = chroms == c
        n_c = int(mask.sum())
        if n_c:
            positions[mask] = np.sort(rng.integers(10_000, 100_000_000, size=n_c))
    snp_ids = [f"rs{100_000 + i}" for i in range(n_variants)]
    alleles = np.array(["A", "C", "G", "T"])
    allele_1 = rng.choice(alleles, size=n_variants)
    allele_2 = np.array([rng.choice(alleles[alleles != a]) for a in allele_1])

    # --- genotypes: independent binomial draws from a random MAF per variant ---
    logger.info(f"Simulating genotypes for {n_samples} samples...")
    maf = rng.uniform(0.01, 0.5, size=n_variants)
    geno = rng.binomial(2, maf, size=(n_samples, n_variants)).astype(np.float64)
    missing_mask = rng.random(size=geno.shape) < missing_rate
    geno[missing_mask] = np.nan

    # --- covariates ---
    age = rng.normal(55, 10, size=n_samples).round(1).clip(30, 80)
    pcs = rng.normal(0, 1, size=(n_samples, 10))

    # --- binary phenotype: genetic + covariate signal scaled to h_geno / h_cov
    # share of variance explained, remainder is noise (mirrors the convention used
    # in genomen.data.simulations.simulate_data.simulate) ---
    causal_idxs = rng.choice(n_variants, size=n_causal, replace=False)
    beta = rng.normal(0, 1 / np.sqrt(n_causal), size=n_causal)
    geno_causal = np.nan_to_num(geno[:, causal_idxs], nan=2 * maf[causal_idxs])
    g_add_raw = geno_causal @ beta
    add_scale = np.sqrt(h_geno) / np.std(g_add_raw) if np.var(g_add_raw) > 0 else 0.0
    g_add = g_add_raw * add_scale

    covar_matrix = np.column_stack([age, sex, pcs])
    gamma = rng.normal(size=covar_matrix.shape[1])
    g_cov_raw = covar_matrix @ gamma
    cov_scale = np.sqrt(h_cov) / np.std(g_cov_raw) if np.var(g_cov_raw) > 0 else 0.0
    g_cov = g_cov_raw * cov_scale

    noise_sd = np.sqrt(1.0 - h_geno - h_cov)
    eps = rng.normal(0, noise_sd, size=n_samples)

    liability = g_add + g_cov + eps
    threshold = np.quantile(liability, 1 - prevalence)
    pheno = (liability > threshold).astype(int)

    split = rng.choice(["train", "val", "test"], size=n_samples, p=[0.7, 0.15, 0.15])

    # --- write PLINK1 binary files ---
    bed_path = out_dir / "demo.bed"
    logger.info(f"Writing {bed_path}, {bed_path.with_suffix('.bim')}...")
    to_bed(
        bed_path,
        geno,
        properties={
            "fid": fids.astype(str).tolist(),
            "iid": iids.astype(str).tolist(),
            "father": ["0"] * n_samples,
            "mother": ["0"] * n_samples,
            "sex": (sex + 1).tolist(),  # PLINK convention: 1=male, 2=female
            "pheno": [-9] * n_samples,
            "chromosome": chroms.tolist(),
            "sid": snp_ids,
            "cm_position": [0.0] * n_variants,
            "bp_position": positions.tolist(),
            "allele_1": allele_1.tolist(),
            "allele_2": allele_2.tolist(),
        },
    )

    # bed_reader writes .fam space-delimited; genomen's loader expects tab-delimited
    fam_path = out_dir / "demo.fam"
    logger.info(f"Rewriting {fam_path} as tab-delimited...")
    pd.DataFrame(
        {
            "fid": fids,
            "iid": iids,
            "father": 0,
            "mother": 0,
            "sex": sex + 1,
            "trait": -9,
        }
    ).to_csv(fam_path, sep="\t", header=False, index=False)

    # --- master / phenotype file ---
    master_path = out_dir / "demo_master.phe"
    logger.info(f"Writing {master_path}...")
    master_df = pd.DataFrame(
        {
            "IID": iids,
            "population": "demo",
            "split": split,
            "age": age,
            "sex": sex,
            **{f"Global_PC{i + 1}": pcs[:, i] for i in range(10)},
            "PHENO1": pheno,
        }
    )
    master_df.to_csv(master_path, sep="\t", index=False)

    logger.info(
        f"Done. {n_samples} samples, {n_variants} variants, "
        f"{pheno.sum()} cases ({100 * pheno.mean():.1f}%). "
        f"var(geno)={np.var(g_add):.3f} (target h_geno={h_geno}), "
        f"var(cov)={np.var(g_cov):.3f} (target h_cov={h_cov})."
    )


if __name__ == "__main__":
    Fire(main)
