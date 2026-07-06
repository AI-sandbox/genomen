import logging
from typing import Literal

import numpy as np
from sklearn.model_selection import train_test_split

from .. import DataSet, split
from .ld_utils import compute_ld_blocks

logger = logging.getLogger(__name__)


def sample_interaction_pairs(
    n_variants,
    add_pos_idxs,
    rng,
    n_epi_causal,
    overlap_add_epi,
    epi_both_add,
    block_idx_arr=None,
    require_diff_block=False,
    require_non_add_block=False,
):
    interaction_pairs = set()

    all_pos = np.arange(n_variants)
    add_set = set(add_pos_idxs.tolist())
    non_add_pos = np.array([i for i in range(n_variants) if i not in add_set])

    # blocks occupied by any additive causal variant (used by require_non_add_block)
    add_block_set: set[int] = set()
    if require_non_add_block and block_idx_arr is not None:
        add_block_set = {int(block_idx_arr[k]) for k in add_pos_idxs if int(block_idx_arr[k]) != -1}

    def reject_pair(i: int, j: int, is_overlap: bool) -> bool:
        if require_diff_block and block_idx_arr is not None:
            bi, bj = int(block_idx_arr[i]), int(block_idx_arr[j])
            if bi != -1 and bi == bj:
                return True
        # non-overlapping pairs must not land in any additive LD block
        if require_non_add_block and not is_overlap and block_idx_arr is not None:
            bi, bj = int(block_idx_arr[i]), int(block_idx_arr[j])
            if bi in add_block_set or bj in add_block_set:
                return True
        return False

    n_epi_overlap = int(n_epi_causal * overlap_add_epi)
    n_epi_no_overlap = n_epi_causal - n_epi_overlap
    max_attempts = max(n_epi_no_overlap, 1) * 50
    attempts = 0
    while len(interaction_pairs) < n_epi_no_overlap and attempts < max_attempts:
        attempts += 1

        if len(non_add_pos) < 2:
            break

        i, j = rng.choice(non_add_pos, size=2, replace=False)
        if reject_pair(int(i), int(j), is_overlap=False):
            continue
        interaction_pairs.add(tuple(sorted((int(i), int(j)))))

    # sample overlap epistatic pairs (at least one SNP is additive causal)
    if n_epi_overlap > 0:
        if len(add_pos_idxs) == 0:
            logger.warning(
                "overlap_add_epi > 0 but no additive causal SNPs are available; skipping overlap pairs."
            )
        else:
            n_overlap_sampled = 0
            max_attempts = n_epi_overlap * 50
            attempts = 0
            while n_overlap_sampled < n_epi_overlap and attempts < max_attempts:
                attempts += 1
                if epi_both_add:  # both SNPs must be additive causal
                    if len(add_pos_idxs) < 2:
                        break
                    i, j = rng.choice(add_pos_idxs, size=2, replace=False)
                else:
                    # at least one SNP must be additive causal
                    i = int(rng.choice(add_pos_idxs))
                    possible_j = all_pos[all_pos != i]
                    if len(possible_j) == 0:
                        break
                    j = rng.choice(possible_j)
                if reject_pair(int(i), int(j), is_overlap=True):
                    continue
                pair = tuple(sorted((int(i), int(j))))
                if pair in interaction_pairs:
                    continue
                interaction_pairs.add(pair)
                n_overlap_sampled += 1

            if n_overlap_sampled < n_epi_overlap:
                logger.warning(
                    f"Could only sample {n_overlap_sampled} overlap interactions "
                    f"instead of requested {n_epi_overlap}"
                )

    interaction_pairs = list(interaction_pairs)
    if len(interaction_pairs) < n_epi_causal:
        logger.warning(
            f"Could only sample {len(interaction_pairs)} valid interactions "
            f"instead of requested {n_epi_causal}"
        )

    return interaction_pairs


def simulate(
    dataset: DataSet,
    task: Literal["cls", "reg"],
    prevalence: float = 0.1,
    n_samples=1_000,
    h_cov: float = 0.1,
    h_add: float = 0.3,
    h_epi: float = 0.05,
    test_set_size: float = 0.1,
    frac_add_causal: float = 0.1,
    n_epi_causal: int = 300,
    overlap_add_epi: float = 0.0,
    epi_both_add: bool | str = True,
    interaction_order: int = 2,
    max_interactions_per_snp: int = 2,
    require_diff_block: bool | str = False,
    require_non_add_block: bool | str = False,
    seed: int = 0,
):
    if isinstance(epi_both_add, str):
        epi_both_add = epi_both_add.lower() not in ("false", "0", "no")
    if isinstance(require_diff_block, str):
        require_diff_block = require_diff_block.lower() not in ("false", "0", "no")
    if isinstance(require_non_add_block, str):
        require_non_add_block = require_non_add_block.lower() not in ("false", "0", "no")

    if interaction_order != 2:
        raise NotImplementedError("Currently only interaction_order=2 is supported.")
    if max_interactions_per_snp != 2:
        raise NotImplementedError("Currently only max_interactions_per_snp=2 is supported.")

    rng = np.random.default_rng(seed)

    logger.info(
        f"Simulating dataset for {dataset.cfg.sample_sampling.max_samples} samples (ancestry={dataset.cfg.populations}) with MAF={dataset.cfg.maf_threshold}."
    )

    logger.info(f"Subset {n_samples} samples...")
    orig_sampling_strat = dataset.cfg.sample_sampling.strat
    orig_max_samples = dataset.cfg.sample_sampling.max_samples
    dataset.cfg.sample_sampling.strat = "random"
    dataset.cfg.sample_sampling.max_samples = n_samples

    sample_idxs = dataset.sample_sample_idxs(seed, skip_class_check=dataset.cfg.simulate)
    dataset.cfg.sample_sampling.strat = orig_sampling_strat
    dataset.cfg.sample_sampling.max_samples = orig_max_samples
    p = test_set_size
    train_val_idxs, test_idxs = train_test_split(
        sample_idxs, test_size=p, random_state=seed, shuffle=True
    )
    train_idxs, val_idxs = train_test_split(
        train_val_idxs, test_size=p / (1 - p), random_state=seed, shuffle=True
    )
    train_set, val_set, test_set = split(dataset, split_idxs=[train_idxs, val_idxs, test_idxs])

    train_set._compute_maf()  # dropping all variants with MAF < MAF threshold
    for ds in [val_set, test_set]:
        if ds is not None:
            ds.genotype.annotation_df = train_set.genotype.annotation_df.copy()

    variant_idxs = train_set.genotype.variant_idxs
    n_variants = len(variant_idxs)

    # Compute LD blocks once if needed by any flag; re-propagate annotation so val/test get block_idx too
    _need_ld = require_diff_block or require_non_add_block
    block_idx_arr = None
    if _need_ld:
        logger.info("Computing LD blocks...")
        compute_ld_blocks(train_set)
        _ann = train_set.genotype.annotation_df
        block_idx_arr = np.array(
            [int(_ann.loc[g, "block_idx"]) for g in variant_idxs], dtype=np.int64
        )
        for ds in [val_set, test_set]:
            if ds is not None:
                ds.genotype.annotation_df = _ann.copy()

    logger.info(
        f"Simulating phenotype with covariate 'heritability' of {h_cov} (age, sex, PC1-10), additive heritability of {h_add} ({frac_add_causal * 100}% of variants causal) and epistatic heritability of {h_epi} ({n_epi_causal} interactions)."
    )
    # covar effects
    gamma = rng.normal(size=len(train_set.cfg.covar_config.covar_keys))

    # additive effects
    n_add = int(frac_add_causal * n_variants)
    beta = np.zeros(n_variants)
    add_pos_idxs = rng.choice(np.arange(n_variants), size=n_add, replace=False)
    beta[add_pos_idxs] = rng.normal(loc=0.0, scale=1.0 / np.sqrt(max(n_add, 1)), size=n_add)

    # environmental noise — sampled before epistatic block so that the same seed produces the same eps across different h_epi values (only the scale changes)
    h_total = h_cov + h_add + h_epi
    if h_total >= 1.0:
        raise ValueError("Need h_add + h_epi < 1 so residual variance stays positive.")
    noise_sd = np.sqrt(1.0 - h_total)
    eps_raw = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    eps_by_sample = dict(zip(sample_idxs, eps_raw * noise_sd))

    # epistatic effects
    if h_epi > 0:
        interaction_pairs = sample_interaction_pairs(
            n_variants,
            add_pos_idxs,
            rng,
            n_epi_causal,
            overlap_add_epi,
            epi_both_add,
            block_idx_arr=block_idx_arr,
            require_diff_block=require_diff_block,
            require_non_add_block=require_non_add_block,
        )
        alpha = rng.normal(
            0.0, 1.0 / np.sqrt(max(len(interaction_pairs), 1)), size=len(interaction_pairs)
        )
    else:
        interaction_pairs = []
        alpha = np.array([], dtype=float)

    eps = eps_raw * noise_sd

    # generate phenotypes
    # Load centered G and covariates per split (DataBatch centers by train MAF automatically)
    splits_ordered = [("train", train_set), ("val", val_set), ("test", test_set)]
    split_G = {}
    split_C = {}
    for name, ds in splits_ordered:
        split_G[name] = ds[ds.phenotype.sample_idxs].X
        if h_cov > 0:
            split_C[name] = ds.get_covars().X

    # MAF centering consistency check.
    # Log the same fingerprint as [MAF-CHECK MODEL pre-setup] in model.py to verify they match.
    for _split_name, _ds in splits_ordered:
        _maf = _ds.genotype.annotation_df["MAF"].values.astype(np.float64)
        _vidxs = _ds.genotype.annotation_df.index.values
        logger.info(
            "[MAF-CHECK SIM %s] n_variants=%d, MAF mean=%.5f std=%.5f min=%.5f max=%.5f | "
            "first5 global_idx=%s MAF=%s",
            _split_name,
            len(_maf),
            _maf.mean(),
            _maf.std(),
            _maf.min(),
            _maf.max(),
            _vidxs[:5].tolist(),
            _maf[:5].round(5).tolist(),
        )
    # Verify centering of split_G["train"]: column means should be ~0, stds ~1
    _col_means = split_G["train"].mean(axis=0)
    _col_stds = split_G["train"].std(axis=0)
    logger.info(
        "[MAF-CHECK SIM] split_G['train'] post-centering: "
        "col_mean max_abs=%.2e (expect ~0), col_std mean=%.5f min=%.5f (expect ~1)",
        float(np.abs(_col_means).max()),
        float(_col_stds.mean()),
        float(_col_stds.min()),
    )

    # Determine scaling from train split only, then apply uniformly to all splits
    g_add_tr = split_G["train"] @ beta
    add_scale = (
        0.0 if h_add == 0 else (np.sqrt(h_add) / np.std(g_add_tr) if np.var(g_add_tr) > 0 else 1.0)
    )
    if h_cov > 0:
        g_cov_tr = split_C["train"] @ gamma
        cov_scale = np.sqrt(h_cov) / np.std(g_cov_tr) if np.var(g_cov_tr) > 0 else 1.0
    else:
        g_cov_tr = np.zeros(len(train_set))
        cov_scale = 0.0

    if h_epi > 0 and len(interaction_pairs) > 0:
        g_epi_tr = np.zeros(len(train_set))
        for k, (i, j) in enumerate(interaction_pairs):
            g_epi_tr += alpha[k] * (split_G["train"][:, i] * split_G["train"][:, j])
        epi_scale = np.sqrt(h_epi) / np.std(g_epi_tr) if np.var(g_epi_tr) > 0 else 1.0
    else:
        epi_scale = 1.0

    beta_scaled = beta * add_scale
    alpha_scaled = alpha * epi_scale

    # log scaled variance components (should match target heritabilities)
    g_eps_tr = eps[0 : len(train_set)]
    g_add_scaled_tr = g_add_tr * add_scale
    g_cov_scaled_tr = g_cov_tr * cov_scale
    log_vals = {
        "add": g_add_scaled_tr,
        "cov": g_cov_scaled_tr,
        "eps": g_eps_tr,
    }
    if "g_epi_tr" in locals():
        g_epi_scaled_tr = g_epi_tr * epi_scale
        log_vals["epi"] = g_epi_scaled_tr
    for name, x in log_vals.items():
        print(f"var({name})", np.var(x))

    if "g_epi_tr" in locals():
        print("corr add-epi", np.corrcoef(g_add_scaled_tr, g_epi_scaled_tr)[0, 1])
        print("corr cov-epi", np.corrcoef(g_cov_scaled_tr, g_epi_scaled_tr)[0, 1])
    print("corr add-cov", np.corrcoef(g_add_scaled_tr, g_cov_scaled_tr)[0, 1])
    total = sum(log_vals.values())
    print("total var", np.var(total))

    # Compute y_liability per split; eps is partitioned by split size (train, val, test order)
    y_liabilities = {}
    pos = 0
    for name, ds in splits_ordered:
        n = len(ds)
        G = split_G[name]
        g_add = (G @ beta) * add_scale
        g_cov = (split_C[name] @ gamma) * cov_scale if h_cov > 0 else np.zeros(n)
        if h_epi > 0 and len(interaction_pairs) > 0:
            g_epi = np.zeros(n)
            for k, (i, j) in enumerate(interaction_pairs):
                g_epi += alpha[k] * (G[:, i] * G[:, j])
            g_epi *= epi_scale
        else:
            g_epi = np.zeros(n)
        sample_ids = ds.phenotype.sample_idxs
        eps_split = np.array([eps_by_sample[s] for s in sample_ids])
        y_liabilities[name] = g_add + g_cov + g_epi + eps_split
        pos += n

    # Global classification threshold across all sim samples
    if task == "cls":
        y_all = np.concatenate([y_liabilities[name] for name, _ in splits_ordered])
        threshold = np.quantile(y_all, 1 - prevalence)

    sim_phenotype_id = (
        f"{dataset.cfg.phenotype_id}"
        f"_task{task}_prev{prevalence}_n{n_samples}"
        f"_hcov{h_cov}_hadd{h_add}_hepi{h_epi}"
        f"_fadd{frac_add_causal}_nepi{n_epi_causal}"
        f"_ove{overlap_add_epi}_epiboth{int(epi_both_add)}_seed{seed}"
    )
    for name, ds in splits_ordered:
        y = (y_liabilities[name] > threshold).astype(int) if task == "cls" else y_liabilities[name]
        ds.phenotype.annotation_df["y"] = y
        ds.cfg.phenotype_id = sim_phenotype_id
        ds.cfg.classification = task == "cls"

    ann = train_set.genotype.annotation_df
    interaction_pairs_arr = (
        np.array(interaction_pairs, dtype=np.int64).reshape(-1, 2)
        if interaction_pairs
        else np.empty((0, 2), dtype=np.int64)
    )
    if len(interaction_pairs_arr) > 0:
        i_idxs = interaction_pairs_arr[:, 0]
        j_idxs = interaction_pairs_arr[:, 1]
        interaction_chrs = np.stack(
            [ann["chr_name"].values[i_idxs], ann["chr_name"].values[j_idxs]], axis=1
        )
        interaction_pos = np.stack(
            [ann["chr_position"].values[i_idxs], ann["chr_position"].values[j_idxs]], axis=1
        )
        interaction_snps = np.stack([ann["snp"].values[i_idxs], ann["snp"].values[j_idxs]], axis=1)
    else:
        interaction_chrs = np.empty((0, 2), dtype=object)
        interaction_pos = np.empty((0, 2), dtype=np.int64)
        interaction_snps = np.empty((0, 2), dtype=object)
    sim_dict = {
        "prevalence": prevalence,
        "sample_idxs": sample_idxs,
        "beta": beta_scaled,
        "alpha": alpha_scaled,
        "add_pos_idxs": add_pos_idxs,
        "add_global_idxs": ann.index.values[add_pos_idxs],
        "add_snps": ann["snp"].values[add_pos_idxs],
        "add_chrs": ann["chr_name"].values[add_pos_idxs],
        "add_pos": ann["chr_position"].values[add_pos_idxs],
        "interaction_pairs_pos": interaction_pairs_arr,
        "interaction_chrs": interaction_chrs,
        "interaction_pos": interaction_pos,
        "interaction_snps": interaction_snps,
        "h_cov": h_cov,
        "h_add": h_add,
        "h_epi": h_epi,
    }

    logger.info("Simulation complete.")
    return train_set, val_set, test_set, sim_dict
