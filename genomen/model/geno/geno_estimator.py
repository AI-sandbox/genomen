import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import wandb

from ... import utils
from ...data import DataBatch, DataSet, split
from ..configs import GenoConfig, ModelConfig, TrainConfig
from .geno_aggregator import GenoAggregator
from .weak_geno_estimator import WeakGenoEstimator


class GenoEstimator:
    @classmethod
    def load(cls, path: str) -> "GenoEstimator":
        return joblib.load(path)

    def __init__(self, cfg: GenoConfig):
        self.cfg = cfg
        self._logger = logging.getLogger(self.__class__.__name__)
        self.model: List[WeakGenoEstimator] | WeakGenoEstimator | None = None
        self.unresid_model: LogisticRegression | LinearRegression | None = None
        self.annotation_df: pd.DataFrame = None
        self.aggregator: GenoAggregator | None = None
        self.interactions: pd.DataFrame | None = None

    def _init_model(self, weak_estimators: List[WeakGenoEstimator]) -> None:
        # aggregate annotation_dfs
        concatenated_annotation_df = pd.concat(
            [estimator.annotation_df.copy() for estimator in weak_estimators],
            ignore_index=False,
        )
        self.annotation_df = (
            concatenated_annotation_df.groupby(level=0)  # group of "variant_idx"
            .first()
            .sort_index()
        )

        # aggregate shap values if present
        has_shaps = all(
            "shap_values" in estimator.annotation_df.columns for estimator in weak_estimators
        )
        if has_shaps:
            has_abs_col = all(
                "shap_values_abs" in est.annotation_df.columns for est in weak_estimators
            )

            def _abs_shap_source_df(estimators_with_global_idx):
                # Per-estimator mean(|shap|), keyed by global estimator_idx.
                if has_abs_col:
                    dfs = [
                        pd.DataFrame(
                            {
                                "variant_idx": est.variant_idxs,
                                "shap_values_abs": est.annotation_df["shap_values_abs"].values,
                                "estimator_idx": global_i,
                            }
                        )
                        for global_i, est in estimators_with_global_idx
                    ]
                else:
                    dfs = [
                        pd.DataFrame(
                            {
                                "variant_idx": est.variant_idxs,
                                "shap_values_abs": np.abs(est.annotation_df["shap_values"].values),
                                "estimator_idx": global_i,
                            }
                        )
                        for global_i, est in estimators_with_global_idx
                    ]
                return pd.concat(dfs, ignore_index=False)

            def _aggregate_abs_shap(estimators_with_global_idx, filter_mask=None):
                # Combine per-estimator mean(|shap|) with the same calibrated aggregator
                # (NaN-aware mean / aggregation weights / filter_mask) used for the signed
                # shap_values below, instead of a raw sum across estimators -- a raw sum
                # would scale with how many estimators happened to include a variant
                # (which differs between e.g. linear and nonlinear weak-estimator counts).
                abs_df = _abs_shap_source_df(estimators_with_global_idx)
                return (
                    abs_df.set_index("variant_idx", drop=True)
                    .groupby(level=0, sort=True)
                    .apply(
                        lambda grp: self.aggregator.aggregate_group(
                            group=grp,
                            columns=["shap_values_abs"],
                            n_estimators=len(weak_estimators),
                            aggregate_coeffs=True,
                            filter_mask=filter_mask,
                        )
                    )
                    .sort_index()
                )

            # aggregate global shap
            shap_dfs = []
            for i, estimator in enumerate(weak_estimators):
                est_shap_df = pd.DataFrame(
                    {
                        "variant_idx": estimator.variant_idxs,
                        "shap_values": estimator.annotation_df["shap_values"].values,
                        "estimator_idx": i,
                    }
                )
                shap_dfs.append(est_shap_df)

            concat_shap_df = pd.concat(shap_dfs, ignore_index=False)
            agg_shap_df = (
                concat_shap_df.set_index("variant_idx", drop=True)
                .groupby(level=0, sort=True)
                .apply(
                    lambda grp: self.aggregator.aggregate_group(
                        group=grp,
                        columns=["shap_values"],
                        n_estimators=len(weak_estimators),
                        aggregate_coeffs=True,  # do not add offset for stacking
                    )
                )
                .sort_index()
            )
            abs_agg_shap_df = _aggregate_abs_shap(list(enumerate(weak_estimators)))

            cols_to_drop = [
                c for c in ["shap_values", "shap_values_abs"] if c in self.annotation_df.columns
            ]
            self.annotation_df = self.annotation_df.drop(columns=cols_to_drop)
            self.annotation_df = self.annotation_df.join(
                agg_shap_df[["shap_values"]], how="left"
            ).join(abs_agg_shap_df[["shap_values_abs"]], how="left")

            # for ensemble models, also save per-model-type shap values
            if self.cfg.is_ensemble:
                linear_estimators = [
                    (i, est)
                    for i, est in enumerate(weak_estimators)
                    if est.cfg.model_type == "linear"
                ]
                nonlinear_estimators = [
                    (i, est)
                    for i, est in enumerate(weak_estimators)
                    if est.cfg.model_type != "linear"
                ]
                for group_name, group_estimators in [
                    ("linear", linear_estimators),
                    ("nonlinear", nonlinear_estimators),
                ]:
                    if not group_estimators:
                        continue
                    group_shap_dfs = []
                    for global_i, estimator in group_estimators:
                        group_shap_dfs.append(
                            pd.DataFrame(
                                {
                                    "variant_idx": estimator.variant_idxs,
                                    "shap_values": estimator.annotation_df["shap_values"].values,
                                    "estimator_idx": global_i,
                                }
                            )
                        )
                    group_concat_shap_df = pd.concat(group_shap_dfs, ignore_index=False)
                    # Build a global-length type mask and AND with the global filter_mask
                    # so that estimators of the wrong type and filtered-out estimators are
                    # both excluded from the per-type SHAP average.
                    type_mask = np.zeros(len(weak_estimators), dtype=bool)
                    for global_i, _ in group_estimators:
                        type_mask[global_i] = True
                    group_filter_mask = (
                        type_mask & self.aggregator.filter_mask
                        if self.aggregator.filter_mask is not None
                        else type_mask
                    )
                    group_agg_shap_df = (
                        group_concat_shap_df.set_index("variant_idx", drop=True)
                        .groupby(level=0, sort=True)
                        .apply(
                            lambda grp: self.aggregator.aggregate_group(
                                group=grp,
                                columns=["shap_values"],
                                n_estimators=len(weak_estimators),
                                aggregate_coeffs=True,
                                filter_mask=group_filter_mask,
                            )
                        )
                        .sort_index()
                    )
                    group_abs_agg_shap_df = _aggregate_abs_shap(
                        group_estimators, filter_mask=group_filter_mask
                    ).rename(columns={"shap_values_abs": f"shap_values_{group_name}_abs"})
                    col_name = f"shap_values_{group_name}"
                    self.annotation_df = self.annotation_df.join(
                        group_agg_shap_df[["shap_values"]].rename(
                            columns={"shap_values": col_name}
                        ),
                        how="left",
                    ).join(group_abs_agg_shap_df, how="left")

        # aggregate models (if possible)
        if (self.cfg.model_config.model_type == "linear") and (not self.cfg.is_ensemble):
            model = [self.fold_linear_estimators(weak_estimators)]
            self.aggregator.reset()  # reset aggregator since folded into model
        else:
            model = weak_estimators
        self.model = model

    def fold_linear_estimators(
        self, linear_estimators: List[WeakGenoEstimator]
    ) -> WeakGenoEstimator:
        """Fold a list of linear weak estimators (e.g. bagged linear_l1 sub-models,
        each fit on its own variant/sample subsample) into a single WeakGenoEstimator
        with one aggregated coef_/intercept_, using this geno model's aggregator
        (mean / loss-weighted / stacking / etc., respecting its filter_strat).

        Each estimator's coefficients are placed in the global variant-index space
        before aggregation, so the folded model predicts using the union of
        variants the input estimators collectively used — no refitting, no
        variant subset selection.
        """
        has_scaler = [estimator.scaler is not None for estimator in linear_estimators]
        if not all(has_scaler) and not all(not x for x in has_scaler):
            raise ValueError(
                "Cannot have some weak_estimators trained with standardization and some without"
            )
        standardization = any(has_scaler)

        # Aggregate estimator information
        estimator_dfs = []
        estimator_intercepts = []
        for i, estimator in enumerate(linear_estimators):
            if getattr(estimator, "_use_sklearn_offset", False):
                # offset appended as last column — exclude it from geno betas
                beta = estimator.model.coef_.reshape(-1)[:-1]
                beta0 = float(np.atleast_1d(estimator.model.intercept_)[0])
            else:
                beta = estimator.model.coef_
                if hasattr(beta, "values"):
                    # sklearn/pandas access pattern
                    beta = beta.values.reshape(-1)
                else:
                    # numpy array
                    beta = beta.reshape(-1)

                beta0 = (
                    estimator.model.intercept_.item()
                    if np.ndim(estimator.model.intercept_)
                    else float(estimator.model.intercept_)
                )
            if standardization:  # adjust for 'raw space'
                beta = beta * estimator.scaler.scale_
                beta0 = beta0 * estimator.scaler.scale_ + estimator.scaler.mean_

            df_estimator = pd.DataFrame(
                {
                    "variant_idx": estimator.variant_idxs,
                    "coefficient": beta,
                    "estimator_idx": i,
                }
            )
            estimator_intercepts.append(beta0)
            estimator_dfs.append(df_estimator)

        # Aggregate estimators
        combined_df = pd.concat(estimator_dfs, ignore_index=False)
        estimator_df = (
            combined_df.set_index("variant_idx", drop=True)
            .groupby(level=0, sort=True)
            .apply(
                lambda grp: self.aggregator.aggregate_group(
                    group=grp,
                    columns=["coefficient"],
                    n_estimators=len(linear_estimators),
                    aggregate_coeffs=True,  # do not add offset for stacking
                )
            )
            .sort_index()
        )
        estimator_intercept = self.aggregator(estimator_intercepts).item()

        # Initialize model parameters
        if self.cfg.model_config.classification:
            coef = estimator_df["coefficient"].values.reshape(1, -1)
            intercept = np.array([estimator_intercept])
            classes = np.array([0, 1])
        else:
            coef = estimator_df["coefficient"].values
            intercept = estimator_intercept
            classes = None

        model_init_params = {
            "coef_": coef,
            "intercept_": intercept,
            "classes_": classes,
            "n_features_in_": estimator_df.shape[0],
        }

        # Create a single weak meta model with the aggregated parameters
        folded = WeakGenoEstimator(linear_estimators[0].cfg, model_init_params=model_init_params)
        folded.annotation_df = self.annotation_df.loc[estimator_df.index]
        folded.scaler = None
        return folded

    def save_annotation_file(self, annotation_dir: Path | str):
        # save geno annotation file if needed
        annotation_file_path = Path(annotation_dir, "annotation_df.parquet")
        self._logger.info(f"Saving annotation file to {annotation_file_path}...")
        columns_to_drop = (
            ["cm", "MAF", "rsID"] + ["block_id"]
            if "block_id" in self.annotation_df
            else [] + ["sampling_prior"] if "sampling_prior" in self.annotation_df else []
        )
        df_to_save = self.annotation_df.drop(columns=columns_to_drop)

        if self.cfg.model_config.model_type == "linear":
            coef = self.model[0].model.coef_
            if hasattr(coef, "_owner"):
                # cuml access pattern
                coef = coef._owner.get()
            elif hasattr(coef, "values"):
                # sklearn/pandas access pattern
                coef = coef.values.reshape(-1)
            else:
                # numpy array
                coef = coef.reshape(-1)
            df_to_save["effect_weight"] = coef
            df_to_save["effect_type"] = "additive"
        else:
            self._logger.warning(
                "Coefficients cannot be aggregated for non-linear weak estimators. Columns 'effect_weight' and 'effect_type' will be empty."
            )

        df_to_save.to_parquet(annotation_file_path)
        if self.interactions is not None:
            interactions_file_path = Path(annotation_dir, "interactions.parquet")
            self._logger.info(f"Saving interactions file to {interactions_file_path}...")
            self.interactions.to_parquet(interactions_file_path)

    def _sample_batches(
        self,
        train_data: DataSet,
        agg_data: DataSet | None,
        val_data: DataSet | None,
        background_sample_idxs: npt.NDArray | None,
        batch_idx: int,
    ) -> Tuple[List[DataBatch], List[DataBatch], List[DataBatch], List[DataBatch | None]]:
        n_batches = self.train_cfg.batch_size if not self.cfg.is_ensemble else 1

        train_batches, agg_batches, val_batches, background_batches = [], [], [], []
        for i in range(n_batches):
            model_idx = (
                batch_idx if self.cfg.is_ensemble else batch_idx * self.train_cfg.batch_size + i
            )
            batch_seed = self.train_cfg.seed + model_idx

            sample_idxs, variant_idxs = train_data.sample_batch_idxs(
                batch_idx=model_idx, seed=batch_seed
            )

            train_batch = train_data[sample_idxs, variant_idxs]
            train_batches.append(train_batch)

            agg_batch = agg_data and agg_data[:, variant_idxs]
            agg_batches.append(agg_batch)

            val_batch = val_data and val_data[:, variant_idxs]
            val_batches.append(val_batch)

            background_batch = (
                train_data[background_sample_idxs, variant_idxs]
                if background_sample_idxs is not None
                else None
            )
            background_batches.append(background_batch)

        return train_batches, agg_batches, val_batches, background_batches

    def _preprocess_batch(
        self,
        train_batch: DataBatch,
        agg_batch: DataBatch | None,
        val_batch: DataBatch | None,
        background_batch: DataBatch | None,
        batch_idx: int,
    ) -> Tuple[DataBatch, DataBatch | None, DataBatch | None, DataBatch | None]:
        batch_seed = self.train_cfg.seed + batch_idx
        y_train = train_batch.get_labels(use_resids=self.cfg.use_resids)
        sample_mask = np.ones(len(y_train), dtype=bool)

        if self.cfg.preprocessing_config.z_score_thresh > 0.0:
            z_scores = np.abs(stats.zscore(y_train))
            sample_mask = np.where(z_scores < self.cfg.preprocessing_config.z_score_thresh)[0]

            train_batch.X = train_batch.X[sample_mask]
            train_batch.pheno_annotation = train_batch.pheno_annotation.iloc[sample_mask]
            train_batch.y = train_batch.y[sample_mask]
            train_batch.residuals = (
                train_batch.residuals[sample_mask] if train_batch.residuals is not None else None
            )
            y_train = y_train[sample_mask]

        if self.cfg.preprocessing_config.standard_labels:
            scaler = StandardScaler()
            y = y_train.reshape(-1, 1)
            scaler.fit(y)
            train_batch.scaler = scaler

        # Feature selection
        feature_mask = None
        if self.cfg.preprocessing_config.feature_selection.method != "none":
            fs_config = self.cfg.preprocessing_config.feature_selection
            X_train = train_batch.X.astype(np.float32, copy=False)

            # Choose score function based on config and task type
            if fs_config.score_func == "f_classif":
                score_func = f_classif
            elif fs_config.score_func == "f_regression":
                score_func = f_regression
            elif fs_config.score_func == "r_regression":
                score_func = r_regression
            elif fs_config.score_func == "chi2":
                score_func = chi2
            else:
                # Auto-select based on task type
                score_func = f_classif if self.cfg.model_config.classification else f_regression

            if fs_config.method == "k_best":
                k = min(fs_config.k, X_train.shape[1])  # Ensure k doesn't exceed number of features
                selector = SelectKBest(score_func=score_func, k=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                feature_mask = selector.get_support()

            elif fs_config.method == "percentile":
                selector = SelectPercentile(score_func=score_func, percentile=fs_config.percentile)
                X_train_selected = selector.fit_transform(X_train, y_train)
                feature_mask = selector.get_support()

            elif fs_config.method == "variance_threshold":
                selector = VarianceThreshold(threshold=fs_config.variance_threshold)
                X_train_selected = selector.fit_transform(X_train)
                feature_mask = selector.get_support()

            elif fs_config.method == "mutual_info":
                if self.cfg.model_config.classification:
                    mi_scores = mutual_info_classif(X_train, y_train, random_state=batch_seed)
                else:
                    mi_scores = mutual_info_regression(
                        X_train,
                        y_train,
                        n_neighbors=3,
                        discrete_features=True,
                        random_state=batch_seed,
                    )

                k = min(fs_config.k, X_train.shape[1])
                feature_indices = np.argsort(mi_scores)[-k:]
                feature_mask = np.zeros(X_train.shape[1], dtype=bool)
                feature_mask[feature_indices] = True
                X_train_selected = X_train[:, feature_mask]

            elif fs_config.method == "rfe":
                # Use a simple estimator for RFE
                if self.cfg.model_config.classification:
                    from sklearn.linear_model import LogisticRegression

                    estimator = LogisticRegression(random_state=batch_seed, max_iter=100)
                else:
                    from sklearn.linear_model import LinearRegression

                    estimator = LinearRegression()

                k = min(fs_config.k, X_train.shape[1])
                selector = RFE(estimator=estimator, n_features_to_select=k)
                X_train_selected = selector.fit_transform(X_train, y_train)
                feature_mask = selector.get_support()

            # Update the training batch with selected features
            train_batch.X = X_train_selected
            train_batch.geno_annotation = train_batch.geno_annotation.iloc[feature_mask]

            # Update the aggregation, validation, background, and test data with selected features
            if feature_mask is not None:
                if agg_batch is not None:
                    agg_batch.X = agg_batch.X[:, feature_mask]
                    agg_batch.geno_annotation = agg_batch.geno_annotation.iloc[feature_mask]
                if val_batch is not None:
                    val_batch.X = val_batch.X[:, feature_mask]
                    val_batch.geno_annotation = val_batch.geno_annotation.iloc[feature_mask]
                if background_batch is not None:
                    background_batch.X = background_batch.X[:, feature_mask]
                    background_batch.geno_annotation = background_batch.geno_annotation.iloc[
                        feature_mask
                    ]

        return train_batch, agg_batch, val_batch, background_batch

    def _get_model_cfgs(self) -> List[ModelConfig]:
        if self.cfg.is_ensemble:
            model_names = self.cfg.model_config.ensemble_estimator_names
            hyperparams = [
                self.cfg.model_config.hyperparameters.get(model_name, {})
                for model_name in model_names
            ]
        else:
            model_names = [self.cfg.model_config.model_name] * self.train_cfg.batch_size
            hyperparams = [self.cfg.model_config.hyperparameters] * self.train_cfg.batch_size

        model_cfgs = []
        based_model_cfg = self.cfg.model_config
        for name, params in zip(model_names, hyperparams):
            batch_model_cfg = replace(
                based_model_cfg,
                model_name=name,
                hyperparameters=params,
                n_jobs=self.train_cfg.n_jobs,
            )
            batch_model_cfg._initialize_from_catalog(overwrite=True)
            # In residualization mode, only LGBM with use_offset trains on original labels
            # and therefore retains the classification flag.  All other estimators train on
            # continuous residuals and must be treated as regression.
            if self.cfg.use_resids and not (
                batch_model_cfg.use_offset and batch_model_cfg.model_name == "lightgbm"
            ):
                batch_model_cfg.classification = False
            model_cfgs.append(batch_model_cfg)

        return model_cfgs

    @staticmethod
    def fit_weak_estimator(
        train_batch: DataBatch,
        agg_batch: DataBatch | None,
        val_batch: DataBatch | None,
        classification: bool,
        model_config: ModelConfig,
        train_cfg: TrainConfig,
        compute_interactions,
        use_resids: bool,
        background_batch: DataBatch | None,
    ):
        # fit weak estimator
        weak_estimator = WeakGenoEstimator(cfg=model_config)
        sample_weight = train_batch.get_sample_weights() if not use_resids else None
        weak_estimator.fit(
            train_batch,
            use_resids,
            sample_weight=sample_weight,
            compute_shap=train_cfg.compute_shap,
            compute_interactions=compute_interactions,
            background=background_batch,
            agg_batch=agg_batch,
            val_batch=val_batch,
            orig_classification=classification,
        )

        if agg_batch is not None:
            y_agg = agg_batch.get_labels()
            agg_preds = weak_estimator.predict(agg_batch)
            agg_score = utils.score(
                y_agg,
                agg_preds,  # no sigmoid bc prob space for cls
                train_cfg.scorer,
                classification,
            )
        else:
            agg_preds = None
            agg_score = None

        if val_batch is not None:
            y_val = val_batch.get_labels()
            weak_estimator_val_preds = weak_estimator.predict(val_batch)
            weak_estimator_val_score = utils.score(
                y_val,
                weak_estimator_val_preds,  # no sigmoid bc prob space for cls
                train_cfg.scorer,
                classification,
            )
        else:
            weak_estimator_val_preds = None
            weak_estimator_val_score = None

        return (
            weak_estimator,
            agg_preds,
            agg_score,
            weak_estimator_val_preds,
            weak_estimator_val_score,
        )

    def fit(
        self,
        train_data: DataSet,
        val_data: DataSet,
        train_cfg: TrainConfig | None = None,
    ):
        train_override_args = {"classification": train_data.cfg.classification}
        if train_data.cfg.compute_shap:
            train_override_args["compute_shap"] = True
        self.train_cfg = train_cfg or utils.init_class(cls=TrainConfig, **train_override_args)
        effective_batch_size = (
            len(self.cfg.model_config.ensemble_estimator_names)
            if self.cfg.is_ensemble
            else self.train_cfg.batch_size
        )
        self.cfg.model_config.n_jobs = max(self.train_cfg.n_jobs // effective_batch_size, 1)

        self.aggregator = GenoAggregator(
            self.cfg.model_config.classification,
            self.cfg.aggregator_config,
            self.train_cfg.scorer,
        )

        # split train into fit and aggregation sets before scoring
        if self.cfg.aggregator_config.hold_out_neeeded:
            print("Getting holdout")
            train_data, agg_data = split(train_data, seed=self.train_cfg.seed)
            train_data.setup()
            agg_labels = agg_data.get_labels(use_resids=self.cfg.use_resids)
            agg_sample_weight = agg_data.get_sampling_weight() if not self.cfg.use_resids else None

            self._logger.info(
                f"Propagating train annotation_df to eval sets (variants: {len(train_data.genotype.annotation_df)})"
            )
            if val_data is not None:
                val_data.genotype.annotation_df = train_data.genotype.annotation_df.copy()
        else:
            agg_data = None
            agg_labels = None
            agg_sample_weight = None

        # init running variables
        weak_estimators: List[WeakGenoEstimator] = []
        aggregation_preds: List[npt.NDArray] = []
        weak_estimator_val_preds: List[npt.NDArray] = []
        metrics = {
            "geno_aggregation_scores": [],
            "geno_strong_estimator_train": [],
            "geno_weak_estimator_val": [],
            "geno_strong_estimator_val": [],
        }
        # init early stopping variables
        current_best_val = float("-inf")
        patience_counter = 0
        best_iter = 0

        if self.cfg.is_ensemble:
            n_batches = self.cfg.n_estimators
            self.train_cfg.batch_size = len(self.cfg.model_config.ensemble_estimator_names)
        else:
            n_batches = math.ceil(self.cfg.n_estimators / self.train_cfg.batch_size)

        background_sample_idxs = (
            train_data.get_background_sample_idxs(n_samples=500, seed=self.train_cfg.seed)
            if self.train_cfg.compute_shap
            else None
        )

        progress_bar = tqdm(range(n_batches))
        for batch_idx in progress_bar:
            # sample batches
            train_batches, agg_batches, val_batches, background_batches = self._sample_batches(
                train_data, agg_data, val_data, background_sample_idxs, batch_idx
            )

            # preprocess batches if needed
            if self.cfg.preprocessing_needed:
                if self.train_cfg.batch_size > 1 and not self.cfg.is_ensemble:
                    # Parallel preprocessing for multiple batches
                    results = Parallel(
                        n_jobs=self.train_cfg.batch_size,
                        backend="threading",
                    )(
                        delayed(self._preprocess_batch)(
                            train_batches[i],
                            agg_batches[i],
                            val_batches[i],
                            background_batches[i],
                            batch_idx * self.train_cfg.batch_size + i,
                        )
                        for i in range(self.train_cfg.batch_size)
                    )
                    train_batches, agg_batches, val_batches, background_batches = zip(*results)
                    train_batches, agg_batches, val_batches, background_batches = (
                        list(train_batches),
                        list(agg_batches),
                        list(val_batches),
                        list(background_batches),
                    )
                else:
                    # Sequential preprocessing (for ensemble or single batch)
                    train_batch, agg_batch, val_batch, background_batch = self._preprocess_batch(
                        train_batches[0],
                        agg_batches[0],
                        val_batches[0],
                        background_batches[0],
                        batch_idx,
                    )
                    train_batches, agg_batches, val_batches, background_batches = (
                        [train_batch],
                        [agg_batch],
                        [val_batch],
                        [background_batch],
                    )

            model_cfgs = self._get_model_cfgs()

            if self.train_cfg.batch_size == 1:
                results = [
                    GenoEstimator.fit_weak_estimator(
                        train_batches[0],
                        agg_batches[0],
                        val_batches[0],
                        train_data.cfg.classification,
                        model_cfgs[0],
                        self.train_cfg,
                        False,  # computed post-training in _compute_and_aggregate_interactions
                        self.cfg.use_resids,
                        background_batches[0],
                    )
                ]
            elif self.cfg.is_ensemble:
                results = [
                    GenoEstimator.fit_weak_estimator(
                        train_batches[0],
                        agg_batches[0],
                        val_batches[0],
                        train_data.cfg.classification,
                        model_cfgs[i],
                        self.train_cfg,
                        False,  # computed post-training in _compute_and_aggregate_interactions
                        self.cfg.use_resids,
                        background_batches[0],
                    )
                    for i in range(self.train_cfg.batch_size)
                ]
            elif self.train_cfg.backend == "gpu":  # fit one model at a time if using gpu
                results = [
                    GenoEstimator.fit_weak_estimator(
                        train_batches[i],
                        agg_batches[i],
                        val_batches[i],
                        train_data.cfg.classification,
                        model_cfgs[i],
                        self.train_cfg,
                        False,  # computed post-training in _compute_and_aggregate_interactions
                        self.cfg.use_resids,
                        background_batches[i],
                    )
                    for i in range(self.train_cfg.batch_size)
                ]
            else:
                results = Parallel(n_jobs=self.train_cfg.batch_size, backend="threading")(
                    delayed(GenoEstimator.fit_weak_estimator)(
                        train_batches[i],
                        agg_batches[i],
                        val_batches[i],
                        train_data.cfg.classification,
                        model_cfgs[i],
                        self.train_cfg,
                        False,  # computed post-training in _compute_and_aggregate_interactions
                        self.cfg.use_resids,
                        background_batches[i],
                    )
                    for i in range(self.train_cfg.batch_size)
                )
            (
                batch_weak_estimators,
                batch_agg_preds,
                batch_agg_scores,
                batch_weak_estimator_val_preds,
                batch_weak_estimator_val_scores,
            ) = zip(*results)

            # add fit results to running variables
            weak_estimators.extend(batch_weak_estimators)
            aggregation_preds.extend(batch_agg_preds)
            weak_estimator_val_preds.extend(batch_weak_estimator_val_preds)
            metrics["geno_aggregation_scores"].extend(batch_agg_scores)
            metrics["geno_weak_estimator_val"].extend(batch_weak_estimator_val_scores)

            # calibrate aggregator using held-out agg set (never train or val)
            _group_size = self.train_cfg.batch_size if self.cfg.is_ensemble else 1
            if self.cfg.aggregator_config.hold_out_neeeded:
                self.aggregator.calibrate(
                    agg_scores=metrics["geno_aggregation_scores"],
                    agg_preds=aggregation_preds,
                    agg_labels=agg_labels,
                    agg_sample_weight=agg_sample_weight,
                    group_size=_group_size,
                )
            else:
                self.aggregator.calibrate(
                    agg_scores=metrics["geno_weak_estimator_val"],
                    group_size=_group_size,
                )

            if val_data:
                y_val = val_data.get_labels(use_resids=False)  # self.cfg.use_resids
                batch_strong_estimator_val_pred = self.aggregator(weak_estimator_val_preds)
                batch_strong_estimator_val_metric = utils.score(
                    y_val,
                    batch_strong_estimator_val_pred,  # no sigmoid bc prob space for cls
                    self.train_cfg.scorer,
                    val_data.cfg.classification,
                )
                metrics["geno_strong_estimator_val"].append(batch_strong_estimator_val_metric)

            # early stopping logic
            if self.train_cfg.patience > 0:
                if val_data is None:
                    raise ValueError("Cannot use early stopping without validation set!")

                if batch_strong_estimator_val_metric > current_best_val:
                    current_best_val = batch_strong_estimator_val_metric
                    best_iter = batch_idx
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.train_cfg.patience:
                    break  # recalibration after for looop

            # logging to wandb
            if self.train_cfg.log_with_wandb:
                strong_estimator_metrics = {
                    k: metrics[k][-1]
                    for k in [
                        "geno_strong_estimator_train",
                        "geno_strong_estimator_val",
                    ]
                    if metrics[k]
                }
                weak_estimator_metrics = {
                    k: metrics[k][-self.train_cfg.batch_size :]
                    for k in ["geno_weak_estimator_val", "geno_aggregation_scores"]
                    if metrics[k]
                }

                estimator_idx = batch_idx * self.train_cfg.batch_size
                for i in range(self.train_cfg.batch_size):
                    log_payload = {"estimator_idx": estimator_idx + i}
                    if strong_estimator_metrics and i == 0:
                        log_payload.update(strong_estimator_metrics)
                    log_payload.update({k: v[i] for k, v in weak_estimator_metrics.items()})
                    # For ensemble: also log metrics tagged by estimator type
                    if self.cfg.is_ensemble:
                        model_name = batch_weak_estimators[i].cfg.model_name
                        log_payload["estimator_type"] = model_name
                        for k, v in weak_estimator_metrics.items():
                            log_payload[f"{k}_{model_name}"] = v[i]
                    wandb.log(log_payload, step=estimator_idx + i)

            # update progress bar
            if val_data:
                latest_weak_metric = sum(metrics["geno_weak_estimator_val"]) / len(
                    metrics["geno_weak_estimator_val"]
                )
                latest_strong_metric = batch_strong_estimator_val_metric
                desc = (
                    f"Batch={batch_idx}: Avg weak val {self.train_cfg.scorer}={latest_weak_metric:.4f} - "
                    f"Strong val {self.train_cfg.scorer}={latest_strong_metric:.4f}"
                )
                desc += f" - Trained={(batch_idx + 1) * self.train_cfg.batch_size}"
                progress_bar.set_description(desc)
            elif metrics["geno_aggregation_scores"]:
                latest_weak_metric = (
                    sum(metrics["geno_aggregation_scores"][-self.train_cfg.batch_size :])
                    / self.train_cfg.batch_size
                )
                progress_bar.set_description(
                    f"Batch={batch_idx}: Avg weak agg {self.train_cfg.scorer}={latest_weak_metric:.4f} - "
                    f"Trained={(batch_idx + 1) * self.train_cfg.batch_size}"
                )
            else:
                progress_bar.set_description(
                    f"Batch={batch_idx} - Trained={(batch_idx + 1) * self.train_cfg.batch_size}"
                )

        if self.train_cfg.patience > 0 and val_data is not None:
            # Always use the best model
            best_n_estimators = (best_iter + 1) * self.train_cfg.batch_size
            if best_n_estimators < len(weak_estimators):
                self._logger.info(
                    f"Early stopping at batch {batch_idx + 1}. Best batch: {best_iter + 1} ({best_n_estimators} estimators)."
                )
                weak_estimators = weak_estimators[:best_n_estimators]
                # recalibrate aggregator on selected estimators
                _group_size = self.train_cfg.batch_size if self.cfg.is_ensemble else 1
                if self.cfg.aggregator_config.hold_out_neeeded:
                    self.aggregator.calibrate(
                        agg_scores=metrics["geno_aggregation_scores"][:best_n_estimators],
                        agg_preds=aggregation_preds[:best_n_estimators],
                        agg_labels=agg_labels,
                        agg_sample_weight=agg_sample_weight,
                        group_size=_group_size,
                    )
                else:
                    self.aggregator.calibrate(
                        agg_scores=metrics["geno_weak_estimator_val"][:best_n_estimators],
                        group_size=_group_size,
                    )

        # Discard weak estimators that did not pass the aggregator's filter (e.g.
        # top-p-percentile). Filtering used to be applied as a boolean mask threaded
        # through every downstream prediction/SHAP/interaction computation; instead we
        # physically drop the filtered-out estimators here -- the same way early
        # stopping above drops batches -- so every downstream consumer of self.model
        # only ever sees estimators that are actually used.
        n_final_estimators = len(weak_estimators)
        final_filter_mask = self.aggregator.filter_mask
        if final_filter_mask is not None and not final_filter_mask.all():
            n_keep = int(final_filter_mask.sum())
            self._logger.info(
                f"Filter strategy '{self.aggregator.cfg.filter_strat}' selected "
                f"{n_keep}/{n_final_estimators} weak estimators. Discarding the rest."
            )
            weak_estimators = [we for we, keep in zip(weak_estimators, final_filter_mask) if keep]
            filtered_agg_preds = [
                p
                for p, keep in zip(aggregation_preds[:n_final_estimators], final_filter_mask)
                if keep
            ]
            filtered_agg_scores = [
                s
                for s, keep in zip(
                    metrics["geno_aggregation_scores"][:n_final_estimators], final_filter_mask
                )
                if keep
            ]
            filtered_val_scores = [
                s
                for s, keep in zip(
                    metrics["geno_weak_estimator_val"][:n_final_estimators], final_filter_mask
                )
                if keep
            ]

            # The kept estimators already passed the filter, so recalibrate without
            # re-filtering: reset filter_mask and temporarily disable filter_strat so
            # the aggregation strategy (mean / loss-weighted / stacking) is refit on
            # exactly the surviving estimators instead of re-applying the threshold.
            self.aggregator.filter_mask = None
            original_filter_strat = self.aggregator.cfg.filter_strat
            self.aggregator.cfg.filter_strat = "none"
            try:
                if self.cfg.aggregator_config.hold_out_neeeded:
                    self.aggregator.calibrate(
                        agg_scores=filtered_agg_scores,
                        agg_preds=filtered_agg_preds,
                        agg_labels=agg_labels,
                        agg_sample_weight=agg_sample_weight,
                    )
                else:
                    self.aggregator.calibrate(agg_scores=filtered_val_scores)
            finally:
                self.aggregator.cfg.filter_strat = original_filter_strat

        # init final model
        self._init_model(weak_estimators)

        if self.cfg.compute_interactions and self.train_cfg.compute_shap:
            self._compute_and_aggregate_interactions(train_data, background_sample_idxs)

    def _compute_and_aggregate_interactions(
        self,
        train_data: DataSet,
        background_sample_idxs: npt.NDArray | None,
        n_samples_shap: int = 2_000,
        chunk_size: int = 20,
    ) -> None:
        """Compute SHAP interaction values post-training for surviving tree estimators."""
        tree_model_pairs = [
            (i, m)
            for i, m in enumerate(self.model)
            if m.cfg.model_name in ["lightgbm", "xgboost", "catboost"]
        ]
        if not tree_model_pairs:
            return

        filter_mask = self.aggregator.filter_mask
        if filter_mask is not None:
            tree_model_pairs = [(i, m) for i, m in tree_model_pairs if filter_mask[i]]
        if not tree_model_pairs:
            return

        all_sample_idxs = train_data.phenotype.sample_idxs

        self._logger.info(
            f"Computing interaction values for {len(tree_model_pairs)} tree estimators sequentially..."
        )

        interact_dfs = []
        pbar = tqdm(total=len(tree_model_pairs), desc="Computing interaction values")
        try:
            for global_i, model in tree_model_pairs:
                explain_batch = train_data[all_sample_idxs, model.variant_idxs]
                bg_batch = (
                    train_data[background_sample_idxs, model.variant_idxs]
                    if background_sample_idxs is not None
                    else None
                )
                model.compute_interaction_values(
                    explain_batch,
                    background=bg_batch,
                    n_samples_shap=n_samples_shap,
                    chunk_size=chunk_size,
                )
                variant_idxs = model.variant_idxs
                interact_values = np.triu(model.interactions, k=1)
                model.interactions = None
                nz_rows, nz_cols = np.nonzero(interact_values)
                if len(nz_rows) > 0:
                    interact_dfs.append(
                        pd.DataFrame(
                            {
                                "variant_i": variant_idxs[nz_rows],
                                "variant_j": variant_idxs[nz_cols],
                                "interact_values": interact_values[nz_rows, nz_cols],
                                "estimator_idx": global_i,
                            }
                        )
                    )
                del interact_values
                pbar.update(1)
        finally:
            pbar.close()

        if not interact_dfs:
            return

        interact_df = pd.concat(interact_dfs, ignore_index=True)
        interact_df[["variant_i", "variant_j"]] = interact_df[["variant_i", "variant_j"]].apply(
            lambda x: pd.Series(sorted(x)), axis=1
        )

        interact_df = (
            interact_df.groupby(["variant_i", "variant_j"], sort=True)
            .apply(
                lambda grp: self.aggregator.aggregate_group(
                    group=grp,
                    columns=["interact_values"],
                    n_estimators=len(self.model),
                    aggregate_coeffs=False,
                    filter_mask=filter_mask,
                ),
                include_groups=False,
            )
            .sort_index()
            .reset_index()
        )

        interact_df = interact_df.merge(
            self.annotation_df.rename(
                columns={"chr_name": "chr_i", "chr_position": "pos_i", "snp": "snp_i"}
            )[["chr_i", "pos_i", "snp_i"]],
            left_on="variant_i",
            right_index=True,
            how="left",
        ).merge(
            self.annotation_df.rename(
                columns={"chr_name": "chr_j", "chr_position": "pos_j", "snp": "snp_j"}
            )[["chr_j", "pos_j", "snp_j"]],
            left_on="variant_j",
            right_index=True,
            how="left",
        )
        self.interactions = interact_df

    def _weak_estimator_predict(
        self,
        weak_estimator: WeakGenoEstimator,
        data_set: DataSet | None = None,
        data_batch: DataBatch | None = None,
        sample_idxs: npt.NDArray | slice = slice(None),
    ) -> npt.NDArray:
        if data_batch is None:
            if data_set is None:
                raise ValueError("Need to provdie at least one, 'data_batch' or 'data_set'.")
            weak_estimator_variants = weak_estimator.variant_idxs
            data_batch = data_set[sample_idxs, weak_estimator_variants]

        pred = weak_estimator.predict(data_batch)

        return pred

    def _linear_weak_estimator_safe_predict(
        self, data_set: DataSet, v_chunk_min: int = 4096
    ) -> npt.NDArray:
        """
        Fast memory-safe prediction for aggregated linear models by streaming variant chunks.
        Avoids constructing the full (n_samples x n_variants) matrix.
        """
        assert (
            isinstance(self.model, list) and len(self.model) == 1
        ), "Aggregated linear model expected"
        weak_estimator: WeakGenoEstimator = self.model[0]

        self._logger.info(f"Using {self.cfg.model_config.ram_mb} MB of storage for inference")

        # get non-zero coefficients (speeds up for L1) and intercept
        coef = weak_estimator.model.coef_
        if hasattr(coef, "_owner"):
            # cuml access pattern
            coef = coef._owner.get()
        elif hasattr(coef, "values"):
            # sklearn/pandas access pattern
            coef = coef.values.reshape(-1)
        else:
            # numpy array
            coef = coef.reshape(-1)
        intercept = (
            weak_estimator.model.intercept_.item()
            if np.ndim(weak_estimator.model.intercept_)
            else float(weak_estimator.model.intercept_)
        )

        nonzero_mask = coef != 0
        # if all coefs zero, return intercept for all samples
        if not np.any(nonzero_mask):
            base = np.full(len(data_set), intercept, dtype=np.float32)
            if self.cfg.model_config.classification:
                base = utils.sigmoid(base)
            return base
        nonzero_coefs = coef[nonzero_mask].astype(np.float32, copy=False)
        nonzero_variant_idxs = weak_estimator.variant_idxs[nonzero_mask].astype(
            np.uint32, copy=False
        )

        # compute size of chunk considering ram budget
        max_bytes = int(float(self.cfg.model_config.ram_mb) * 1024**2)
        n_samples = len(data_set)
        n_vars = len(nonzero_variant_idxs)
        s_chunk = max(1, min(n_samples, max_bytes // (5 * v_chunk_min)))

        all_preds: List[npt.NDArray] = []
        all_sample_idxs = data_set.phenotype.sample_idxs
        for sample_start_idx in range(0, n_samples, s_chunk):
            sample_end_idx = min(n_samples, sample_start_idx + s_chunk)
            batch_sample_idxs = all_sample_idxs[np.arange(sample_start_idx, sample_end_idx)]
            n_samples_batch = len(batch_sample_idxs)
            batch_preds = np.zeros(n_samples_batch, dtype=np.float32)

            # compute variant chunk size given memory budget for float32 cast (4 bytes)
            denom = max(1, (n_samples_batch) * 4)
            v_chunk = max(1, min(n_vars, max_bytes // denom))

            # stream over variants
            for variant_start_idx in range(0, n_vars, v_chunk):
                variant_end_idx = min(n_vars, variant_start_idx + v_chunk)
                batch_varaint_idxs = nonzero_variant_idxs[variant_start_idx:variant_end_idx]
                batch_coefs = nonzero_coefs[variant_start_idx:variant_end_idx]

                batch = data_set[batch_sample_idxs, batch_varaint_idxs]
                batch_preds += batch.X.astype(np.float32) @ batch_coefs
                del batch
            # add intercept
            batch_preds += intercept

            all_preds.append(batch_preds)

        return np.concatenate(all_preds, axis=0)

    def predict(
        self, data_set: DataSet, return_resids: bool = False, min_batch_size: int = 1
    ) -> Tuple[npt.NDArray, npt.NDArray | None, "dict[str, npt.NDArray] | None"]:
        # predict with weak estimators
        if (self.cfg.model_config.model_type == "linear") and (not self.cfg.is_ensemble):
            preds = self._linear_weak_estimator_safe_predict(data_set)
            per_type_preds = None
        elif self.train_cfg.backend == "gpu":
            all_preds = []
            model_iter = tqdm(
                range(len(self.model)),
                desc="Predict (Weak estimators)",
                leave=False,
            )
            for i in model_iter:
                preds = self._weak_estimator_predict(self.model[i], data_set)
                all_preds.append(preds)

            preds = self.aggregator(all_preds)
            per_type_preds = self._compute_per_type_preds(all_preds)
        else:
            all_preds = []
            step = (
                max(self.train_cfg.batch_size, min_batch_size)
                if not self.cfg.is_ensemble
                else self.train_cfg.batch_size
            )
            total_batches = math.ceil(len(self.model) / step)
            batch_iter = tqdm(
                range(0, len(self.model), step),
                total=total_batches,
                desc="Predict (Weak estimators)",
                leave=False,
            )
            for i in batch_iter:
                batch_estimators = self.model[i : i + step]
                if self.cfg.is_ensemble:
                    variant_idxs = batch_estimators[0].variant_idxs
                    data_batch = data_set[:, variant_idxs]

                    batch_preds = []
                    for weak_estimator in batch_estimators:
                        # do not use parallelism since data is the same
                        pred = self._weak_estimator_predict(weak_estimator, data_set, data_batch)
                        batch_preds.append(pred)
                else:
                    batch_preds = Parallel(
                        n_jobs=self.train_cfg.n_jobs,
                        batch_size=step,
                    )(
                        delayed(self._weak_estimator_predict)(
                            weak_estimator,
                            data_set,
                        )
                        for weak_estimator in batch_estimators
                    )
                all_preds.extend(batch_preds)

            preds = self.aggregator(all_preds)
            per_type_preds = self._compute_per_type_preds(all_preds)

        resid_preds = preds if return_resids else None
        geno_preds = preds  # no sigmoid bc prob space for cls

        return geno_preds, resid_preds, per_type_preds

    def _compute_per_type_preds(
        self, all_preds: List[npt.NDArray]
    ) -> "dict[str, npt.NDArray] | None":
        if not self.cfg.is_ensemble:
            return None
        type_pred_lists: dict[str, list] = {}
        for i, pred in enumerate(all_preds):
            name = self.model[i].cfg.model_name
            type_pred_lists.setdefault(name, []).append(pred)
        return {name: np.mean(preds_list, axis=0) for name, preds_list in type_pred_lists.items()}

    def _aggregate_local_shap_chunks(
        self,
        shap_dfs: List[pd.DataFrame],
        n_estimators: int,
        chunk_size: int,
        filter_mask: npt.NDArray | None = None,
    ) -> pd.DataFrame:
        concat_df = pd.concat(shap_dfs, ignore_index=False)
        variant_idxs = [c for c in concat_df.columns if c != "estimator_idx"]
        chunks = []
        print(
            f"Aggregating chunk with n_estimatos={n_estimators} and filter_mask of shap {filter_mask.shape if filter_mask is not None else 'None'}"
        )
        for start_idx in range(0, len(variant_idxs), chunk_size):
            end_idx = min(start_idx + chunk_size, len(variant_idxs))
            chunk_variant_idxs = variant_idxs[start_idx:end_idx]
            chunk_columns = chunk_variant_idxs + ["estimator_idx"]
            chunk_df = concat_df[chunk_columns].copy()
            chunk_shap_df = (
                chunk_df.groupby(level=0, sort=True)
                .apply(
                    lambda grp: self.aggregator.aggregate_group(
                        group=grp,
                        n_estimators=n_estimators,
                        columns=chunk_variant_idxs,
                        aggregate_coeffs=True,
                        filter_mask=filter_mask,
                    ),
                    include_groups=False,
                )
                .sort_index()
            )
            chunks.append(chunk_shap_df)
        return pd.concat(chunks, axis=1)

    def compute_local_shap(
        self,
        data_set: DataSet,
        background_samples: int = 500,
        seed: int = 42,
        *,
        sample_idxs: npt.NDArray | slice = slice(None),
        chunk_size: int = 2_000,
    ) -> dict[str, pd.DataFrame]:
        sample_idxs = np.sort(sample_idxs).astype(np.uint32)
        assert (
            len(data_set) >= background_samples
        ), "Dataset has to be larger than requested number of background samples"

        background_sample_idxs = data_set.get_background_sample_idxs(
            n_samples=background_samples, seed=seed
        )

        we_shap_values = []
        batch_size = 2 * self.train_cfg.batch_size
        n_batches = math.ceil(len(self.model) / batch_size)
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.model))
            batch_we_shap_values = Parallel(n_jobs=self.train_cfg.n_jobs, backend="threading")(
                delayed(self.model[i].compute_shap_values)(
                    data_set[sample_idxs, self.model[i].variant_idxs],
                    type="local",
                    background=data_set[background_sample_idxs, self.model[i].variant_idxs],
                )
                for i in range(start_idx, end_idx)
            )
            we_shap_values.extend(batch_we_shap_values)

        we_shap_dfs = []
        for i, model in enumerate(self.model):
            we_shap_df = pd.DataFrame(
                we_shap_values[i], columns=model.variant_idxs, index=sample_idxs
            )
            we_shap_df["estimator_idx"] = i
            we_shap_dfs.append(we_shap_df)

        ensemble_df = self._aggregate_local_shap_chunks(we_shap_dfs, len(self.model), chunk_size)
        ensemble_df["iid"] = ensemble_df.index.map(data_set.phenotype.annotation_df["iid"])
        result = {"ensemble": ensemble_df}

        if self.cfg.is_ensemble:
            linear_idxs = [i for i, m in enumerate(self.model) if m.cfg.model_type == "linear"]
            nonlinear_idxs = [i for i, m in enumerate(self.model) if m.cfg.model_type != "linear"]
            for group_name, group_idxs in [("linear", linear_idxs), ("nonlinear", nonlinear_idxs)]:
                if not group_idxs:
                    continue
                # Build type_mask (global length) and AND with filter_mask so that
                # filtered-out estimators remain excluded in the per-type aggregation.
                type_mask = np.zeros(len(self.model), dtype=bool)
                for global_i in group_idxs:
                    type_mask[global_i] = True
                group_filter_mask = (
                    type_mask & self.aggregator.filter_mask
                    if self.aggregator.filter_mask is not None
                    else type_mask
                )
                group_shap_dfs = []
                for global_i in group_idxs:
                    df = pd.DataFrame(
                        we_shap_values[global_i],
                        columns=self.model[global_i].variant_idxs,
                        index=sample_idxs,
                    )
                    df["estimator_idx"] = global_i
                    group_shap_dfs.append(df)
                group_df = self._aggregate_local_shap_chunks(
                    group_shap_dfs, len(self.model), chunk_size, filter_mask=group_filter_mask
                )
                group_df["iid"] = group_df.index.map(data_set.phenotype.annotation_df["iid"])
                result[group_name] = group_df

        return result

    def get_interactions(
        self,
        data_set: DataSet,
        *,
        background: DataBatch | None = None,
        sample_idxs: npt.NDArray | slice = slice(None),
    ):
        tree_models = [
            model
            for model in self.model
            if model.cfg.model_name in ["lightgbm", "xgboost", "catboost"]
        ]
        if not tree_models:
            raise ValueError(
                "Can only compute interactions for tree-based models. No tree-based models found!"
            )

        if self.annotation_df is None or "shap_values" not in self.annotation_df:
            raise ValueError(
                "SHAP values not found in model anootation data. Model must be trained with 'train_cfg.compute_shap = True' to compute interaction values"
            )

        """abs_shap_values = np.abs(self.annotation_df["shap_values"].values)
        percentile_threshold = np.percentile(abs_shap_values, p)
        top_snp_mask = abs_shap_values >= percentile_threshold
        top_variant_idxs = self.annotation_df.index.values[top_snp_mask]"""

        # self._logger.info(f"Computing interaction values for {len(top_variant_idxs)} SNPs using {len(tree_models)} tree models...")
        all_we_interactions = []
        for model in tree_models:
            explain_batch = data_set[sample_idxs, model.variant_idxs]
            we_interactions = model.compute_interaction_values(explain_batch, background)
            all_we_interactions.append(we_interactions)
