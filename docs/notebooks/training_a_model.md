# Training a GenomEn model

## Configuring your training run

Genomen uses a YAML-based configuration system that provides a declarative way to define all aspects of your experiment. Again, we provide a [template config](https://github.com/AI-sandbox/MetaPRS/blob/main/config.yml.template) with default values:

```bash
---
DataSetConfig:
  phenotype_id: "HC337"             # must be column master file
  classification: true              # true if phenotype is binary else false
  file_format: "plink"              # only supports plink for now
  populations: ["white_british"]    # list of population groups to include
  include_x_chromosome: false       # whether to include variants from the X chromosome
  maf_threshold: 0.05               # minimum minor allele frequency (MAF) required for variants to be retained
  sex: null                         # optional filtering by sex ("m" for male, "w" for female, or null for both)
  covar_config:
    include_covars: false           # whether to include covariates at all in the model
    covar_keys: ["age", "sex", "Global_PC1", "Global_PC2", "Global_PC3", "Global_PC4", "Global_PC5", "Global_PC6", "Global_PC7", "Global_PC8", "Global_PC9", "Global_PC10"]
  sample_sampling:
    strat: "stratify"               # strategy for sampling individuals; "random" or "stratify" (k:1 balanced classes)
    max_samples: 50                 # maximum number of samples to include per patch (higher -> better)
    balance_pops: false             # whether to balance the number of samples per patch across populations
  variant_sampling:
    strat: "random"                 # options: "random", "window", "LD", "GWAS"
    max_features: 50                # maximum number of variants (features) to include per patch. Used as window size for strat = "window"
    ld_config:                      # used when strat = "LD"
      prune_kb: 250                 # distance window in kilobases for LD pruning
      prune_step: 50                # step size (in variants) used during pruning
      prune_r2: 0.1                 # LD threshold for pruning
      tau: 0.1
      ld_window_kb: 1_000           # max window size for LD blocks in kb
      ld_window: 50_000             # max window size for LD blocks in number of variants
      eps: 0.0                      # epsilon parameter for epsilon-greedy sampling
      eps_schedule: "constant"      # epsilon annilation schedule ("constant" or "step")
      eps_step_size: 0.0            # step size for epsilon updates (if applicable)
    gwas_config:                    # used when strat = "GWAS"
      path: ""                      # path to a GWAS summary statistics file
      snps_column: "variant_id"     # column name for SNP identifiers
      pvalue_column: "pvalue"       # column name for raw p values
      nlogpvalue_column: "LOG10P"   # column name for negative log of p values (will be used if provided)
      sep: '\s+'                    # delimiter (space by default)
      impute_val: 0.1               # imputation value for variants not in GWAS
    window_overlap_ratio: 0.0       # stride as percentage of window size for window sampling
---
GenomenModelConfig:
  covar_config:                     # configuration of covariate model
    covar_strat: "residualization"  # "residualization" or "predictive"
    model_config:                   # config of model used for covariate prediction
      model_name: lightgbm          # model name, check genomen/model/models.json for overview of available model
      hyperparameters: {}           # hyperparameters of model
      balance_classes: true         # whether to balance classes in estimator loss
  geno_config:                      # configuration of genotype model
    n_estimators: 2                 # number of estimators
    compute_interactions: True      # whether to compute interaction values at training time (ca. 2x training time)
    preprocessing_config:           
      z_score_thresh: 3.0           # z-score threshold to filter outliers
      standard_labels: false        # whether to standardize labels
      feature_selection:
        method: "none"              # options: "none", "k_best", "percentile", "variance_threshold", "mutual_info", "rfe"
        k: 15_000                   # number of variants selected in case of method="topk"
        percentile: 0.75            # percentile of variants selected in case of method="percentile"
        variance_threshold: 0.05    # variance_threshold for variants in case of method="variance_threshold"
        score_func: "f_classif"     # scoring function used scoring ("f_classif", "f_regression", or "chi2")
    model_config:
      model_name: lightgbm          # model name, check genomen/model/models.json for overview of available model
      ensemble_estimator_names: []  # names of weak estimator models to be used in ensemble (model_name="ensemble")
      hyperparameters: {}           # hyperparameters of model
      balance_classes: true         # whether to balance classes in estimator loss
    aggregator_config:
      filter_strat: "geq-average"   # filtering strategy ("none", "positive", "geq-average", "top-p-percentile")
      agg_stat: "rank-mean"         # aggregation strategy ("mean", "loss-weighted-average", "stacking")
      model_config: 
        model_name: lightgbm        # model name of stacking model, check genomen/model/models.json for overview of available model
        hyperparameters: {}         # hyperparameters of model
        balance_classes: true       # whether to balance classes in estimator loss
      p: 0.75                       # p used for top-p-percentile filtering
      temp: 0.05                    # temperature temp used for softmax in filter_strat="loss-weighted-average"
---
TrainConfig:
  batch_size: 2                     # number of models trained in parallel
  n_jobs: 32                        # number of jobs that can be run in paralell
  backend: "cpu"                    # backend to use ("cpu" or "gpu")
  ram_mb: 16000                     # available ram
  scorer: "rocauc"                  # scoring function for early stopping ("r2", "rocauc", "pearson_corr")
  patience: 30                      # patience in number of batches
  seed: 42                          # seed for reproducability
  log_with_wandb: false             # whether to log with wanbd
  save_annotation: false            # whether to save annotation files (e.g., effect sizes or vairant importance) to file 
  save_model: false                 # whether to save model artifacts
  compute_shap: false               # whether to compute shap values

The configuration is divided into three main sections that control dataset preparation, model behavior, and training parameters:
- DataSetConfig defines the dataset used for training, including the phenotype to predict, input file format, populations, covariates, and sampling strategies for both samples and genetic variants (e.g., LD-based or GWAS-based selection).
- GenomenModelConfig specifies the architecture and hyperparameters of the models used to process covariates and genotypes, as well as the ensemble and aggregation strategies for combining model outputs.
- TrainConfig controls how models are trained and evaluated, including compute backend, batch size, number of jobs, evaluation metric, early stopping, and logging options.

Together, these sections provide a flexible way to reproduce or customize GenomEn experiments—from input preprocessing to model training and interpretation.

## Setting up training run

Now, before we can train the model we have to point the software to the location of the YAML file detailing the desired configuration. This is done via the one-liner:


```python
import genomen.utils as utils

utils.set_config_path("config.yml")
```

    ########## Welcome to Genomic Ensembling (GenomEn) - Polygenic risk and association beyond linearity ##########


## Loading the dataset

Once everything is set up, we can load the dataset via the `DataSet` class. The helper function `split` allows to split a `DataSet` into train and test splits (randomly or via a pre-defined split column in the master table).


```python
from genomen.data import DataSet, split

dataset = DataSet()
train_set, test_set, val_set = split(dataset, split_by_col=("split", ("train", "test", "val")))
```

    INFO:DataSet:Looking for cached dataset...
    INFO:DataSet:Found cached dataset. Proceeding to loading data...
    INFO:DataSet:Got 479 cases in the train set (0.20 %). Balancing with k=5 (4790 samples per batch).


## Training the model

Finally, we can initialize the `GenomenModel` and fit it on the data.


```python
from genomen.model import GenomenModel

model = GenomenModel()
model.fit(train_set, val_set)
```

    INFO:genomen.model.model:Fitting covar model...
    INFO:genomen.model.model:Validation covar-only score: 0.7755
    INFO:genomen.model.model:Fitting geno model...
    INFO:DataSet:Got 390 cases in the train set (0.21 %). Balancing with k=5 (3900 samples per batch).


    Got train_data with 235991 samples


    Batch=7: Avg weak rocauc=0.4992 - Strong rocauc=0.5519 - Trained=16: 100%|███████████████████████| 8/8 [04:04<00:00, 30.62s/it]
    INFO:GenoEstimator:Early stopping at batch 8. Best batch: 6 (12 estimators).


## Making prediction

Once the model is trained we can simply use it to make predictions on new data


```python
geno_preds, covar_preds, preds = model.predict(test_set)
```
