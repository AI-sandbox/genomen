# Getting started with GenomEn

This is a short tutorial to get started with the [genomen](https://pypi.org/project/genomen/) Python package. Here, we briefly explain how to set-up GenomEn to train models on yout own data. For more in-depth information on GenomEn please refer to the [paper]().

## Install with pip

Genomen is available on PyPI and can be installed directly without cloning the repository. This is the recommended approach for users who just want to use the package.


```
pip install genomen
```

Genomen provides optional dependency groups for extended functionality. Depending on your package manager, the installation method differs. The signature to install a dependency group `dep-group` is:


```
pip install genomen[dep-group]
```

GenomEn supports dependency groups `dev`, `gpu`, and `dnn`. Details on each dependency group can be found in the following install from source section. 

## Install from source

For users that want to contribute to GenomEn or adapt the code base for their own purposes, we recommend to install the package from [source](https://github.com/AI-sandbox/MetaPRS/tree/1bedf812b90137bf2935d5a487d44f9780f6d738).

### Download uv (if needed)

We decide to use [uv](https://docs.astral.sh/uv/) for dependency management. If you do not have uv installed, you will have to do so before getting started. The following command installs uv:


```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

Next, you will have to install the project dependencies. You can find an overview of the dependencies in the [pyproject.toml](https://github.com/AI-sandbox/MetaPRS/blob/main/pyproject.toml) file. We differentiate between 4 different groups of dependencies, each for specific use cases.

### Default dependencies


```
uv sync
```

This group includes the default dependencies that have to be installed in any case. These include basic framework like numpy and pandas, and base estimator dependencies such as LGBM and XGBoost.

### Dev dependencies

This group includes dependencies that are not necessarily needed for usage of the package but will be essential for contributors actively working on the project as it includes additional dependencies such as black and pytest required for new PRs.


```
uv sync --group dev
```

### GPU dependencies

As the name suggests, this dependency group is needed to run genomen on GPU. It adds GPU-aware implementations of many methods used by GenomEn such as [cuML](https://docs.rapids.ai/api/cuml/stable/). Note, that this only works for NVIDIA GPU and more specifically GPUs running on CUDA version 12. If your GPU device uses a different verion, please fork the repository and adapt the requirements or do a feature request.


```
uv sync --group gpu
```

### DNN dependencies

This dependency group allows to train GenomEn with custom deep neural networks as weak estimators (```backend="gpu"```). For an overview of pre-implemented DNNs, check [here](https://github.com/AI-sandbox/MetaPRS/tree/main/genomen/model/custom).


```
uv sync --group dnn
```

It is alway possible to downlaod multiple dependency groups simultaneously or to download them all at the same time via


```
uv sync --all-groups
```

## Setting up the environment

Genomen uses environment variables loaded from a `.env` file (via `python-dotenv`). The repo already contains a `.env.template` file with the following variables

```bash
# plink 1 files
FAM_PATH=""
BED_PATH=""
MASTER_PATH=""
BIM_PATH=""

# result path
RESULT_PATH="./genomen_artifacts"
```

The first block of variables configures the data source from which genotype and covariate data will be loaded. For now, GenomEn only supports the Plink 1 input data format (.fam, .bed, .bim). If your data has a different format, please use external tools such as [plink2](https://www.cog-genomics.org/plink/2.0/). The master file should be a tab-delimited text file containing phenotype information for each sample (identified by `IID`). <p>
The result path is pre-configured to `./genomen_artifacts` but can be changed if needed. Model artifacts like generated annotation files and model checkpoints can be found in that location. <p>
Use the following command to create your own `.env` file and subsequently update the respective fields for GenomEn to work.


```
cp .env.template .env
```

Genomen uses a YAML-based configuration system that provides a declarative way to define all aspects of your experiment. Again, we provide a [template config](https://github.com/AI-sandbox/MetaPRS/blob/main/config.yml.template) with default values:
