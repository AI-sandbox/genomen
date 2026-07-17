<div align="center">

# GenomEn

**Biobank-scale Polygenic Risk Prediction with Nonlinear Estimators**

[![PyPI version](https://img.shields.io/pypi/v/genomen.svg)](https://pypi.org/project/genomen/)
[![PyPI downloads](https://img.shields.io/pypi/dm/genomen.svg)](https://pypi.org/project/genomen/)
[![Website](https://img.shields.io/badge/🌐%20Website-visit-blue)](https://genomen-website.vercel.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Format Check](https://github.com/AI-sandbox/genomen/actions/workflows/format-check.yml/badge.svg)](https://github.com/AI-sandbox/genomen/actions/workflows/format-check.yml)

</div>

## Overview

Genomic Ensembling (GenomEn) is an ensemble framework for genotype-to-phenotype prediction that uses both linear and non-linear estimators to potentially capture gene-gene interactions often overlooked by traditional polygenic risk score (PRS) methods. For more informations on the methods, please refer to our [paper](https://github.com/AI-sandbox). 

The package enables researchers to improve predictive performance beyond conventional linear PRS approaches by modeling higher-order genetic interactions. GenomEn also natively supports variants on the X sex chromosome, which are often neglected due to integration challenges with autosomes, further improving predictive performance and simplifying the study of X-linked traits. Finally, GenomEn allows for local and global variant-level interpretability via [SHAP](https://arxiv.org/abs/1705.07874) values, allowing to gain new insights into complex traits.

## Installation

Install from PyPI:

```bash
pip install genomen
```

Install with optional dependency groups:

```bash
# Development dependencies (black, pytest, etc.)
pip install genomen[dev]

# GPU support (CUDA 12)
pip install genomen[gpu]
```

GenomEn also requires [PLINK](https://www.cog-genomics.org/plink/) for some genotype computations. A Linux x86-64 binary is bundled at [assets/plink](assets/plink) and used automatically; if it doesn't work on your platform, install PLINK yourself and replace that file with your own executable.

A clean `pip install genomen` takes about 90 seconds on average (measured across 10 installs in a fresh virtual environment). It's mostly download-bound (lightgbm, xgboost, catboost, shap, pandas, etc.), so it'll be quicker on a faster network connection, and installing via [uv](https://docs.astral.sh/uv/) (`uv pip install genomen`) cut this to ~16 seconds in our testing.

## Quick Start

The repo ships a small, fully-synthetic demo dataset (see [data/demo/](data/demo/)) so you can try GenomEn right away, without your own genotype data.

Point `.env` at the demo files:

```bash
cp .env.template .env
```

```bash
# in .env
FAM_PATH="data/demo/demo.fam"
BED_PATH="data/demo/demo.bed"
BIM_PATH="data/demo/demo.bim"
MASTER_PATH="data/demo/demo_master.phe"
```

Then train and predict using the demo dataset's matching config ([docs/configs/demo.yml](docs/configs/demo.yml)):

```python
import genomen.utils as utils
from genomen.data import DataSet, split
from genomen.model import GenomenModel

utils.set_config_path("docs/configs/demo.yml")

# Load and split data
dataset = DataSet()
train_set, test_set, val_set = split(dataset, split_by_col=("split", ("train", "test", "val")))

# Train model
model = GenomenModel()
model.fit(train_set, val_set)

# Make predictions
geno_preds, covar_preds, preds = model.predict(test_set)
```

Or equivalently, run the training script directly from the command line:

```bash
python docs/scripts/train.py --cfg_path=docs/configs/demo.yml
```

To use your own data, follow the [getting started guide](docs/notebooks/getting_started.md) and point `.env` / `config.yml` at your own PLINK files instead.

### Simulating a trait

To test GenomEn against a phenotype with known genetic/covariate architecture (instead of a real or demo trait), use [docs/scripts/simulate.py](docs/scripts/simulate.py). It simulates a binary or continuous phenotype from your genotype data with configurable heritability, trains a model, and reports test-set performance:

```bash
python docs/scripts/simulate.py --cfg_path=docs/configs/demo_sim.yml --task=cls  # or --task=reg for a continuous trait
```

## Documentation

For detailed documentation, tutorials, and examples, please visit the [official documentation site](https://genomen-website.vercel.app/docs) or browse the local documentation in the [docs/](docs/) directory.

## Citation

If you use GenomEn in your research, please cite:

```bibtex
@article{Thomassin2026,
  title   = {Biobank-scale Polygenic Risk Prediction with Nonlinear Estimators},
  author  = {Thomassin, Christophe and Franquesa Mon{\'e}s, Marc and Bonet, David and Gerlach, Peter A. and Comajoan Cara, Mar{\c{c}}al and Mas Montserrat, Daniel and Ioannidis, Alexander G.},
  year    = {2026},
  url     = {https://genomen-website.vercel.app/docs}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For development setup, see the [getting started guide](docs/notebooks/getting_started.md).

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

## Links

- **Website**: https://genomen-website.vercel.app/
- **Documentation**: https://genomen-website.vercel.app/docs
- **Phenotype Browser**: https://genomen-website.vercel.app/browser
- **PyPI**: https://pypi.org/project/genomen/
- **GitHub**: https://github.com/AI-sandbox/genomen
