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

## Quick Start

```python
from genomen.data import DataSet, split
from genomen.model import GenomenModel

# Load and split data
dataset = DataSet()
train_set, test_set, val_set = split(dataset)

# Train model
model = GenomenModel()
model.fit(train_set, val_set)

# Make predictions
geno_preds, covar_preds, preds = model.predict(test_set)
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
