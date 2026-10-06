# How to install

## From PyPI

```bash
python -m pip install glacier-flow-tools
```

## From source, in a conda environment

There are two conda environments. The runtime `environment.yml` has what is
needed to run and test the package. `environment-dev.yml` adds the packages
for building this documentation.

```bash
git clone https://github.com/pism/glacier-flow-tools.git
cd glacier-flow-tools
conda env create -f environment.yml
conda activate glacier-flow-tools
python -m pip install -e .
```

Or using [Mamba](https://mamba.readthedocs.io/) instead, which resolves the
environment considerably faster:

```bash
git clone https://github.com/pism/glacier-flow-tools.git
cd glacier-flow-tools
mamba env create -f environment.yml
mamba activate glacier-flow-tools
python -m pip install -e .
```

For development, including the documentation build, create the environment
from `environment-dev.yml` instead. See {doc}`../developer/documentation`.

## Check the installation

Both command-line tools print their options:

```bash
compute_pathlines --help
compute_profiles --help
```

To run the tests from a source checkout:

```bash
python -m pip install -e ".[test]"
pytest
```
