# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.4]

### Fixed
- `compute_profiles` crashed while plotting with `RuntimeError: Entry point name 'inline' duplicated` when there was more than one profile. `plot_glacier` and `plot_obs_sims` chose the matplotlib backend the wrong way round: the non-interactive default selected the notebook inline backend, which is not thread-safe in the dask workers. They now use `agg` unless `interactive=True`, as documented and as the other plotting functions do.

### Added
- Documentation in `docs/`, built with Sphinx and MyST. Its layout, configuration and build setup are adapted from pism-terra. It has pages on installation, a quick start, pathlines and profiles, a gallery example that computes pathlines in a rotating flow, an API reference, a command-line reference and the release notes.
- `environment-dev.yml`, the conda environment for development. It has the packages for building the documentation.
- A `docs` extra in `pyproject.toml` with the same documentation packages. New ones are `sphinx-copybutton`, `sphinx-autobuild` and `ipykernel`.
- An end-to-end test for `compute_profiles` on synthetic observations, simulations and profiles.

### Changed
- The documentation packages moved from `environment.yml` to `environment-dev.yml`. `autovizwidget` and `graphviz` are no longer installed, because the documentation does not use them.
- `.readthedocs.yaml` builds from `docs/source/conf.py` and installs the package with the `docs` extra, instead of the conda environment.
- The old `doc/` skeleton, which pointed at xDEM logos that were never in the repository, and the empty `examples/basic` and `examples/advanced` folders are replaced.
- The help of `compute_profiles` describes profiles. It had the description of `compute_pathlines`.

## [0.2.3]

### Fixed
- The console scripts `compute_pathlines` and `compute_profiles` failed with an `ImportError` on start. Their entry points named functions that did not exist. The scripts' code now lives in the package, as `glacier_flow_tools.compute_pathlines` and `glacier_flow_tools.compute_profiles`, each with a `main()` function.
- `h5py` is a requirement. `h5netcdf` no longer pulls it in, so NetCDF files could not be opened after a plain `pip install`.
- `pathlines/compute_pathlines.py` and `profiles/compute_profiles.py` are thin wrappers around these, so `%run` in the notebooks still works.

### Added
- Tests that the console scripts resolve and print their help.
- `*.egg-info/` is ignored by git.

## [0.2.2]

### Added
- A `.gitignore` that ignores `__pycache__/`.

### Changed
- The test workflow installs the package in editable mode, so coverage is measured on the repository's source files. With the installed copy, Codecov could not match the paths and reported 0% coverage.
- The Codecov project check compares coverage against the base branch (`target: auto`, 1% threshold) instead of requiring 100%, which failed on every pull request.

## [0.2.1]

### Fixed
- The classifier `Topic :: Scientific/Engineering :: Postprocessing` is not valid on PyPI and made the upload of 0.2.0 fail. It is replaced by `Topic :: Scientific/Engineering :: GIS`. Version 0.2.0 was never published to PyPI.
- The PyPI workflow builds with `pyproject-build`, because `python -m build` imported the repository's `build.py` and failed. When run by hand it now asks for the tag to publish.

## [0.2.0]

### Added
- GitHub Actions workflows for checking that this changelog is updated, checking that PRs to `release` are labeled, tagging versions, creating releases, secrets analysis and pre-commit, and publishing to PyPI when a version is tagged.
- Dependabot updates for GitHub Actions.

### Changed
- The version number is computed from the git tags by `setuptools_scm`.
- The test workflow uses micromamba, runs on Ubuntu and macOS with Python 3.10, 3.12, 3.13 and 3.14, and replaces `python-package.yml`.
- `environment.yml` no longer pins Python to 3.11.7.

### Fixed
- `profiles.normal` no longer uses `np.cross` on 2D vectors, which NumPy 2.3 removed. `test_normal` and `test_extract_profiles` failed with it on Python 3.12 and later.
- A stray `b` line before the module docstring in `profiles/compute_profiles.py` raised a `NameError` on import.
- Trailing whitespace in `README.md`, import order in the pathline and plotting scripts, and missing numpydoc docstrings, so that pre-commit passes.

## [0.1.1]

## [0.1.0]
