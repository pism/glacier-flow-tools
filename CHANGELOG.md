# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- A `.gitignore` that ignores `__pycache__/`.

### Changed
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
