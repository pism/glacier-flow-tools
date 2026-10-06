# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions workflows for checking that this changelog is updated, checking that PRs to `release` are labeled, tagging versions, creating releases, secrets analysis and pre-commit, and publishing to PyPI when a version is tagged.
- Dependabot updates for GitHub Actions.

### Changed
- The version number is computed from the git tags by `setuptools_scm`.
- The test workflow uses micromamba, runs on Ubuntu and macOS with Python 3.10, 3.12, 3.13 and 3.14, and replaces `python-package.yml`.
- `environment.yml` no longer pins Python to 3.11.7.

## [0.1.1]

## [0.1.0]
