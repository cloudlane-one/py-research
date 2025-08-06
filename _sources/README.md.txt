# py-research

[![PyPI - Version](https://img.shields.io/pypi/v/py-research)](https://pypi.org/project/py-research)
[![Docs](https://github.com/cloudlane-one/py-research/actions/workflows/docs.yml/badge.svg)](https://cloudlane-one.github.io/py-research)
[![Tests](https://github.com/cloudlane-one/py-research/actions/workflows/lint-and-test.yml/badge.svg)](https://github.com/cloudlane-one/py-research/actions/workflows/lint-and-test.yml)
[![Build](https://github.com/cloudlane-one/py-research/actions/workflows/release.yml/badge.svg)](https://github.com/cloudlane-one/py-research/actions/workflows/release.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://codecov.io/gh/cloudlane-one/py-research/graph/badge.svg?token=J8GLAWTPWX)](https://codecov.io/gh/cloudlane-one/py-research)
[![CodeFactor](https://www.codefactor.io/repository/github/cloudlane-one/py-research/badge)](https://www.codefactor.io/repository/github/cloudlane-one/py-research)
[![CodeClimate Maintainability](https://api.codeclimate.com/v1/badges/4cc310489b325e793d1d/maintainability)](https://codeclimate.com/github/cloudlane-one/py-research/maintainability)

## About this project

This repository is a collection of Python utilities to help you analyze & visualize data, automate workflows and manage knowledge while working in an interdisciplinary R&D project. More precisely, it serves these main puposes:

1. Extend existing, well-established packages (e.g. pandas, numpy sqlalchemy, structlog, ...) with small helper functions and abstractions to provide lacking functionality or make them easier to use & automate.
2. Enforce best practices on scientific code to make it more reliable, quick to deploy, and easy to monitor.
3. Provide a framework for knowledge and information management backed by different data sources and databases, to which analysis / dataviz functions can tie in seamlessly.

## Current status

This project is actively maintained, but still under construction. The modules in their current form are working and continuously tested, but documentation is still minimal and many components of the framework are not yet implemented. Release versioning follows the [Semver Spec](https://semver.org/), so breaking changes in existing API will only happen with major version upgrades, of which there are expected to be a few before the framework is complete.

Furthermore, some of the current functionality may be outsourced into a separate repo and package with these upgrades.

## How to install

`py-research` is available as a Python package on PyPI:

```bash
pip install py-research
```

Alternatively, you can install it directly from git via:

```bash
pip install git+https://github.com/cloudlane-one/py-research.git
```

## Contents

```{toctree}
docs/ref-api/index
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
