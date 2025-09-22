# `colocationpy`

> [!WARNING]  
> This package is in the early stages of development and should not be installed unless you are one of the developers.

[![PyPI - Version](https://img.shields.io/pypi/v/colocationpy.svg)](https://pypi.org/project/colocationpy)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/colocationpy.svg)](https://pypi.org/project/colocationpy)
[![Lint](https://github.com/ksuchak1990/colocationpy/actions/workflows/ruff.yml/badge.svg)](https://github.com/ksuchak1990/colocationpy/actions/workflows/ruff.yml)
[![Tests](https://github.com/ksuchak1990/colocationpy/actions/workflows/pytest.yml/badge.svg)](https://github.com/ksuchak1990/colocationpy/actions/workflows/pytest.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17176568.svg)](https://doi.org/10.5281/zenodo.17176568)

> A package to identify instances of co-location between mobile individuals in a
> population.

-----

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [License](#license)
- [Funding](#funding)

## Features

The package provides:

* **Core functionality** for detecting when and where co-location occurs.
* **Supporting metrics** for analysing the outcomes of co-location (entropy, clustering, interaction networks, modularity, etc.).
* **Transformations and utilities** to handle spatial and temporal alignment, distances, and proximity checks.

## Installation

```bash
pip install colocationpy
```

## Quickstart

Import the library and detect co-locations from mobility data, then use the built-in tools to explore the results.

Example outline:

* Provide trajectories with individual IDs, positions, and timestamps.
* Use co-location utilities to compute who was in the same place at the same time.
* Analyse the resulting interaction network or entropy measures.

## Modules

* **colocationpy (top-level)**
  Main API: co-location detection and related helpers.

* **colocationpy.metrics**
  Functions to analyse co-location results: entropies, clustering coefficient, interaction networks, mutual information, modularity.

* **colocationpy.transformations**
  Functions to align coordinate systems and map simulation timesteps to real datetimes. Used to make heterogeneous mobility data comparable.

* **colocationpy.utils**
  Distance measures, proximity functions, barrier handling, and other helpers used internally and available for advanced users.

## License

`colocationpy` is distributed under the terms of the
[BSD-3-Clause](https://github.com/ksuchak1990/colocationpy/blob/main/LICENSE) license.

## Funding

This package is being developed as part of work on the [EPSRC Digital Health Hub for Antimicrobial Resistance](https://www.digitalamr.org/) (EP/X031276/1).
