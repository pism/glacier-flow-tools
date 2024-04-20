[![License: GPL-3.0](https://img.shields.io:/github/license/pism/pypac)](https://opensource.org/licenses/GPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

# Glacier Flow Tools

Glacier Flow Tools facilitates analysis of glacier flow. It provides modules to compute pathlines (trajectories), compute flow across profiles (flux gates), etc. This tools is currently under development.

## Installation

Get glacier-flow-tools source from GitHub:

    $ git clone git@github.com:pism/glacier-flow-tools.git
    $ cd glacier-flow-tools

Optionally create Conda environment named *glacier-flow-tools*:

    $ conda env create -f environment.yml
    $ conda activate glacier-flow-tools

Install glacier-flow-tools:

    $ pip install -e .


## Examples

![Pathlines starting from the Jakobshaven Isbræ flux gate.](https://github.com/pism/glacier-flow-tools/blob/main/images/jak_obs_speed.png)
