# Building the documentation

This documentation is Sphinx + MyST. It builds from `docs/source/`, and you
can build it locally before pushing.

## Environment

The docs build needs the **development** environment. The runtime
`environment.yml` leaves out everything that is not needed to run and test the
package:

```bash
conda env create -f environment-dev.yml
conda activate glacier-flow-tools
```

Or with [Mamba](https://mamba.readthedocs.io/), which resolves considerably
faster:

```bash
mamba env create -f environment-dev.yml
mamba activate glacier-flow-tools
```

Then install the package together with the `docs` extras, which pull in Sphinx,
the theme and the MyST/autosummary machinery:

```bash
python -m pip install -e ".[docs]"
```

## Build

```bash
cd docs
make html
open _build/html/index.html
```

While editing, `make livehtml` rebuilds on save and serves the result at
<http://127.0.0.1:8000>:

```bash
cd docs
make livehtml
```

`make clean` removes the build, the generated gallery and the generated API
pages.

## Layout

`docs/source/conf.py`
: The Sphinx configuration.

`docs/source/getting_started`, `features`, `developer`, `reference`
: The pages, in Markdown ([MyST](https://myst-parser.readthedocs.io/)).

`examples/`
: The gallery. Every `plot_*.py` file there is run during the build by
  [sphinx-gallery](https://sphinx-gallery.github.io/), and its figures and
  output become a page under `auto_examples/`. Examples should use synthetic
  data, so that the build needs no downloads.

`reference/api.md`
: The API reference. Add new public functions to the matching `autosummary`
  table.

## Docstrings

The API reference is generated from the docstrings, which follow the
[numpydoc](https://numpydoc.readthedocs.io/) style. The `numpydoc-validation`
pre-commit hook checks them.

## Read the Docs

`.readthedocs.yaml` builds the documentation by installing the package with
the `docs` extras, the same way as above.
