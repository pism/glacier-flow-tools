"""
Sphinx configuration for glacier-flow-tools.

Adapted from pism-terra. Layout mirrors xDEM (https://xdem.readthedocs.io/):
sphinx-book-theme, sphinx-design, sphinx-gallery for runnable ``plot_*.py``
examples, and myst-nb for Markdown and notebook pages.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

# -- Project information -----------------------------------------------------

project = "glacier-flow-tools"
author = "Andy Aschwanden, Constantine Khroulev"
copyright = f"2024-2026, {author}"  # pylint: disable=redefined-builtin

try:
    release = metadata.version("glacier-flow-tools")
except metadata.PackageNotFoundError:
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.programoutput",
    "myst_nb",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # Sphinx-gallery generates .rst, .py, .ipynb, .codeobj.json, .py.md5 and
    # .zip side-by-side under ``auto_examples/``. Every one of them looks like
    # the same document to Sphinx ("multiple files found for the document").
    # Let the .rst be the canonical source.
    "auto_examples/*.ipynb",
    "auto_examples/*.py",
    "auto_examples/*.py.md5",
    "auto_examples/*.codeobj.json",
    "auto_examples/*.zip",
    "auto_examples/**/*.ipynb",
    "auto_examples/**/*.py",
    "auto_examples/**/*.py.md5",
    "auto_examples/**/*.codeobj.json",
    "auto_examples/**/*.zip",
]

# myst-nb registers parsers for both .md and .ipynb on its own; .rst stays
# the default Sphinx parser. No explicit ``source_suffix`` mapping needed.

# -- Theme -------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "glacier-flow-tools"

html_theme_options = {
    "repository_url": "https://github.com/pism/glacier-flow-tools",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_edit_page_button": False,
    # Always show the full nav tree in the left sidebar, expanded.
    "navigation_depth": 4,
    "show_navbar_depth": 2,
    "collapse_navbar": False,
    "show_toc_level": 2,
    "home_page_in_toc": True,
    "path_to_docs": "docs/source",
}

# -- autodoc / autosummary / numpydoc ---------------------------------------

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# numpydoc validation lives in pyproject.toml; only suppress the noisy
# class-attribute table here.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -- MyST / MyST-NB ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Don't execute notebooks at build time.
nb_execution_mode = "off"

# -- sphinx-gallery ---------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": [str(Path(__file__).parent / "../../examples")],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r"plot_",
    "remove_config_comments": True,
    "show_signature": False,
    "download_all_examples": False,
}

# -- intersphinx ------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
}

# -- copybutton -------------------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Warning suppression ----------------------------------------------------

suppress_warnings = [
    # autosummary :toctree: warns until generated/ exists at first build.
    "autosummary",
    # myst-nb tolerated warnings on unconfigured cell-metadata keys.
    "mystnb.unknown_mime_type",
]
