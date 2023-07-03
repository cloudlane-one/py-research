# pylint: disable=C0114  # noqa: D100
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Py-Research"
copyright = "2023, Lorenzo Wormer"  # pylint: disable=W0622
author = "Lorenzo Wormer"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["docs/_templates"]

default_role = "any"

autodoc_default_options = {
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_default_flags = ["members", "inherited-members", "show-inheritance"]
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True  # If no docstring, inherit from base class

add_module_names = False  # Remove namespaces from class/method signatures

intersphinx_mapping = {
    "python": (
        f"https://docs.python.org/{sys.version_info.major}",
        None,
    ),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "README"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    ".*",
    "**/local",
    "**/.cache",
    "Thumbs.db",
    ".DS_Store",
    "docs-html",
    "docs-pdf",
    "poetry.lock",
]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

nbsphinx_execute = "never"
