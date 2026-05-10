"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "simpletorchtrainer"
copyright = "2026, Balazs Paszkal Halmos"
author = "Balazs Paszkal Halmos"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = [
    "sphinx.ext.autodoc",  # pulls docstrings from code
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # supports NumPy/Google style
    "numpydoc",  # better NumPy support
    "sphinx.ext.viewcode",  # adds source code links
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = []

# Enable Numpy style docstring
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Autosummary
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
