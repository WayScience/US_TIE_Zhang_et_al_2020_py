# Configuration file for the Sphinx documentation builder.
#
# For a full list of built-in configuration options see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Allow autodoc to find the package when building locally without installing.
# On Read the Docs the package is installed via pip, so this path is a no-op.
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project   = "US_TIE_Zhang_et_al_2020_py"
copyright = "2026, Way Science Lab"
author    = "Way Science Lab"
release   = "0.1.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",       # pull docstrings from source
    "sphinx.ext.napoleon",      # parse NumPy/Google-style docstrings
    "sphinx.ext.viewcode",      # link to highlighted source
    "sphinx.ext.intersphinx",   # cross-link to NumPy, SciPy, etc.
    "sphinx.ext.mathjax",       # render LaTeX math in HTML
    "sphinx.ext.autosummary",   # generate summary tables automatically
    "sphinx_design",            # grid cards / tabs in guide pages
    "sphinx_copybutton",        # copy button on code blocks
    "myst_nb",                  # parse .md (README landing page) + render .ipynb notebooks
]

# ---------------------------------------------------------------------------
# MyST / myst-nb settings
# ---------------------------------------------------------------------------

# MyST extensions used in the README and guide pages
myst_enable_extensions = [
    "colon_fence",   # ::: fences as an alternative to ``` for directives
    "deflist",       # definition lists
    "dollarmath",    # $...$ and $$...$$ for inline and display math
]

# Do NOT re-execute notebooks at build time; use the outputs already stored
# in the .ipynb file (including the embedded phase image in the tutorial).
nb_execution_mode = "off"

# Suppress warnings for repo files that are valid GitHub links but not
# resolvable as Sphinx cross-references (CONTRIBUTING.md, LICENSE, notebook).
suppress_warnings = ["myst.xref_missing"]

# ---------------------------------------------------------------------------
# Napoleon settings — we use NumPy-style docstrings
# ---------------------------------------------------------------------------

napoleon_google_docstring      = False
napoleon_numpy_docstring       = True
napoleon_include_init_with_doc = True
napoleon_use_param             = True
napoleon_use_rtype             = True

# ---------------------------------------------------------------------------
# autodoc / autosummary
# ---------------------------------------------------------------------------

autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "private-members":  False,
    "show-inheritance": True,
    "member-order":     "bysource",
}
autodoc_typehints    = "description"
autosummary_generate = True

# ---------------------------------------------------------------------------
# Cross-links to other packages
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
}

# ---------------------------------------------------------------------------
# Source / exclude patterns
# ---------------------------------------------------------------------------

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# HTML output — Furo theme
# ---------------------------------------------------------------------------

html_theme = "furo"

html_theme_options = {
    "source_repository": "https://github.com/wayscience/US_TIE_Zhang_et_al_2020_py",
    "source_branch":     "main",
    "source_directory":  "docs/",
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary":   "#0369a1",   # blue-700
        "color-brand-content":   "#0369a1",
    },
    "dark_css_variables": {
        "color-brand-primary":   "#38bdf8",   # sky-400
        "color-brand-content":   "#38bdf8",
    },
}

html_title       = "US_TIE_Zhang_et_al_2020_py"
html_logo        = "_static/logo.png"
html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# LaTeX / PDF output
# ---------------------------------------------------------------------------

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble":  r"\usepackage{amsmath,amssymb}",
}
