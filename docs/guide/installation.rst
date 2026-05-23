.. _installation:

Installation
============

.. contents:: On this page
   :local:
   :depth: 1

Requirements
------------

- Python 3.10 or later
- NumPy ≥ 1.24
- SciPy ≥ 1.10
- tifffile ≥ 2023.1.1 (for reading TIFF files in examples)

From PyPI (stable release)
---------------------------

.. code-block:: bash

   pip install US_TIE_Zhang_et_al_2020_py

From source (development install with uv)
------------------------------------------

This project is managed with `uv <https://docs.astral.sh/uv/>`_, a fast Python package manager.
Install uv first if you do not have it:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Then clone the repository and set up the environment in one step:

.. code-block:: bash

   git clone https://github.com/wayscience/US_TIE_Zhang_et_al_2020_py
   cd US_TIE_Zhang_et_al_2020_py
   uv sync --group dev

``uv sync`` creates a virtual environment in ``.venv/`` and installs the package in editable mode together with all development dependencies.
You never need to activate the environment manually — prefix commands with ``uv run`` and uv handles it.

Installing documentation dependencies
----------------------------------------

To build the documentation locally, add the ``docs`` optional dependencies:

.. code-block:: bash

   uv sync --group dev
   uv pip install -e ".[docs]"
   uv run sphinx-build docs docs/_build/html
   open docs/_build/html/index.html

For a live-reloading preview that rebuilds on every save:

.. code-block:: bash

   uv run sphinx-autobuild docs docs/_build/html --open-browser

Or use the Makefile shorthand from inside ``docs/``:

.. code-block:: bash

   cd docs
   uv run make html
   open _build/html/index.html

Verifying the installation
---------------------------

.. code-block:: python

   import US_TIE_Zhang_et_al_2020_py
   print(US_TIE_Zhang_et_al_2020_py.__version__)   # should print '0.1.0' or later

To run the full test suite:

.. code-block:: bash

   uv run pytest

All 82 tests should pass.
