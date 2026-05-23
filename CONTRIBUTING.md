# Contributing to US_TIE_Zhang_et_al_2020_py

Thank you for your interest in contributing!
This guide covers everything you need to get a working development environment, run the tests, and submit changes.

---

## Prerequisites

This project is managed with [uv](https://docs.astral.sh/uv/), a fast Python package and project manager.
Install it once with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or via pip if you prefer:

```bash
pip install uv
```

See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for other options (Homebrew, Windows, etc.).

---

## Setting up the development environment

Clone the repository and let uv create the environment and install all dependencies in one step:

```bash
git clone https://github.com/wayscience/US_TIE_Zhang_et_al_2020_py
cd US_TIE_Zhang_et_al_2020_py
uv sync --group dev
```

`uv sync` reads `pyproject.toml`, creates a virtual environment in `.venv/`, and installs the project (editable) plus the `dev` dependency group.
You never need to activate the environment manually — prefix commands with `uv run` and uv handles it.

---

## Running the tests

```bash
uv run pytest
```

To run a specific test file:

```bash
uv run pytest tests/test_dct.py
```

To run with coverage:

```bash
uv run pytest --cov=US_TIE_Zhang_et_al_2020_py --cov-report=term-missing
```

All 68 tests should pass before you open a pull request.

---

## Building the documentation

Install the docs dependency group alongside dev:

```bash
uv sync --group dev --group docs
```

Then build the HTML docs:

```bash
uv run --group docs sphinx-build docs docs/_build/html
```

Or use the Makefile shorthand from inside `docs/`:

```bash
cd docs
uv run --group docs make html
open _build/html/index.html
```

---

## Adding dependencies

Add a runtime dependency (goes into `[project.dependencies]`):

```bash
uv add some-package
```

Add a development-only dependency (goes into `[dependency-groups] dev`):

```bash
uv add --group dev some-package
```

Add a docs dependency (goes into `[dependency-groups] docs`):

```bash
uv add --group docs some-package
```

After adding, commit both `pyproject.toml` and the updated `uv.lock`.

---

## Project structure

```
US_TIE_Zhang_et_al_2020_py/
├── US_TIE_Zhang_et_al_2020_py/
│   ├── __init__.py       # public API and package docstring
│   ├── pipeline.py       # retrieve_phase, compute_dIdz (start here)
│   ├── core.py           # TIESolver, Poisson solvers, forward models
│   ├── solvers.py        # functional wrappers (universal_solution, fft_tie_solution)
│   ├── propagation.py    # numerical_propagation (Angular Spectrum / Fresnel / TIE)
│   └── utils.py          # remove_piston, rmse
├── tests/
│   ├── test_core.py      # Poisson solvers and TIE operators
│   ├── test_dct.py       # DCT backend and Neumann BC solver
│   ├── test_pipeline.py  # retrieve_phase and compute_dIdz
│   ├── test_propagation.py
│   └── test_solvers.py
├── docs/                 # Sphinx source
├── pyproject.toml
├── uv.lock
├── .python-version       # pinned Python version for uv
├── LICENSE
└── CONTRIBUTING.md       # this file
```

---

## Code style

- Follow the existing code style (PEP 8, type hints, NumPy-style docstrings).
- Every public function and class must have a docstring.
- Docstrings should explain *what* and *why* in plain language before presenting any mathematics.
- One sentence per line in Markdown files.
- Keep the SPDX license header at the top of every `.py` file.

---

## Attribution

This project is adapted from MATLAB code by Zheng et al. (2020) under CC BY 4.0.
See `LICENSE` for full attribution details.
Any contribution you make will be released under the same CC BY 4.0 license.
If you use AI assistance (e.g. Claude, Copilot) to write code or documentation, please say so in your pull request description.

---

## Submitting a pull request

1. Fork the repository and create a branch from `main`.
2. Make your changes and add tests if you are adding new functionality.
3. Run `uv run pytest` and confirm all tests pass.
4. Open a pull request with a clear description of what you changed and why.
