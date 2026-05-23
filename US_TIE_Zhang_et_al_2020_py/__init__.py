# SPDX-License-Identifier: CC-BY-4.0
#
# US_TIE_Zhang_et_al_2020_py — Python adaptation of the MATLAB US-TIE reference code by
# Zheng et al. (2020), https://arxiv.org/abs/1912.07371
# Adapted by Way Science Lab (2026) with AI assistance from Claude (Anthropic).
# Original MATLAB code: CC BY 4.0.  This adaptation: CC BY 4.0.
# See LICENSE for full attribution and license text.

"""
US_TIE_Zhang_et_al_2020_py — Quantitative Phase Imaging via the Transport-of-Intensity Equation.

What this package does
----------------------
When a beam of light passes through a transparent object — a living cell, a
polymer film, a glass fibre — the object does not absorb the light.  Instead,
it changes the **phase** of the light waves: the waves arrive at the camera
slightly delayed or advanced compared to how they would travel through empty
space.  This invisible delay carries rich information about the object's
thickness and composition, but an ordinary camera only records **intensity**
(brightness), not phase.

This package recovers that hidden phase information from a small set of
conventional brightfield microscopy images taken at slightly different focus
positions.  The mathematical link between these out-of-focus intensity changes
and the underlying phase is called the **Transport-of-Intensity Equation (TIE)**:

.. math::

    k \\frac{\\partial I}{\\partial z} = -\\nabla \\cdot [I(\\mathbf{r})\\,
    \\nabla \\phi(\\mathbf{r})]

where *k* = 2π/λ is the optical wavenumber, *I* is the intensity, *φ* is the
phase, and *z* is the axial (focus) direction.

This package implements the **Universal Solution to the TIE (US-TIE)** from:

    Zheng et al., "On a universal solution to the transport-of-intensity
    equation," *Optics Letters* (2020).
    `arXiv:1912.07371 <https://arxiv.org/abs/1912.07371>`_

Quick start
-----------
The simplest way to recover a phase image is :func:`retrieve_phase`.  Pass
your images, defocus step, pixel size, and wavelength — and get a phase image
back in radians::

    from US_TIE_Zhang_et_al_2020_py import retrieve_phase
    phase = retrieve_phase(
        images=[I_under, I_focus, I_over],  # three brightfield images
        dz=1e-6,             # 1 µm between consecutive planes
        pixelsize=162.5e-9,  # 162.5 nm effective pixel size
        wavelength=532e-9,   # 532 nm green illumination
    )

Public API
----------
**End-to-end pipeline** (start here):

- :func:`retrieve_phase` — images → phase (recommended entry point)
- :func:`compute_dIdz` — compute the axial intensity derivative from images

**Lower-level solvers** (for advanced use):

- :class:`TIESolver` — reusable solver class (fast for batch processing)
- :func:`universal_solution` — iterative US-TIE (functional wrapper)
- :func:`fft_tie_solution` — classical two-step FFT-TIE (non-iterative)
- :func:`poisson_dct` — standalone Neumann Poisson solver

**Utilities**:

- :func:`numerical_propagation` — simulate defocused images from a known phase
- :func:`remove_piston` — remove the constant phase offset
- :func:`rmse` — root-mean-square phase error (piston-corrected)
"""
__version__ = "0.1.0"

from .pipeline import compute_dIdz, retrieve_phase
from .algorithms import fft_tie_solution, universal_solution
from .propagation import numerical_propagation
from .solver import TIESolver
from .poisson import poisson_dct
from .utils import remove_piston, rmse

__all__ = [
    # Primary user-facing API
    "retrieve_phase",
    "compute_dIdz",
    # Lower-level solvers
    "universal_solution",
    "fft_tie_solution",
    "TIESolver",
    "poisson_dct",
    # Utilities
    "numerical_propagation",
    "remove_piston",
    "rmse",
]
