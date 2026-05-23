# SPDX-License-Identifier: CC-BY-4.0
# Adapted from Zheng et al. (2020) MATLAB source (CC BY 4.0) by Way Science Lab (2026),
# with AI assistance from Claude (Anthropic). See LICENSE for full attribution.

"""
TIE forward model and single-step operator.

The **Transport-of-Intensity Equation (TIE)** relates the axial rate of change
of image intensity to the phase of the light:

.. math::

    k\\,\\frac{\\partial I}{\\partial z} =
    -\\nabla \\cdot \\bigl[ I_0(\\mathbf{r})\\, \\nabla\\phi(\\mathbf{r}) \\bigr]

:func:`tie_forward` evaluates the right-hand side — given a phase and an
intensity image, it predicts what the through-focus intensity change should
look like.
This is the "forward direction": phase → observable.

:func:`tie_max_solver` does one step in the *inverse* direction — it takes an
observed intensity change and estimates the phase that produced it, using the
maximum-intensity Poisson approximation from Zheng et al. (2020).

Both functions are primarily used internally by
:class:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver`.
They are exposed here for testing, validation, and custom pipelines.
"""
from __future__ import annotations

import numpy as np

from .poisson import _freq_grids, poisson_fft


def tie_forward(
    phi: np.ndarray,
    I0: np.ndarray,
    pixelsize: float,
    k: float,
) -> np.ndarray:
    """Compute the axial intensity derivative predicted by the TIE.

    The **TIE forward model** takes a known phase field ``φ`` and in-focus
    intensity ``I0`` and predicts how the intensity would change if you
    slightly defocused the microscope.
    It evaluates:

    .. math::

        \\frac{\\partial I}{\\partial z} = -\\frac{1}{k}\\,
        \\nabla \\cdot \\bigl[ I_0(\\mathbf{r})\\, \\nabla \\phi(\\mathbf{r}) \\bigr]

    This function is primarily used **inside** the iterative TIE solver to
    check how well a candidate phase explains the observed intensity change.
    It can also be used to generate synthetic through-focus images for testing.

    Parameters
    ----------
    phi : array_like, shape (Ny, Nx)
        Phase image in radians.
    I0 : array_like, shape (Ny, Nx)
        In-focus intensity image (must be non-negative).
    pixelsize : float
        Pixel size in metres.
    k : float
        Optical wavenumber ``2π / λ`` in rad/m.

    Returns
    -------
    dIdz : ndarray, shape (Ny, Nx)
        Predicted axial intensity derivative ``∂I/∂z``.

    Examples
    --------
    >>> import numpy as np
    >>> from US_TIE_Zhang_et_al_2020_py.tie import tie_forward
    >>> N = 64
    >>> phi = np.zeros((N, N))       # flat phase → no intensity change
    >>> I0  = np.ones((N, N))
    >>> dIdz = tie_forward(phi, I0, pixelsize=2e-6, k=2*np.pi/0.633e-6)
    >>> np.allclose(dIdz, 0.0)
    True

    See Also
    --------
    TIESolver.forward : Same computation with precomputed FFT grids (faster
                        for repeated calls on the same image size).
    tie_max_solver : One inverse step — intensity change → phase estimate.
    """
    ny, nx = phi.shape
    U, V = _freq_grids(ny, nx, pixelsize)
    Cx = 2j * np.pi * U
    Cy = 2j * np.pi * V
    Fphi = np.fft.fft2(phi)
    dphidx = np.real(np.fft.ifft2(Fphi * Cx))
    dphidy = np.real(np.fft.ifft2(Fphi * Cy))
    laplace_psi = (
        np.real(np.fft.ifft2(np.fft.fft2(I0 * dphidx) * Cx))
        + np.real(np.fft.ifft2(np.fft.fft2(I0 * dphidy) * Cy))
    )
    return laplace_psi / (-k)


def tie_max_solver(
    phi_curr: np.ndarray,
    dIdz_curr: np.ndarray,
    I0: np.ndarray,
    pixelsize: float,
    k: float,
    reg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One step of the US-TIE iteration using the FFT Poisson solver.

    This is a convenience function for single-step use.
    For repeated calls on images of the same size, use
    :class:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver`, which precomputes
    the frequency grids and is noticeably faster.

    The step replaces the spatially-varying ``I0(r)`` with the scalar maximum
    ``I_max = max(I0)``, which turns the variable-coefficient PDE into a
    standard Poisson equation solvable by FFT.
    This approximation is exact when ``I0`` is uniform and converges
    iteratively for non-uniform ``I0``.

    Parameters
    ----------
    phi_curr : array_like, shape (Ny, Nx)
        Current phase estimate.
        Unused in the FFT step itself but kept for API symmetry with other
        solver implementations.
    dIdz_curr : array_like, shape (Ny, Nx)
        Current residual axial intensity derivative.
    I0 : array_like, shape (Ny, Nx)
        In-focus intensity.
    pixelsize : float
        Pixel size in metres.
    k : float
        Optical wavenumber ``2π / λ`` in rad/m.
    reg : float
        Tikhonov regularisation parameter.

    Returns
    -------
    phi_est : ndarray, shape (Ny, Nx)
        Phase correction estimate for this iteration.
    dIdz_est : ndarray, shape (Ny, Nx)
        Predicted intensity derivative from ``phi_est`` and ``I0``.

    See Also
    --------
    TIESolver.step : Equivalent method with precomputed kernels.
    tie_forward : The forward model used to compute ``dIdz_est``.
    """
    I_max = float(np.nanmax(I0))
    ny, nx = dIdz_curr.shape
    U, V = _freq_grids(ny, nx, pixelsize)
    lap = (2j * np.pi * U) ** 2 + (2j * np.pi * V) ** 2
    Fphi = np.fft.fft2(-k * dIdz_curr / I_max) * lap / (reg / pixelsize**4 + lap**2)
    phi_est = np.real(np.fft.ifft2(Fphi))
    dIdz_est = tie_forward(phi_est, I0, pixelsize, k)
    return phi_est, dIdz_est
