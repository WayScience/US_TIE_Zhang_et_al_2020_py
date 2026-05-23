# SPDX-License-Identifier: CC-BY-4.0
# Adapted from Zheng et al. (2020) MATLAB source (CC BY 4.0) by Way Science Lab (2026),
# with AI assistance from Claude (Anthropic). See LICENSE for full attribution.

"""
TIE phase retrieval algorithms — functional (stateless) API.

This module provides two distinct TIE algorithms as plain functions.
Each builds a :class:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver` internally
on every call.

For processing many images of the same size, construct a :class:`TIESolver`
once and call :meth:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver.solve` repeatedly — this avoids
recomputing frequency grids and Poisson filter kernels on every call.

Available functions
-------------------
:func:`universal_solution`
    Iterative US-TIE solver (Zheng et al. 2020).  The recommended method for
    most applications.  Works for non-uniform illumination and handles cases
    where intensity is near zero.

:func:`fft_tie_solution`
    Classical two-step FFT-TIE (Teague's formulation).  Non-iterative and
    slightly faster, but less robust for specimens with very low intensity in
    some regions.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.fft as _sfft

from .solver import TIESolver
from .poisson import _freq_grids
from .utils import rmse


def universal_solution(
    dIdz: np.ndarray,
    I0: np.ndarray,
    pixelsize: float,
    k: float,
    reg: float = np.finfo(float).eps,
    max_iter: int = 500,
    tol: float = 1e-3,
    true_phase: np.ndarray | None = None,
    fft_workers: int = -1,
    backend: str = "fft",
) -> dict:
    """Iterative Universal Solution to the TIE (US-TIE).

    Recovers a quantitative phase image from the axial intensity derivative
    and in-focus intensity using the algorithm of Zheng et al. (2020).

    .. tip::

        If you are processing many images of the **same size**, construct a
        :class:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver` once and call
        :meth:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver.solve` in a loop — this avoids
        re-allocating frequency grids on every call and can be 2–5× faster.

    Parameters
    ----------
    dIdz : array_like, shape (Ny, Nx)
        Axial intensity derivative ``∂I/∂z`` in intensity units per metre.
        Compute from images using :func:`US_TIE_Zhang_et_al_2020_py.compute_dIdz`.
    I0 : array_like, shape (Ny, Nx)
        In-focus intensity image.
    pixelsize : float
        Camera pixel size mapped to the sample plane, in metres.
    k : float
        Optical wavenumber ``2π / λ`` in rad/m.
    reg : float, optional
        Tikhonov regularisation parameter (FFT backend only).  Default
        (≈ machine epsilon) gives essentially no regularisation while
        preventing DC division by zero.  Increase to ``1e-3`` to suppress
        low-frequency noise when the background is non-uniform.
    max_iter : int, optional
        Maximum number of US-TIE iterations.  The algorithm typically
        converges in 10–50 iterations.
    tol : float, optional
        Convergence threshold: stop when ``max|residual| < tol × max|dIdz|``.
    true_phase : array_like, shape (Ny, Nx), optional
        Ground-truth phase for RMSE tracking during iteration.  Intended for
        validation and research; not needed for routine use.
    fft_workers : int, optional
        Number of CPU threads for FFT computation.  ``-1`` (default) uses all
        available cores.
    backend : {'fft', 'dct'}, optional
        Poisson solver backend.

        ``'fft'`` *(default)*
            Spectral inverse-Laplacian, periodic boundary conditions.  Fast
            and accurate when the specimen is away from the image border.

        ``'dct'``
            DCT-II inverse-Laplacian, Neumann boundary conditions (∂φ/∂n = 0).
            Eliminates boundary ringing for specimens near the image edge.

    Returns
    -------
    result : dict
        Dictionary with keys:

        ``'phase'`` : ndarray, shape (Ny, Nx)
            Recovered phase in radians.
        ``'rmse'`` : list of float
            Per-iteration RMSE values (empty unless ``true_phase`` is given).
        ``'times'`` : list of float
            Wall-clock time per iteration in seconds (empty unless
            ``true_phase`` is given).
        ``'iterations'`` : int
            Number of iterations performed.

    Examples
    --------
    >>> import numpy as np
    >>> from US_TIE_Zhang_et_al_2020_py import universal_solution, compute_dIdz
    >>>
    >>> dIdz, I0 = compute_dIdz([I_under, I_focus, I_over], dz=1e-6)
    >>> result = universal_solution(
    ...     dIdz, I0,
    ...     pixelsize=162.5e-9,
    ...     k=2 * np.pi / 532e-9,
    ... )
    >>> phase = result['phase']

    See Also
    --------
    retrieve_phase : End-to-end wrapper (images → phase) for new users.
    TIESolver : Reusable solver class for batch processing.
    fft_tie_solution : Classical non-iterative FFT-TIE alternative.
    """
    solver = TIESolver(
        shape=I0.shape,
        pixelsize=pixelsize,
        k=k,
        reg=reg,
        backend=backend,
        fft_workers=fft_workers,
    )
    return solver.solve(dIdz, I0, max_iter=max_iter, tol=tol, true_phase=true_phase)


def fft_tie_solution(
    dIdz: np.ndarray,
    I0: np.ndarray,
    pixelsize: float,
    k: float,
    reg: float = np.finfo(float).eps,
    int_threshold: float = 0.01,
    true_phase: np.ndarray | None = None,
    fft_workers: int = -1,
) -> dict:
    """Classical FFT-TIE solver (Teague's two-step formulation).

    Solves the TIE in two sequential Poisson solves using Teague's
    simplifying assumption that ``I₀`` can be treated as approximately
    uniform.  This is **non-iterative** (one pass) and slightly faster than
    :func:`universal_solution`, but less robust:

    * Assumes the intensity ``I₀`` is slowly varying across the image.
    * Pixels where ``I₀`` is near zero are clamped to a threshold to prevent
      division by zero; this can introduce artefacts in dark regions.
    * May show ringing near specimen boundaries due to the periodic (FFT)
      boundary condition assumption.

    Use :func:`universal_solution` for specimens with non-uniform illumination
    or very low intensity in some regions.

    Parameters
    ----------
    dIdz : array_like, shape (Ny, Nx)
        Axial intensity derivative ``∂I/∂z``.
    I0 : array_like, shape (Ny, Nx)
        In-focus intensity.
    pixelsize : float
        Pixel size in metres.
    k : float
        Optical wavenumber ``2π / λ`` in rad/m.
    reg : float, optional
        Tikhonov regularisation parameter.  Default ≈ machine epsilon.
    int_threshold : float, optional
        Pixels with ``I0 < int_threshold × max(I0)`` are clamped to
        ``int_threshold × max(I0)`` to prevent division by zero.
        Default is 0.01 (1 % of maximum intensity).
    true_phase : array_like, shape (Ny, Nx), optional
        Ground-truth phase for RMSE computation.  If provided, the RMSE
        (after piston removal) is included in the output dictionary.
    fft_workers : int, optional
        Number of CPU threads for FFT computation.  ``-1`` = all cores.

    Returns
    -------
    result : dict
        Dictionary with keys:

        ``'phase'`` : ndarray, shape (Ny, Nx)
            Recovered phase in radians.
        ``'rmse'`` : float or None
            RMSE against ``true_phase`` (``None`` if not provided).
        ``'time'`` : float
            Total wall-clock time in seconds.

    Notes
    -----
    The algorithm (following Paganin & Nugent 1998 / Teague 1983):

    1. Solve ``∇²ψ = −k · dIdz`` for the auxiliary field ``ψ``.
    2. Compute the gradient ``∇ψ`` and divide by ``I₀``: this gives ``∇φ``.
    3. Solve ``∇²φ = ∇·(∇ψ / I₀)`` for the phase.

    Both Poisson solves use the regularised FFT formula.

    Examples
    --------
    >>> from US_TIE_Zhang_et_al_2020_py import fft_tie_solution
    >>> result = fft_tie_solution(dIdz, I0, pixelsize=162.5e-9,
    ...                           k=2*np.pi/532e-9)
    >>> phase = result['phase']

    See Also
    --------
    universal_solution : Iterative solver, more robust for non-uniform I₀.
    retrieve_phase : High-level entry point for routine use.
    """
    t0 = time.perf_counter()
    ny, nx = dIdz.shape
    J = -k * np.asarray(dIdz, dtype=float)

    U, V = _freq_grids(ny, nx, pixelsize)
    Cx = 2j * np.pi * U
    Cy = 2j * np.pi * V
    lap = Cx ** 2 + Cy ** 2
    poisson_k = lap / (reg / pixelsize ** 4 + lap ** 2)

    w = int(fft_workers)

    def fft2(x):   return _sfft.fft2(x, workers=w)
    def rifft2(x): return np.real(_sfft.ifft2(x, workers=w))

    # Step 1: solve ∇²ψ = J  →  get ∇ψ
    Fpsi = fft2(J) * poisson_k
    dpsidx = rifft2(Fpsi * Cx)
    dpsidy = rifft2(Fpsi * Cy)

    # Step 2: compute ∇φ = ∇ψ / I₀  (with intensity thresholding)
    I0f = np.asarray(I0, dtype=float).copy()
    threshold = int_threshold * float(np.nanmax(I0f))
    I0f[I0f < threshold] = threshold

    # Step 3: solve ∇²φ = ∇·(∇ψ / I₀)
    laplace_phi = (rifft2(fft2(dpsidx / I0f) * Cx)
                   + rifft2(fft2(dpsidy / I0f) * Cy))
    phi = rifft2(fft2(laplace_phi) * poisson_k)

    elapsed = time.perf_counter() - t0
    result: dict = {"phase": phi, "time": elapsed, "rmse": None}
    if true_phase is not None:
        result["rmse"] = rmse(phi, true_phase)
    return result