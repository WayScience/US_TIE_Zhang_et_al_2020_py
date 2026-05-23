# SPDX-License-Identifier: CC-BY-4.0
# Adapted from Zheng et al. (2020) MATLAB source (CC BY 4.0) by Way Science Lab (2026),
# with AI assistance from Claude (Anthropic). See LICENSE for full attribution.

"""
Standalone Poisson solvers.

These functions solve the 2-D Poisson equation ∇²φ = f, which is the core
mathematical step in TIE phase retrieval.
Two solvers are provided, differing only in their boundary condition assumption:

``poisson_fft``
    Periodic boundary conditions (image tiles like wallpaper).
    Fast and accurate when the specimen is away from the image border.

``poisson_dct``
    Neumann boundary conditions (∂φ/∂n = 0 at the image edge).
    Physically correct for images of finite specimens; eliminates boundary
    ringing when objects reach the image edge.

For most users, these functions are called automatically by
:class:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver`.
They are exposed here for users who need direct access — for example, to
apply a Poisson solve to a custom right-hand side, or to compare solver
outputs between the two boundary conditions.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as _sfft


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _freq_grids(ny: int, nx: int, pixelsize: float) -> tuple[np.ndarray, np.ndarray]:
    """Return 2-D spatial-frequency grids in standard FFT order (DC at [0, 0]).

    Parameters
    ----------
    ny, nx : int
        Image dimensions in pixels (rows × columns).
    pixelsize : float
        Pixel size in metres.

    Returns
    -------
    U : ndarray, shape (ny, nx)
        Horizontal (x) spatial frequencies in cycles per metre.
    V : ndarray, shape (ny, nx)
        Vertical (y) spatial frequencies in cycles per metre.
    """
    return np.meshgrid(
        np.fft.fftfreq(nx, d=pixelsize),
        np.fft.fftfreq(ny, d=pixelsize),
    )


# ---------------------------------------------------------------------------
# Poisson solvers
# ---------------------------------------------------------------------------

def poisson_fft(rhs: np.ndarray, pixelsize: float, reg: float) -> np.ndarray:
    """Solve ∇²φ = rhs via FFT with Tikhonov regularisation.

    Solves the 2-D Poisson equation in the Fourier domain, assuming
    **periodic** boundary conditions.
    The DC component (mean of ``φ``) is always fixed to zero.

    The spectral solution is:

    .. math::

        \\hat{\\phi} = \\hat{f} \\cdot
        \\frac{L}{\\varepsilon / dx^4 + L^2}, \\quad
        L = (2\\pi i f_x)^2 + (2\\pi i f_y)^2

    where ``ε`` is the Tikhonov regularisation parameter and ``dx`` is the
    pixel size.
    The denominator prevents division by zero at the DC frequency and damps
    the response at very low spatial frequencies when ``ε`` is increased.

    Parameters
    ----------
    rhs : array_like, shape (Ny, Nx)
        Right-hand side of ∇²φ = rhs.
    pixelsize : float
        Pixel size in metres.
    reg : float
        Tikhonov regularisation parameter ``ε``.
        The default in :class:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver`
        is machine epsilon (≈ 2.2e-16), which gives essentially no
        regularisation.
        Increase (e.g. to ``1e-3``) to reduce low-frequency noise in images
        with a non-uniform background.

    Returns
    -------
    phi : ndarray, shape (Ny, Nx)
        Phase solution with zero mean.

    See Also
    --------
    poisson_dct : Neumann-BC Poisson solver using the DCT.
    TIESolver : High-level iterative solver with precomputed kernels.
    """
    ny, nx = rhs.shape
    U, V = _freq_grids(ny, nx, pixelsize)
    lap = (2j * np.pi * U) ** 2 + (2j * np.pi * V) ** 2
    Fphi = np.fft.fft2(rhs) * lap / (reg / pixelsize**4 + lap**2)
    return np.real(np.fft.ifft2(Fphi))


def poisson_dct(
    rhs: np.ndarray,
    pixelsize: float,
    zero_mean: bool = True,
) -> np.ndarray:
    """Solve ∇²u = rhs with Neumann boundary conditions via the DCT.

    **What this does in plain terms**: given a right-hand side image ``rhs``
    (the Laplacian of the unknown phase), this function finds the phase ``u``
    that produced it — but unlike the FFT solver, it enforces that no
    "optical flux" crosses the image border.
    This is physically correct for microscopy images of finite specimens.

    **How it works**: the Discrete Cosine Transform (DCT-II) diagonalises the
    finite-difference Laplacian with Neumann boundary conditions.
    In the DCT domain, the solution is a pointwise division by the Laplacian
    eigenvalues:

    .. math::

        \\hat{u}[m, n] = \\frac{\\hat{f}[m, n]}{\\lambda[m, n]}, \\quad
        \\lambda[m, n] =
        \\frac{2\\cos(\\pi m / N_x) - 2}{dx^2} +
        \\frac{2\\cos(\\pi n / N_y) - 2}{dy^2}

    The eigenvalue at ``(0, 0)`` is zero (the DC / constant mode), so the
    solution is only determined up to an additive constant.
    This function always returns a zero-mean solution.

    Parameters
    ----------
    rhs : array_like, shape (Ny, Nx)
        Right-hand side of ∇²u = rhs.
    pixelsize : float
        Pixel size in metres (assumes square pixels; ``dx = dy = pixelsize``).
    zero_mean : bool, optional
        If ``True`` (default), subtract the mean of ``rhs`` before solving.
        The Neumann Poisson equation has a solution *only* when the integral
        of ``rhs`` over the domain is zero (the compatibility condition).
        Subtracting the mean enforces this.
        Set to ``False`` only when ``rhs`` is already guaranteed to be
        zero-mean.

    Returns
    -------
    u : ndarray, shape (Ny, Nx)
        Phase solution with zero mean.

    Notes
    -----
    The DCT-II eigenvalues used here correspond exactly to the second-order
    central-difference Laplacian with one-sided differences at the boundary::

        d²u/dx²[j=0]  = (u[1]  − u[0])  / dx²
        d²u/dx²[j=-1] = (u[-2] − u[-1]) / dx²

    This means a round-trip (apply FD Laplacian, then ``poisson_dct``) recovers
    the original field to machine precision.
    See ``test_round_trip_random_field`` for a demonstration.

    Examples
    --------
    >>> import numpy as np
    >>> from US_TIE_Zhang_et_al_2020_py import poisson_dct
    >>> rhs = np.random.default_rng(0).standard_normal((64, 64))
    >>> u = poisson_dct(rhs, pixelsize=2e-6)
    >>> abs(u.mean()) < 1e-12    # always zero-mean
    True

    See Also
    --------
    poisson_fft : Periodic-BC Poisson solver using the FFT.
    TIESolver : High-level iterative solver that wraps this function.
    """
    rhs = np.asarray(rhs, dtype=np.float64)
    ny, nx = rhs.shape

    if zero_mean:
        rhs = rhs - rhs.mean()

    rhs_hat = _sfft.dctn(rhs, type=2, norm="ortho")

    lam_x = 2.0 * (np.cos(np.pi * np.arange(nx) / nx) - 1.0) / pixelsize**2
    lam_y = 2.0 * (np.cos(np.pi * np.arange(ny) / ny) - 1.0) / pixelsize**2
    denom = lam_y[:, None] + lam_x[None, :]   # (ny, nx)

    u_hat = np.zeros_like(rhs_hat)
    nz = denom != 0
    u_hat[nz] = rhs_hat[nz] / denom[nz]
    u_hat[0, 0] = 0.0                          # zero-mean uniqueness fix

    return _sfft.idctn(u_hat, type=2, norm="ortho")
