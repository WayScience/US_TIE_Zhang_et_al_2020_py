# SPDX-License-Identifier: CC-BY-4.0
# Adapted from Zheng et al. (2020) MATLAB source (CC BY 4.0) by Way Science Lab (2026),
# with AI assistance from Claude (Anthropic). See LICENSE for full attribution.

"""
TIESolver — the iterative reconstruction engine.

This module contains a single public class, :class:`TIESolver`, which is the
heart of the US-TIE algorithm.
It precomputes frequency grids and Poisson filter kernels once at construction
and reuses them on every call, making it efficient for time-lapse or batch
processing.

For a one-off reconstruction, the convenience function
:func:`~US_TIE_Zhang_et_al_2020_py.retrieve_phase` wraps this class and is
the recommended starting point for new users.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.fft as _sfft

from .poisson import _freq_grids
from .utils import rmse


class TIESolver:
    """Precomputed, multi-threaded iterative TIE solver for a fixed image shape.

    **When to use this class**: if you are processing many images of the same
    size (e.g. a time-lapse), construct one :class:`TIESolver` and call
    :meth:`solve` repeatedly.  The frequency grids and Poisson filter are
    computed once at construction and reused on every call, saving significant
    overhead.  For a single image, the convenience function
    :func:`US_TIE_Zhang_et_al_2020_py.retrieve_phase` is more concise.

    **Algorithm**: iterative Universal Solution to the TIE (US-TIE, Zheng
    et al. 2020).  Each iteration:

    1. Solves ``∇²φ_est = −k · dIdz_curr / I_max`` (Poisson step).
    2. Computes ``dIdz_est = ∇·(I_0 ∇φ_est) / (−k)`` (forward model).
    3. Updates the residual: ``dIdz_curr ← dIdz_curr − dIdz_est``.
    4. Accumulates: ``φ ← φ + φ_est``.

    Iterations stop when the residual ``max|dIdz_curr|`` falls below
    ``tol × max|dIdz_initial|``, or when it increases (indicating divergence).

    Parameters
    ----------
    shape : tuple of int (Ny, Nx)
        Image dimensions in pixels.  All subsequent calls to :meth:`solve`
        must use arrays of this exact shape.
    pixelsize : float
        Camera pixel size mapped to the sample plane, in metres.
        Example: a 40× objective with a 6.5 µm camera pixel gives
        ``6.5e-6 / 40 = 162.5e-9`` m.
    k : float
        Optical wavenumber ``2π / λ`` in rad/m.  For λ = 532 nm:
        ``k = 2 * np.pi / 532e-9``.
    reg : float, optional
        Tikhonov regularisation parameter for the **FFT backend only**.
        Default is machine epsilon (≈ 2.2e-16), which prevents DC
        division by zero with negligible effect on the solution.  Increase
        (e.g. to ``1e-3``) to suppress low-frequency noise when the
        background intensity is non-uniform.  Has no effect for
        ``backend='dct'``.
    backend : {'fft', 'dct'}, optional
        Poisson solver backend.

        ``'fft'`` *(default)*
            Spectral inverse-Laplacian.  Assumes **periodic** boundary
            conditions.  Fastest option; one fewer FFT per iteration than
            the DCT backend (Fourier coefficients are reused for the
            gradient computation).

        ``'dct'``
            DCT-II inverse-Laplacian.  Enforces **Neumann** boundary
            conditions (∂φ/∂n = 0).  Eliminates boundary ringing for
            specimens near the image edge.  Uses a finite-difference forward
            model internally to stay self-consistent with the DCT eigenvalues.

    fft_workers : int, optional
        Number of CPU threads used for FFT and DCT operations.  ``-1``
        (default) uses all available logical cores via ``scipy.fft``.  Set
        to ``1`` for reproducible single-threaded benchmarks.

    Raises
    ------
    ValueError
        If ``backend`` is not ``'fft'`` or ``'dct'``.

    Examples
    --------
    **Single image** (use :func:`US_TIE_Zhang_et_al_2020_py.retrieve_phase` instead for simplicity):

    >>> from US_TIE_Zhang_et_al_2020_py import TIESolver
    >>> import numpy as np
    >>> solver = TIESolver(shape=(512, 512), pixelsize=162.5e-9,
    ...                    k=2*np.pi/532e-9)
    >>> result = solver.solve(dIdz, I0)
    >>> phase = result['phase']   # radians

    **Reusing for a time-lapse** (construct once, call many times):

    >>> solver = TIESolver(shape=(512, 512), pixelsize=162.5e-9,
    ...                    k=2*np.pi/532e-9, backend='dct')
    >>> phases = [solver.solve(dIdz_t, I0_t)['phase']
    ...           for dIdz_t, I0_t in zip(dIdz_series, I0_series)]

    See Also
    --------
    retrieve_phase : End-to-end wrapper that handles image loading and
                     ``dIdz`` computation for you.
    poisson_dct : Standalone DCT Poisson solver.
    poisson_fft : Standalone FFT Poisson solver.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        pixelsize: float,
        k: float,
        reg: float = np.finfo(float).eps,
        backend: str = "fft",
        fft_workers: int = -1,
    ) -> None:
        backend = backend.lower()
        if backend not in ("fft", "dct"):
            raise ValueError(f"backend must be 'fft' or 'dct', got '{backend}'")

        self._k = float(k)
        self._workers = int(fft_workers)
        self._backend = backend
        self._pixelsize = float(pixelsize)

        ny, nx = shape
        U, V = _freq_grids(ny, nx, float(pixelsize))
        Cx = 2j * np.pi * U
        Cy = 2j * np.pi * V

        # Frequency-domain gradient operators, shared by both backends for
        # the FFT-based forward model.
        self._Cx: np.ndarray = np.ascontiguousarray(Cx)
        self._Cy: np.ndarray = np.ascontiguousarray(Cy)

        if backend == "fft":
            lap = Cx**2 + Cy**2
            self._poisson_k: np.ndarray = np.ascontiguousarray(
                lap / (reg / float(pixelsize)**4 + lap**2)
            )
        else:   # dct
            lam_x = (2.0 * (np.cos(np.pi * np.arange(nx) / nx) - 1.0)
                     / float(pixelsize)**2)
            lam_y = (2.0 * (np.cos(np.pi * np.arange(ny) / ny) - 1.0)
                     / float(pixelsize)**2)
            denom = lam_y[:, None] + lam_x[None, :]
            self._dct_denom: np.ndarray = denom
            self._dct_nonzero: np.ndarray = denom != 0

    # ------------------------------------------------------------------
    # Internal FFT wrappers
    # ------------------------------------------------------------------

    def _fft2(self, x: np.ndarray) -> np.ndarray:
        return _sfft.fft2(x, workers=self._workers)

    def _ifft2(self, x: np.ndarray) -> np.ndarray:
        return _sfft.ifft2(x, workers=self._workers)

    def _rifft2(self, x: np.ndarray) -> np.ndarray:
        return np.real(self._ifft2(x))

    # ------------------------------------------------------------------
    # Internal DCT Poisson solve
    # ------------------------------------------------------------------

    def _poisson_dct(self, rhs: np.ndarray) -> np.ndarray:
        """DCT-based Neumann Poisson solve (uses precomputed eigenvalue table)."""
        rhs_hat = _sfft.dctn(
            rhs - rhs.mean(), type=2, norm="ortho", workers=self._workers
        )
        u_hat = np.zeros_like(rhs_hat)
        u_hat[self._dct_nonzero] = (
            rhs_hat[self._dct_nonzero] / self._dct_denom[self._dct_nonzero]
        )
        u_hat[0, 0] = 0.0
        return _sfft.idctn(u_hat, type=2, norm="ortho", workers=self._workers)

    # ------------------------------------------------------------------
    # Internal FD forward model (DCT backend)
    # ------------------------------------------------------------------

    def _forward_fd(self, phi: np.ndarray, I0: np.ndarray) -> np.ndarray:
        """TIE forward model via cell-centred finite differences (Neumann BCs).

        Uses the same spatial discretisation as the DCT-II Poisson eigenvalues,
        making the DCT backend exactly self-consistent: one application of
        ``_forward_fd`` followed by ``_poisson_dct`` (or vice versa) is an
        exact round-trip for any zero-mean field.

        For uniform ``I0 = 1`` the forward model reduces to the standard
        finite-difference Laplacian with one-sided differences at the boundary:

        .. code-block:: text

            d²u/dx²[0]  = (u[1]  − u[0])  / dx²
            d²u/dx²[-1] = (u[-2] − u[-1]) / dx²

        For non-uniform ``I0``, cell-face intensities (arithmetic means of
        neighbouring pixels) weight the fluxes:

        .. code-block:: text

            I_x[i, j+½] = (I0[i,j] + I0[i,j+1]) / 2
            flux_x[i, j+½] = I_x · (phi[i,j+1] − phi[i,j]) / dx

        Parameters
        ----------
        phi : ndarray, shape (Ny, Nx)
            Phase image in radians.
        I0 : ndarray, shape (Ny, Nx)
            In-focus intensity.

        Returns
        -------
        dIdz : ndarray, shape (Ny, Nx)
            ``∇·(I₀ ∇φ) / (−k)``
        """
        dx = self._pixelsize

        # Cell-face intensities (arithmetic mean of neighbouring pixels)
        Ix = 0.5 * (I0[:, :-1] + I0[:, 1:])    # (ny, nx-1)
        Iy = 0.5 * (I0[:-1, :] + I0[1:, :])    # (ny-1, nx)

        # Fluxes at cell faces
        flux_x = Ix * (phi[:, 1:] - phi[:, :-1]) / dx   # (ny, nx-1)
        flux_y = Iy * (phi[1:, :] - phi[:-1, :]) / dx   # (ny-1, nx)

        # Divergence: Neumann ↔ zero-flux on all exterior faces
        div = np.zeros_like(phi)
        div[:, 1:-1]  = (flux_x[:, 1:] - flux_x[:, :-1]) / dx   # interior x
        div[:, 0]     =  flux_x[:, 0]  / dx                       # left edge
        div[:, -1]    = -flux_x[:, -1] / dx                       # right edge
        div[1:-1, :] += (flux_y[1:, :] - flux_y[:-1, :]) / dx   # interior y
        div[0,    :] +=  flux_y[0, :]  / dx                       # top edge
        div[-1,   :] -= flux_y[-1, :]  / dx                       # bottom edge

        return div / (-self._k)

    # ------------------------------------------------------------------
    # Public operators
    # ------------------------------------------------------------------

    def poisson(self, rhs: np.ndarray) -> np.ndarray:
        """Solve ∇²φ = rhs using the configured backend.

        Dispatches to the FFT (periodic BCs) or DCT (Neumann BCs) Poisson
        solver depending on ``backend``.

        Parameters
        ----------
        rhs : ndarray, shape (Ny, Nx)
            Right-hand side of the Poisson equation.

        Returns
        -------
        phi : ndarray, shape (Ny, Nx)
            Solution with zero mean.
        """
        if self._backend == "fft":
            return self._rifft2(self._fft2(rhs) * self._poisson_k)
        else:
            return self._poisson_dct(rhs)

    def forward(self, phi: np.ndarray, I0: np.ndarray) -> np.ndarray:
        """TIE forward model: compute ``∂I/∂z`` from phase and intensity.

        Dispatches to the appropriate forward model for the configured backend:

        * ``'fft'`` backend: spectral differentiation (FFT-based), periodic BCs.
        * ``'dct'`` backend: finite-difference divergence (FD-based), Neumann BCs.

        Using the matching forward model for each backend keeps the solver
        self-consistent — the Poisson solve and the forward model are exact
        inverses of each other for uniform ``I0``.

        Parameters
        ----------
        phi : ndarray, shape (Ny, Nx)
            Phase image in radians.
        I0 : ndarray, shape (Ny, Nx)
            In-focus intensity.

        Returns
        -------
        dIdz : ndarray, shape (Ny, Nx)
            Predicted axial intensity derivative ``∂I/∂z``.
        """
        if self._backend == "dct":
            return self._forward_fd(phi, I0)
        Fphi = self._fft2(phi)
        dphidx = self._rifft2(Fphi * self._Cx)
        dphidy = self._rifft2(Fphi * self._Cy)
        laplace_psi = (
            self._rifft2(self._fft2(I0 * dphidx) * self._Cx)
            + self._rifft2(self._fft2(I0 * dphidy) * self._Cy)
        )
        return laplace_psi / (-self._k)

    def step(
        self,
        dIdz_curr: np.ndarray,
        I0: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One US-TIE iteration step.

        Solves the maximum-intensity Poisson problem for a phase correction
        ``φ_est`` and computes the corresponding intensity-derivative estimate
        ``dIdz_est`` via the forward model.

        **FFT backend optimisation**: ``Fphi`` (the FFT of the phase correction)
        is computed once and reused for both the phase estimate and its
        gradients, saving one FFT per iteration compared to a naive
        implementation.

        **DCT backend**: the Poisson solve uses the precomputed DCT eigenvalue
        table, and the forward model uses the cell-centred finite-difference
        divergence to stay self-consistent with the DCT discretisation.

        Parameters
        ----------
        dIdz_curr : ndarray, shape (Ny, Nx)
            Current residual axial intensity derivative.
        I0 : ndarray, shape (Ny, Nx)
            In-focus intensity.

        Returns
        -------
        phi_est : ndarray, shape (Ny, Nx)
            Phase correction for this iteration.
        dIdz_est : ndarray, shape (Ny, Nx)
            Intensity-derivative prediction from ``phi_est`` and ``I0``.
        """
        I_max = float(np.nanmax(I0))
        rhs = -self._k * dIdz_curr / I_max

        if self._backend == "fft":
            # 1 fft2 + 3 rifft2; Fphi reused for both phi_est and gradients
            Fphi = self._fft2(rhs) * self._poisson_k
            phi_est = self._rifft2(Fphi)
            dphidx = self._rifft2(Fphi * self._Cx)
            dphidy = self._rifft2(Fphi * self._Cy)
            laplace_psi = (
                self._rifft2(self._fft2(I0 * dphidx) * self._Cx)
                + self._rifft2(self._fft2(I0 * dphidy) * self._Cy)
            )
            dIdz_est = laplace_psi / (-self._k)
            return phi_est, dIdz_est

        else:   # dct
            phi_est = self._poisson_dct(rhs)
            dIdz_est = self._forward_fd(phi_est, I0)
            return phi_est, dIdz_est

    # ------------------------------------------------------------------
    # Iterative solver
    # ------------------------------------------------------------------

    def solve(
        self,
        dIdz: np.ndarray,
        I0: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-3,
        true_phase: np.ndarray | None = None,
    ) -> dict:
        """Iteratively solve the TIE for the phase.

        Runs the US-TIE algorithm until the residual is small (convergence)
        or increases (divergence).  Returns the accumulated phase estimate
        together with optional diagnostic information.

        **Convergence criterion**: iterations stop when
        ``max|dIdz_curr| < tol × max|dIdz_initial|``.

        **Divergence check**: if the residual grows by more than 5 % in a
        single iteration, a :class:`RuntimeWarning` is issued and the loop
        stops early.  This most commonly happens when the input ``dIdz`` is
        not consistent with the solver's boundary condition convention.

        Parameters
        ----------
        dIdz : array_like, shape (Ny, Nx)
            Axial intensity derivative ``∂I/∂z``.  Compute this from images
            using :func:`US_TIE_Zhang_et_al_2020_py.compute_dIdz`.
        I0 : array_like, shape (Ny, Nx)
            In-focus intensity image.  Must contain at least one positive
            pixel.
        max_iter : int, optional
            Maximum number of iterations.  The algorithm typically converges
            in 10–50 iterations; 500 is a conservative ceiling.
        tol : float, optional
            Convergence threshold.  Iterations stop when
            ``max|residual| < tol × max|dIdz_initial|``.
        true_phase : array_like, shape (Ny, Nx), optional
            Ground-truth phase for tracking reconstruction accuracy during
            iteration.  If provided, the RMSE (with piston removed) is
            appended to the ``'rmse'`` list after each iteration.  Intended
            for research and validation, not routine use.

        Returns
        -------
        result : dict
            Dictionary with the following keys:

            ``'phase'`` : ndarray, shape (Ny, Nx)
                Recovered phase in radians.
            ``'rmse'`` : list of float
                Per-iteration RMSE values (empty if ``true_phase`` is not
                given).
            ``'times'`` : list of float
                Wall-clock time in seconds for each iteration (only populated
                when ``true_phase`` is given).
            ``'iterations'`` : int
                Number of iterations actually performed.

        Warns
        -----
        RuntimeWarning
            If the residual increases by more than 5 % in a single iteration,
            indicating that the solver is diverging.

        Examples
        --------
        >>> solver = TIESolver(shape=(512, 512), pixelsize=162.5e-9,
        ...                    k=2*np.pi/532e-9)
        >>> result = solver.solve(dIdz, I0)
        >>> phase = result['phase']
        >>> print(f"Converged in {result['iterations']} iterations")
        """
        phi_curr = np.zeros_like(dIdz, dtype=float)
        dIdz_curr = np.asarray(dIdz, dtype=float).copy()
        dIdz_prev = dIdz_curr.copy()

        rmse_history: list[float] = []
        time_history: list[float] = []

        max_orig = float(np.nanmax(np.abs(dIdz_curr)))

        if true_phase is not None:
            rmse_history.append(rmse(phi_curr, true_phase))

        n = 0
        for n in range(1, max_iter + 1):
            t0 = time.perf_counter()
            phi_est, dIdz_est = self.step(dIdz_curr, I0)
            dt = time.perf_counter() - t0

            dIdz_prev = dIdz_curr
            dIdz_curr = dIdz_curr - dIdz_est
            phi_curr  = phi_curr  + phi_est

            if true_phase is not None:
                rmse_history.append(rmse(phi_curr, true_phase))
                time_history.append(dt)

            max_curr = float(np.nanmax(np.abs(dIdz_curr)))
            max_prev = float(np.nanmax(np.abs(dIdz_prev)))

            if max_curr < tol * max_orig:
                break
            if max_curr > 1.05 * max_prev:
                warnings.warn(
                    f"US-TIE residual increased at iteration {n}; stopping early.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

        return {
            "phase":      phi_curr,
            "rmse":       rmse_history,
            "times":      time_history,
            "iterations": n,
        }
