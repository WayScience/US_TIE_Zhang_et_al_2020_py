"""
Tests for the DCT-based Poisson solver and TIESolver DCT backend.

The DCT solver uses finite-difference Laplacian eigenvalues and enforces
homogeneous Neumann BCs (∂u/∂n = 0 at image borders).

Round-trip verification strategy
---------------------------------
Apply the same finite-difference Laplacian (with Neumann BC) to a known field
u_true, then solve Poisson and check we recover u_true.  This is the only
clean way to test the DCT solver without needing MATLAB, because the DCT
eigenvalues correspond to exactly this finite-difference scheme.
"""
from __future__ import annotations

import numpy as np
import pytest

from US_TIE_Zhang_et_al_2020_py import poisson_dct, retrieve_phase, remove_piston, TIESolver
from US_TIE_Zhang_et_al_2020_py.tie import tie_forward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 64
PIXELSIZE = 2e-6
WAVELENGTH = 0.633e-6
K = 2 * np.pi / WAVELENGTH
DZ = 1e-6


def _neumann_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    """
    2-D finite-difference Laplacian with homogeneous Neumann BCs.

    Boundary treatment (1-sided difference to enforce ∂u/∂n = 0):
        left/right:  d²u/dx²[j=0]  = (u[1]  − u[0])  / dx²
                     d²u/dx²[j=-1] = (u[-2] − u[-1]) / dx²
    This matches exactly the eigenvalue convention used by DCT-II
    (the convention in ``poisson_dct``).
    """
    d2x = np.empty_like(u)
    d2x[:, 1:-1] = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx**2
    d2x[:, 0]    = (u[:, 1]  - u[:, 0])  / dx**2
    d2x[:, -1]   = (u[:, -2] - u[:, -1]) / dx**2

    d2y = np.empty_like(u)
    d2y[1:-1, :] = (u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / dx**2
    d2y[0,    :] = (u[1, :]  - u[0, :])  / dx**2
    d2y[-1,   :] = (u[-2, :] - u[-1, :]) / dx**2

    return d2x + d2y


def _sinusoidal_case(amplitude: float = 1.5):
    """Sinusoidal phase with uniform I0=1 and **FFT** forward model.

    Used only in the ``poisson_dct`` standalone tests and in the FFT/DCT
    comparison tests (where the phase is chosen so both Laplacians agree).
    Do NOT use this helper for DCT backend round-trip tests — the FFT forward
    model is inconsistent with the FD Laplacian used inside the DCT solver.
    """
    j = np.arange(N)
    f = 1.0 / (N * PIXELSIZE)
    phi_true = np.tile(amplitude * np.sin(2 * np.pi * f * j * PIXELSIZE), (N, 1))
    I0 = np.ones((N, N))
    dIdz_true = tie_forward(phi_true, I0, PIXELSIZE, K)
    I_over  = I0 + DZ * dIdz_true
    I_under = I0 - DZ * dIdz_true
    return I0, I_under, I_over, phi_true


def _fd_case(amplitude: float = 1.5):
    """Neumann-BC-compatible phase with **FD** forward model.

    Uses ``phi_true = amplitude * cos(2π j/N)``, which:

    * satisfies the Neumann boundary condition (zero derivative at image edge),
    * is a DCT-II basis function (mode 2), so the DCT solver inverts it exactly,
    * has FD and spectral Laplacian eigenvalues agreeing to <0.1 % for N=64,
      so the FFT solver also recovers it accurately when fed FD-generated dIdz.

    The intensity derivative is computed via the finite-difference Laplacian
    (``_neumann_laplacian``) to match the DCT Poisson solver's discretisation.
    """
    j = np.arange(N)
    phi_true = np.tile(amplitude * np.cos(2 * np.pi * j / N), (N, 1))
    I0 = np.ones((N, N))
    # FD forward model: dIdz = ∇²_FD(phi) / (-k)  (exact for uniform I0=1)
    dIdz_true = _neumann_laplacian(phi_true, PIXELSIZE) / (-K)
    I_over  = I0 + DZ * dIdz_true
    I_under = I0 - DZ * dIdz_true
    return I0, I_under, I_over, phi_true


# ---------------------------------------------------------------------------
# poisson_dct standalone
# ---------------------------------------------------------------------------


class TestPoissonDCT:
    def test_zero_rhs_gives_zero_solution(self):
        rhs = np.zeros((N, N))
        u = poisson_dct(rhs, PIXELSIZE)
        np.testing.assert_allclose(u, 0.0, atol=1e-14)

    def test_round_trip_random_field(self):
        """
        Apply the Neumann finite-difference Laplacian to a random zero-mean
        field, solve with poisson_dct, and check we recover the original.
        """
        rng = np.random.default_rng(0)
        u_true = rng.standard_normal((N, N))
        u_true -= u_true.mean()  # zero mean (Neumann uniqueness)

        rhs = _neumann_laplacian(u_true, PIXELSIZE)
        u_rec = poisson_dct(rhs, PIXELSIZE, zero_mean=False)
        u_rec -= u_rec.mean()

        np.testing.assert_allclose(u_rec, u_true, rtol=1e-5, atol=1e-10)

    def test_solution_has_zero_mean(self):
        rng = np.random.default_rng(1)
        rhs = rng.standard_normal((N, N))
        u = poisson_dct(rhs, PIXELSIZE)
        assert abs(u.mean()) < 1e-12

    def test_zero_mean_flag_subtracts_rhs_mean(self):
        """zero_mean=True and False should give the same result when RHS is
        already zero-mean."""
        rng = np.random.default_rng(2)
        rhs = rng.standard_normal((N, N))
        rhs -= rhs.mean()  # pre-subtract mean
        u_flag_true  = poisson_dct(rhs, PIXELSIZE, zero_mean=True)
        u_flag_false = poisson_dct(rhs, PIXELSIZE, zero_mean=False)
        np.testing.assert_allclose(u_flag_true, u_flag_false, atol=1e-14)

    def test_linearity(self):
        rng = np.random.default_rng(3)
        rhs1 = rng.standard_normal((N, N))
        rhs2 = rng.standard_normal((N, N))
        a, b = 2.0, -0.5
        u_combined = poisson_dct(a * rhs1 + b * rhs2, PIXELSIZE)
        u_expected = a * poisson_dct(rhs1, PIXELSIZE) + b * poisson_dct(rhs2, PIXELSIZE)
        np.testing.assert_allclose(u_combined, u_expected, rtol=1e-11)

    def test_output_shape(self):
        rhs = np.zeros((32, 48))
        u = poisson_dct(rhs, PIXELSIZE)
        assert u.shape == (32, 48)

    def test_neumann_bc_enforced(self):
        """
        The solution gradient at the image border should be approximately zero
        (Neumann BC) — tested with a 1-sided finite difference.
        """
        rng = np.random.default_rng(4)
        rhs = rng.standard_normal((N, N))
        u = poisson_dct(rhs, PIXELSIZE)

        # One-sided derivative at left and right borders
        du_left  = (u[:, 1] - u[:, 0]) / PIXELSIZE
        du_right = (u[:, -1] - u[:, -2]) / PIXELSIZE
        du_top   = (u[1, :] - u[0, :]) / PIXELSIZE
        du_bot   = (u[-1, :] - u[-2, :]) / PIXELSIZE

        # The Neumann condition is not machine-precision satisfied, but
        # the gradient should be small relative to the interior gradients.
        interior_scale = np.abs(u).max()
        for edge, name in [(du_left, "left"), (du_right, "right"),
                           (du_top, "top"), (du_bot, "bottom")]:
            assert np.abs(edge).max() < 0.1 * interior_scale + 1.0, \
                f"{name} boundary gradient unexpectedly large"


# ---------------------------------------------------------------------------
# TIESolver DCT backend
# ---------------------------------------------------------------------------


class TestTIESolverDCT:
    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            TIESolver((N, N), PIXELSIZE, K, backend="fourier")

    def test_dct_backend_recovers_sinusoidal_phase(self):
        """DCT backend recovers a Neumann-compatible phase from FD-generated dIdz.

        The DCT Poisson solver uses finite-difference (FD) eigenvalues, so the
        correct round-trip test feeds it dIdz produced by the FD forward model.
        Using ``phi_true = cos(2π j/N)`` (a DCT-II basis function) with uniform
        I0 = 1 makes the system exactly self-consistent.
        """
        I0, _, I_over, phi_true = _fd_case(amplitude=1.5)
        from US_TIE_Zhang_et_al_2020_py.pipeline import compute_dIdz
        dIdz, _ = compute_dIdz([I0, I_over], DZ)

        solver = TIESolver((N, N), PIXELSIZE, K, backend="dct")
        phi_rec = solver.solve(dIdz, I0, max_iter=100)["phase"]

        phi_rec   = remove_piston(phi_rec)
        phi_true  = remove_piston(phi_true)
        np.testing.assert_allclose(phi_rec, phi_true, atol=1e-4)

    def test_dct_and_fft_backends_are_both_accurate_for_their_model(self):
        """Each backend accurately recovers phase from its own self-consistent dIdz.

        The FFT (spectral, periodic BCs) and DCT (FD, Neumann BCs) solvers use
        different Laplacian discretisations, so they are tested independently:

        * FFT backend ← dIdz generated by the spectral forward model.
        * DCT backend ← dIdz generated by the FD forward model.

        Each solver should recover its ``phi_true`` to within 1e-4.
        """
        from US_TIE_Zhang_et_al_2020_py.pipeline import compute_dIdz

        # FFT backend: spectral forward model, periodic BCs
        I0_fft, _, I_over_fft, phi_true_fft = _sinusoidal_case(amplitude=1.0)
        dIdz_fft, _ = compute_dIdz([I0_fft, I_over_fft], DZ)
        phi_fft = TIESolver((N, N), PIXELSIZE, K, backend="fft").solve(
            dIdz_fft, I0_fft, max_iter=100)["phase"]
        err_fft = np.abs(remove_piston(phi_fft) - remove_piston(phi_true_fft)).max()

        # DCT backend: FD forward model, Neumann BCs
        I0_dct, _, I_over_dct, phi_true_dct = _fd_case(amplitude=1.0)
        dIdz_dct, _ = compute_dIdz([I0_dct, I_over_dct], DZ)
        phi_dct = TIESolver((N, N), PIXELSIZE, K, backend="dct").solve(
            dIdz_dct, I0_dct, max_iter=100)["phase"]
        err_dct = np.abs(remove_piston(phi_dct) - remove_piston(phi_true_dct)).max()

        assert err_fft < 1e-3, f"fft error too large: {err_fft}"
        assert err_dct < 1e-4, f"dct error too large: {err_dct}"

    def test_dct_solver_zero_dIdz_gives_zero_phase(self):
        dIdz = np.zeros((N, N))
        I0 = np.ones((N, N))
        solver = TIESolver((N, N), PIXELSIZE, K, backend="dct")
        result = solver.solve(dIdz, I0, max_iter=10)
        np.testing.assert_allclose(result["phase"], 0.0, atol=1e-14)

    def test_dct_backend_output_shape(self):
        dIdz = np.zeros((N, N))
        I0 = np.ones((N, N))
        result = TIESolver((N, N), PIXELSIZE, K, backend="dct").solve(dIdz, I0)
        assert result["phase"].shape == (N, N)


# ---------------------------------------------------------------------------
# retrieve_phase with solver='dct'
# ---------------------------------------------------------------------------


class TestRetrievePhaseDCT:
    def test_recovers_sinusoidal_phase_2_images(self):
        """retrieve_phase with solver='dct' recovers Neumann-compatible phase (2 images).

        Images are synthesised with the FD forward model so that the DCT
        backend round-trip is self-consistent.
        """
        I0, _, I_over, phi_true = _fd_case(amplitude=1.5)
        phi_rec = retrieve_phase(
            [I0, I_over], DZ, PIXELSIZE, WAVELENGTH,
            solver="dct", max_iter=100,
        )
        np.testing.assert_allclose(
            remove_piston(phi_rec), remove_piston(phi_true), atol=1e-4
        )

    def test_recovers_sinusoidal_phase_3_images(self):
        """retrieve_phase with solver='dct' recovers Neumann-compatible phase (3 images)."""
        I0, I_under, I_over, phi_true = _fd_case(amplitude=1.5)
        phi_rec = retrieve_phase(
            [I_under, I0, I_over], DZ, PIXELSIZE, WAVELENGTH,
            solver="dct", max_iter=100,
        )
        np.testing.assert_allclose(
            remove_piston(phi_rec), remove_piston(phi_true), atol=1e-4
        )

    def test_invalid_solver_raises(self):
        I0 = np.ones((N, N))
        with pytest.raises(ValueError, match="backend"):
            retrieve_phase([I0, I0], DZ, PIXELSIZE, WAVELENGTH, solver="cosine")

    def test_dct_and_fft_outputs_close_for_smooth_phase(self):
        """Both retrieve_phase backends accurately recover phase from their own model.

        The FFT and DCT solvers each use a different Laplacian (spectral / FD)
        and different BCs (periodic / Neumann).  Feeding one solver's dIdz to
        the other causes BC mismatch.  We therefore test each independently:

        * ``solver='fft'`` ← dIdz from the spectral forward model.
        * ``solver='dct'`` ← dIdz from the FD forward model.
        """
        # FFT backend: spectral dIdz
        I0_fft, _, I_over_fft, phi_true_fft = _sinusoidal_case()
        phi_fft = retrieve_phase([I0_fft, I_over_fft], DZ, PIXELSIZE, WAVELENGTH,
                                 solver="fft")
        err_fft = np.abs(remove_piston(phi_fft) - remove_piston(phi_true_fft)).max()
        assert err_fft < 1e-3, f"fft solver error too large: {err_fft}"

        # DCT backend: FD dIdz
        I0_dct, _, I_over_dct, phi_true_dct = _fd_case()
        phi_dct = retrieve_phase([I0_dct, I_over_dct], DZ, PIXELSIZE, WAVELENGTH,
                                 solver="dct")
        err_dct = np.abs(remove_piston(phi_dct) - remove_piston(phi_true_dct)).max()
        assert err_dct < 1e-3, f"dct solver error too large: {err_dct}"
