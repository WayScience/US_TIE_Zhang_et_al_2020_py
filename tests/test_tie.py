"""
Tests for the TIE forward model and single-step operator.

All tests use analytical ground truth — no MATLAB required.
The key identity: for φ(x) = A·sin(2πfx) with uniform I₀ = 1,
    tie_forward(φ, I₀) = (2πf)²·φ / k
"""
import numpy as np
import pytest

from US_TIE_Zhang_et_al_2020_py.tie import tie_forward, tie_max_solver


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N = 64
PIXELSIZE = 2e-6       # 2 µm
WAVELENGTH = 0.633e-6  # 633 nm
K = 2 * np.pi / WAVELENGTH
REG = np.finfo(float).eps


def _sinusoidal_setup(amplitude: float = 1.0):
    """Return (phi_true, dIdz, I0) for a 1-cycle sinusoidal phase, uniform I."""
    j = np.arange(N)
    f = 1.0 / (N * PIXELSIZE)
    phi_1d = amplitude * np.sin(2 * np.pi * f * j * PIXELSIZE)
    phi_true = np.tile(phi_1d, (N, 1))
    dIdz = (2 * np.pi * f) ** 2 * phi_true / K
    I0 = np.ones((N, N))
    return phi_true, dIdz, I0


# ---------------------------------------------------------------------------
# tie_forward
# ---------------------------------------------------------------------------

def test_tie_forward_zero_phase():
    """Zero phase with arbitrary intensity → zero dIdz."""
    phi = np.zeros((N, N))
    I0 = np.random.default_rng(2).uniform(0.5, 1.5, (N, N))
    dIdz = tie_forward(phi, I0, PIXELSIZE, K)
    np.testing.assert_allclose(dIdz, 0.0, atol=1e-12)


def test_tie_forward_uniform_intensity_matches_laplacian():
    """With uniform I₀=1: tie_forward(φ) = (2πf)²·φ / k."""
    phi_true, dIdz_expected, I0 = _sinusoidal_setup(amplitude=1.5)
    dIdz_computed = tie_forward(phi_true, I0, PIXELSIZE, K)
    np.testing.assert_allclose(dIdz_computed, dIdz_expected, rtol=1e-6, atol=1e-9)


def test_tie_forward_is_linear_in_phase():
    """tie_forward is linear in φ for fixed I."""
    phi1, _, I0 = _sinusoidal_setup(amplitude=1.0)
    phi2 = np.roll(phi1, N // 4, axis=1)
    a, b = 2.0, -0.5

    result_combined = tie_forward(a * phi1 + b * phi2, I0, PIXELSIZE, K)
    result_expected = (a * tie_forward(phi1, I0, PIXELSIZE, K)
                       + b * tie_forward(phi2, I0, PIXELSIZE, K))
    np.testing.assert_allclose(result_combined, result_expected, rtol=1e-11)


# ---------------------------------------------------------------------------
# tie_max_solver
# ---------------------------------------------------------------------------

def test_tie_max_solver_zero_dIdz():
    """Zero intensity derivative → zero phase correction."""
    dIdz_curr = np.zeros((N, N))
    I0 = np.ones((N, N))
    phi_curr = np.zeros((N, N))
    phi_est, dIdz_est = tie_max_solver(phi_curr, dIdz_curr, I0, PIXELSIZE, K, REG)
    np.testing.assert_allclose(phi_est, 0.0, atol=1e-14)
    np.testing.assert_allclose(dIdz_est, 0.0, atol=1e-14)


def test_tie_max_solver_exact_in_one_step():
    """With uniform I₀=1, one step recovers the exact phase."""
    phi_true, dIdz, I0 = _sinusoidal_setup(amplitude=2.0)
    phi_curr = np.zeros_like(phi_true)
    phi_est, _ = tie_max_solver(phi_curr, dIdz, I0, PIXELSIZE, K, REG)

    phi_est -= phi_est.mean()
    phi_true -= phi_true.mean()
    np.testing.assert_allclose(phi_est, phi_true, rtol=1e-6, atol=1e-12)


def test_tie_max_solver_forward_model_consistent():
    """dIdz_est from tie_max_solver reproduces dIdz when I₀ is uniform."""
    phi_true, dIdz, I0 = _sinusoidal_setup(amplitude=0.8)
    phi_curr = np.zeros_like(phi_true)
    _, dIdz_est = tie_max_solver(phi_curr, dIdz, I0, PIXELSIZE, K, REG)
    np.testing.assert_allclose(dIdz_est, dIdz, rtol=1e-6, atol=1e-10)
