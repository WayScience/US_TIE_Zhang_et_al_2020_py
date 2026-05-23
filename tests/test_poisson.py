"""
Tests for the standalone Poisson solvers: poisson_fft and poisson_dct.

All tests use analytical ground truth — no MATLAB required.
The key identity: for φ(x) = A·sin(2πfx), the Laplacian is -(2πf)²·φ,
so applying the Poisson solver to -(2πf)²·φ should recover φ.
"""
import numpy as np
import pytest

from US_TIE_Zhang_et_al_2020_py.poisson import poisson_fft


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

N = 64
PIXELSIZE = 2e-6       # 2 µm
REG = np.finfo(float).eps


def _sinusoidal_rhs(amplitude: float = 1.0):
    """Return (phi_true, rhs) for a 1-cycle sinusoidal phase.
    rhs = -(2πf)²·phi_true  (the Laplacian of a sinusoid).
    """
    j = np.arange(N)
    f = 1.0 / (N * PIXELSIZE)
    phi_1d = amplitude * np.sin(2 * np.pi * f * j * PIXELSIZE)
    phi_true = np.tile(phi_1d, (N, 1))
    rhs = -(2 * np.pi * f) ** 2 * phi_true
    return phi_true, rhs


# ---------------------------------------------------------------------------
# poisson_fft
# ---------------------------------------------------------------------------

def test_poisson_fft_zero_rhs():
    """Zero right-hand side → zero solution."""
    rhs = np.zeros((N, N))
    phi = poisson_fft(rhs, PIXELSIZE, REG)
    np.testing.assert_allclose(phi, 0.0, atol=1e-14)


def test_poisson_fft_recovers_sinusoid():
    """Analytical round-trip: solve ∇²φ = -(2πf)²·A·sin(2πfx)."""
    A = 2.0
    phi_true, rhs = _sinusoidal_rhs(amplitude=A)

    phi_rec = poisson_fft(rhs, PIXELSIZE, REG)
    phi_rec -= phi_rec.mean()
    phi_true -= phi_true.mean()

    np.testing.assert_allclose(phi_rec, phi_true, rtol=1e-6, atol=1e-12)


def test_poisson_fft_linearity():
    """Solver is linear: poisson_fft(a·rhs₁ + b·rhs₂) = a·φ₁ + b·φ₂."""
    rng = np.random.default_rng(0)
    rhs1 = rng.standard_normal((N, N))
    rhs2 = rng.standard_normal((N, N))
    a, b = 3.0, -1.5

    phi_combined = poisson_fft(a * rhs1 + b * rhs2, PIXELSIZE, REG)
    phi_expected = (a * poisson_fft(rhs1, PIXELSIZE, REG)
                    + b * poisson_fft(rhs2, PIXELSIZE, REG))

    np.testing.assert_allclose(phi_combined, phi_expected, rtol=1e-11)


def test_poisson_fft_dc_is_zero():
    """The DC component of the solution is always zero (piston-free)."""
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal((N, N))
    phi = poisson_fft(rhs, PIXELSIZE, REG)
    assert abs(phi.mean()) < 1e-12
