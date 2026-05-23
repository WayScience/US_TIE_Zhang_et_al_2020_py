"""
Tests for the high-level TIE solvers (universal_solution, fft_tie_solution).
"""
import numpy as np
import pytest

from US_TIE_Zhang_et_al_2020_py import fft_tie_solution, remove_piston, rmse, universal_solution


N = 64
PIXELSIZE = 2e-6
WAVELENGTH = 0.633e-6
K = 2 * np.pi / WAVELENGTH
REG = np.finfo(float).eps


def _make_test_case(amplitude: float = 1.0):
    """
    Sinusoidal phase with uniform intensity — the simplest nontrivial case
    where both solvers should recover the phase exactly (US-TIE in 1 iteration,
    FFT-TIE in 1 pass).
    """
    j = np.arange(N)
    f = 1.0 / (N * PIXELSIZE)
    phi_true = np.tile(amplitude * np.sin(2 * np.pi * f * j * PIXELSIZE), (N, 1))
    # TIE: -k·dIdz = ∇²φ → dIdz = (2πf)²·φ / k
    dIdz = (2 * np.pi * f) ** 2 * phi_true / K
    I0 = np.ones((N, N))
    return phi_true, dIdz, I0


# ---------------------------------------------------------------------------
# universal_solution
# ---------------------------------------------------------------------------


def test_us_tie_zero_dIdz_gives_zero_phase():
    dIdz = np.zeros((N, N))
    I0 = np.ones((N, N))
    result = universal_solution(dIdz, I0, PIXELSIZE, K, reg=REG)
    np.testing.assert_allclose(result["phase"], 0.0, atol=1e-14)


def test_us_tie_recovers_sinusoidal_phase():
    """US-TIE should recover sinusoidal phase with near-zero error."""
    phi_true, dIdz, I0 = _make_test_case(amplitude=2.0)
    result = universal_solution(
        dIdz, I0, PIXELSIZE, K, reg=REG, max_iter=50, true_phase=phi_true
    )
    phi_rec = remove_piston(result["phase"])
    phi_true_centered = remove_piston(phi_true)
    np.testing.assert_allclose(phi_rec, phi_true_centered, atol=1e-10)


def test_us_tie_converges_in_one_iteration_for_uniform_intensity():
    """
    With uniform I₀ the max-intensity assumption is exact, so one iteration
    should fully explain the intensity derivative (residual → 0).
    """
    phi_true, dIdz, I0 = _make_test_case(amplitude=1.5)
    result = universal_solution(
        dIdz, I0, PIXELSIZE, K, reg=REG, max_iter=100, tol=1e-3
    )
    assert result["iterations"] == 1


def test_us_tie_rmse_tracked_when_true_phase_provided():
    """RMSE list length should equal iterations + 1 (includes iteration 0)."""
    phi_true, dIdz, I0 = _make_test_case()
    result = universal_solution(
        dIdz, I0, PIXELSIZE, K, reg=REG, max_iter=5, true_phase=phi_true
    )
    assert len(result["rmse"]) == result["iterations"] + 1


def test_us_tie_rmse_not_tracked_without_true_phase():
    _, dIdz, I0 = _make_test_case()
    result = universal_solution(dIdz, I0, PIXELSIZE, K, reg=REG, max_iter=5)
    assert result["rmse"] == []
    assert result["times"] == []


def test_us_tie_initial_rmse_greater_than_final():
    """RMSE at iteration 0 should be larger than at the last iteration."""
    phi_true, dIdz, I0 = _make_test_case(amplitude=3.0)
    result = universal_solution(
        dIdz, I0, PIXELSIZE, K, reg=REG, max_iter=30, true_phase=phi_true
    )
    assert result["rmse"][0] > result["rmse"][-1]


def test_us_tie_nonuniform_intensity_converges():
    """
    Non-uniform (Gaussian) intensity: US-TIE should still converge.
    We verify RMSE at convergence is lower than initial RMSE.
    """
    j = np.arange(N)
    f = 1.0 / (N * PIXELSIZE)
    phi_true = np.tile(np.sin(2 * np.pi * f * j * PIXELSIZE), (N, 1))

    # Gaussian intensity profile
    x = (j - N / 2) * PIXELSIZE
    sigma = N * PIXELSIZE / 6
    gauss_1d = np.exp(-(x**2) / (2 * sigma**2))
    I0 = np.outer(gauss_1d, gauss_1d)
    I0 = I0 / I0.max()  # normalise to [0, 1]

    # Generate dIdz numerically via propagation so it is self-consistent
    from US_TIE_Zhang_et_al_2020_py.tie import tie_forward
    dIdz = tie_forward(phi_true, I0, PIXELSIZE, K)

    result = universal_solution(
        dIdz, I0, PIXELSIZE, K, reg=REG, max_iter=100, true_phase=phi_true
    )
    assert result["rmse"][-1] < result["rmse"][0]


def test_us_tie_output_shape():
    _, dIdz, I0 = _make_test_case()
    result = universal_solution(dIdz, I0, PIXELSIZE, K)
    assert result["phase"].shape == (N, N)


# ---------------------------------------------------------------------------
# fft_tie_solution
# ---------------------------------------------------------------------------


def test_fft_tie_zero_dIdz_gives_zero_phase():
    dIdz = np.zeros((N, N))
    I0 = np.ones((N, N))
    result = fft_tie_solution(dIdz, I0, PIXELSIZE, K)
    np.testing.assert_allclose(result["phase"], 0.0, atol=1e-14)


def test_fft_tie_recovers_sinusoidal_phase():
    phi_true, dIdz, I0 = _make_test_case(amplitude=1.0)
    result = fft_tie_solution(dIdz, I0, PIXELSIZE, K, reg=REG, true_phase=phi_true)
    phi_rec = remove_piston(result["phase"])
    phi_true_c = remove_piston(phi_true)
    np.testing.assert_allclose(phi_rec, phi_true_c, atol=1e-10)


def test_fft_tie_rmse_returned_with_true_phase():
    phi_true, dIdz, I0 = _make_test_case()
    result = fft_tie_solution(dIdz, I0, PIXELSIZE, K, true_phase=phi_true)
    assert isinstance(result["rmse"], float)
    assert result["rmse"] >= 0.0


def test_fft_tie_rmse_none_without_true_phase():
    _, dIdz, I0 = _make_test_case()
    result = fft_tie_solution(dIdz, I0, PIXELSIZE, K)
    assert result["rmse"] is None


def test_fft_tie_output_shape():
    _, dIdz, I0 = _make_test_case()
    result = fft_tie_solution(dIdz, I0, PIXELSIZE, K)
    assert result["phase"].shape == (N, N)


def test_fft_tie_timing_recorded():
    _, dIdz, I0 = _make_test_case()
    result = fft_tie_solution(dIdz, I0, PIXELSIZE, K)
    assert result["time"] > 0.0
