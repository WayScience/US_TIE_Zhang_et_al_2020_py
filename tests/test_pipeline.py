"""
Tests for the end-to-end pipeline: compute_dIdz and retrieve_phase.

Strategy: use the TIE linearisation as the forward model so that the
recovered phase is exact by construction, giving a clean analytical test
without needing MATLAB.

    Forward: Iz  = I0 + dz · dIdz    (TIE, 1st-order)
    Inverse: retrieve_phase([I0, Iz], dz, ...) should return the true phase.

For three images the central difference is also exact under this model:
    I_over = I0 + dz · dIdz
    I_under = I0 - dz · dIdz
    (I_over - I_under) / (2·dz) = dIdz  ✓
"""
from __future__ import annotations

import numpy as np
import pytest

from US_TIE_Zhang_et_al_2020_py import compute_dIdz, retrieve_phase, remove_piston, TIESolver
from US_TIE_Zhang_et_al_2020_py.tie import tie_forward


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

N = 64
PIXELSIZE = 2e-6
WAVELENGTH = 0.633e-6
K = 2 * np.pi / WAVELENGTH
DZ = 1e-6  # 1 µm defocus step


def _analytical_case(amplitude: float = 1.5):
    """
    Return (I0, I_under, I_over, phi_true) for a sinusoidal phase.

    I0 = 1 (uniform), phi = A·sin(2πfx), and the intensities are generated
    from the TIE forward model so the recovery is exact.
    """
    j = np.arange(N)
    f = 1.0 / (N * PIXELSIZE)
    phi_true = np.tile(amplitude * np.sin(2 * np.pi * f * j * PIXELSIZE), (N, 1))
    I0 = np.ones((N, N))
    dIdz_true = tie_forward(phi_true, I0, PIXELSIZE, K)
    I_over = I0 + DZ * dIdz_true
    I_under = I0 - DZ * dIdz_true
    return I0, I_under, I_over, phi_true


# ---------------------------------------------------------------------------
# compute_dIdz
# ---------------------------------------------------------------------------


class TestComputeDIdz:
    def test_two_images_forward_difference(self):
        I0, _, I_over, phi_true = _analytical_case()
        dIdz, returned_I0 = compute_dIdz([I0, I_over], DZ)
        # (I_over - I0)/dz recovers the same dIdz used to build I_over
        dIdz_true = tie_forward(phi_true, I0, PIXELSIZE, K)
        np.testing.assert_allclose(returned_I0, I0)
        # atol covers FFT noise (~1e-11) at zero crossings of the sinusoid
        np.testing.assert_allclose(dIdz, dIdz_true, rtol=1e-10, atol=1e-9)

    def test_three_images_central_difference(self):
        I0, I_under, I_over, phi_true = _analytical_case()
        dIdz, returned_I0 = compute_dIdz([I_under, I0, I_over], DZ)
        dIdz_true = tie_forward(phi_true, I0, PIXELSIZE, K)
        np.testing.assert_allclose(returned_I0, I0)
        np.testing.assert_allclose(dIdz, dIdz_true, rtol=1e-10, atol=1e-9)

    def test_three_image_central_diff_is_more_accurate_than_two_image(self):
        """Central difference should give exactly dIdz; forward diff is biased."""
        I0, I_under, I_over, phi_true = _analytical_case(amplitude=2.0)
        dIdz_true = tie_forward(phi_true, I0, PIXELSIZE, K)

        dIdz_2, _ = compute_dIdz([I0, I_over], DZ)
        dIdz_3, _ = compute_dIdz([I_under, I0, I_over], DZ)

        err_2 = np.abs(dIdz_2 - dIdz_true).max()
        err_3 = np.abs(dIdz_3 - dIdz_true).max()
        assert err_3 < err_2  # central diff is more accurate here

    def test_stack_input_accepted(self):
        """3D numpy array should work identically to a list of arrays."""
        I0, _, I_over, _ = _analytical_case()
        stack = np.stack([I0, I_over])
        dIdz_list, _ = compute_dIdz([I0, I_over], DZ)
        dIdz_stack, _ = compute_dIdz(stack, DZ)
        np.testing.assert_array_equal(dIdz_list, dIdz_stack)

    def test_wrong_number_of_images_raises(self):
        I0, _, _, _ = _analytical_case()
        with pytest.raises(ValueError, match="Expected 2 or 3"):
            compute_dIdz([I0, I0, I0, I0], DZ)

    def test_single_image_raises(self):
        I0, _, _, _ = _analytical_case()
        with pytest.raises(ValueError):
            compute_dIdz(I0, DZ)


# ---------------------------------------------------------------------------
# retrieve_phase — two-image
# ---------------------------------------------------------------------------


class TestRetrievePhase2Image:
    def test_recovers_sinusoidal_phase(self):
        I0, _, I_over, phi_true = _analytical_case(amplitude=1.5)
        phi_rec = retrieve_phase(
            [I0, I_over], DZ, PIXELSIZE, WAVELENGTH,
            reg=np.finfo(float).eps, max_iter=50,
        )
        phi_rec = remove_piston(phi_rec)
        phi_true = remove_piston(phi_true)
        np.testing.assert_allclose(phi_rec, phi_true, atol=1e-9)

    def test_zero_dIdz_gives_zero_phase(self):
        I0 = np.ones((N, N))
        # identical images → dIdz = 0 → phase = 0
        phi = retrieve_phase([I0, I0], DZ, PIXELSIZE, WAVELENGTH)
        np.testing.assert_allclose(phi, 0.0, atol=1e-14)

    def test_output_shape(self):
        I0, _, I_over, _ = _analytical_case()
        phi = retrieve_phase([I0, I_over], DZ, PIXELSIZE, WAVELENGTH)
        assert phi.shape == (N, N)

    def test_output_is_float_array(self):
        I0, _, I_over, _ = _analytical_case()
        phi = retrieve_phase([I0, I_over], DZ, PIXELSIZE, WAVELENGTH)
        assert phi.dtype.kind == "f"

    def test_stack_input_same_as_list(self):
        I0, _, I_over, _ = _analytical_case()
        phi_list = retrieve_phase([I0, I_over], DZ, PIXELSIZE, WAVELENGTH)
        phi_stack = retrieve_phase(np.stack([I0, I_over]), DZ, PIXELSIZE, WAVELENGTH)
        np.testing.assert_array_equal(phi_list, phi_stack)


# ---------------------------------------------------------------------------
# retrieve_phase — three-image
# ---------------------------------------------------------------------------


class TestRetrievePhase3Image:
    def test_recovers_sinusoidal_phase(self):
        I0, I_under, I_over, phi_true = _analytical_case(amplitude=1.5)
        phi_rec = retrieve_phase(
            [I_under, I0, I_over], DZ, PIXELSIZE, WAVELENGTH,
            reg=np.finfo(float).eps, max_iter=50,
        )
        phi_rec = remove_piston(phi_rec)
        phi_true = remove_piston(phi_true)
        np.testing.assert_allclose(phi_rec, phi_true, atol=1e-9)

    def test_three_image_same_accuracy_as_two_for_linear_model(self):
        """Both should recover the phase; 3-image should be at least as good."""
        I0, I_under, I_over, phi_true = _analytical_case(amplitude=2.0)
        reg = np.finfo(float).eps

        phi_2 = retrieve_phase([I0, I_over], DZ, PIXELSIZE, WAVELENGTH, reg=reg)
        phi_3 = retrieve_phase([I_under, I0, I_over], DZ, PIXELSIZE, WAVELENGTH, reg=reg)

        err_2 = np.abs(remove_piston(phi_2) - remove_piston(phi_true)).max()
        err_3 = np.abs(remove_piston(phi_3) - remove_piston(phi_true)).max()
        assert err_3 <= err_2 + 1e-12  # at least as good


# ---------------------------------------------------------------------------
# TIESolver reuse (batch throughput)
# ---------------------------------------------------------------------------


class TestTIESolverReuse:
    def test_reuse_gives_same_result_as_fresh(self):
        """Reusing a TIESolver should give the same result as creating a new one."""
        I0, _, I_over, _ = _analytical_case()
        dIdz, _ = compute_dIdz([I0, I_over], DZ)

        solver = TIESolver(I0.shape, PIXELSIZE, K)
        phi_reused = solver.solve(dIdz, I0, max_iter=50)["phase"]

        phi_fresh = retrieve_phase([I0, I_over], DZ, PIXELSIZE, WAVELENGTH, max_iter=50)
        np.testing.assert_allclose(phi_reused, phi_fresh, rtol=1e-12)

    def test_batch_of_independent_images(self):
        """Processing a batch via solver reuse gives correct results for each image."""
        solver = TIESolver((N, N), PIXELSIZE, K)
        amplitudes = [0.5, 1.0, 2.0]
        for amp in amplitudes:
            I0, _, I_over, phi_true = _analytical_case(amplitude=amp)
            dIdz, _ = compute_dIdz([I0, I_over], DZ)
            phi_rec = solver.solve(dIdz, I0, max_iter=50)["phase"]
            phi_rec = remove_piston(phi_rec)
            phi_true = remove_piston(phi_true)
            np.testing.assert_allclose(phi_rec, phi_true, atol=1e-9,
                                       err_msg=f"Failed for amplitude={amp}")


# ---------------------------------------------------------------------------
# Speed smoke-test (just checks it finishes quickly for 512×512)
# ---------------------------------------------------------------------------


def test_retrieve_phase_large_image_finishes_quickly():
    """512×512 phase retrieval should complete in under 5 seconds."""
    import time
    rng = np.random.default_rng(42)
    N_big = 512
    I0 = np.ones((N_big, N_big))
    # Small random phase → small dIdz via forward model
    phi_small = 0.1 * rng.standard_normal((N_big, N_big))
    solver = TIESolver((N_big, N_big), PIXELSIZE, K)
    dIdz = solver.forward(phi_small, I0)
    I_over = I0 + DZ * dIdz

    t0 = time.perf_counter()
    retrieve_phase([I0, I_over], DZ, PIXELSIZE, WAVELENGTH, max_iter=50)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, f"retrieve_phase took {elapsed:.2f}s on 512×512 (expected <5s)"
