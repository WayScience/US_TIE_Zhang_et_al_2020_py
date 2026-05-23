"""
Tests for numerical_propagation.
"""
import numpy as np
import pytest

from US_TIE_Zhang_et_al_2020_py import numerical_propagation


N = 64
PIXELSIZE = 2e-6
WAVELENGTH = 0.633e-6
K = 2 * np.pi / WAVELENGTH


def _plane_wave():
    """Uniform amplitude, zero phase — a simple plane wave."""
    return np.ones((N, N), dtype=complex)


def _gaussian_beam(sigma_px: float = 10.0):
    """Gaussian amplitude with zero phase."""
    j = np.arange(N) - N / 2
    x, y = np.meshgrid(j, j)
    amp = np.exp(-(x**2 + y**2) / (2 * sigma_px**2))
    return amp.astype(complex)


# ---------------------------------------------------------------------------
# angular_spectrum
# ---------------------------------------------------------------------------


def test_angular_spectrum_identity_dz0():
    """Propagating by dz=0 should return the original field."""
    U0 = _gaussian_beam()
    Uz, Iz, Phiz = numerical_propagation(U0, dz=0.0, pixelsize=PIXELSIZE,
                                          wavelength=WAVELENGTH, method="angular_spectrum")
    np.testing.assert_allclose(np.abs(Uz), np.abs(U0), atol=1e-12)


def test_angular_spectrum_energy_conserved():
    """Total intensity (energy) should be conserved after propagation."""
    U0 = _gaussian_beam()
    _, Iz, _ = numerical_propagation(U0, dz=1e-4, pixelsize=PIXELSIZE,
                                      wavelength=WAVELENGTH, method="angular_spectrum")
    I0 = np.abs(U0) ** 2
    np.testing.assert_allclose(Iz.sum(), I0.sum(), rtol=1e-6)


def test_angular_spectrum_returns_correct_shapes():
    U0 = _gaussian_beam()
    Uz, Iz, Phiz = numerical_propagation(U0, dz=1e-5, pixelsize=PIXELSIZE,
                                          wavelength=WAVELENGTH, method="angular_spectrum")
    assert Uz.shape == (N, N)
    assert Iz.shape == (N, N)
    assert Phiz.shape == (N, N)


def test_angular_spectrum_intensity_nonnegative():
    U0 = _gaussian_beam()
    _, Iz, _ = numerical_propagation(U0, dz=5e-5, pixelsize=PIXELSIZE,
                                      wavelength=WAVELENGTH, method="angular_spectrum")
    assert np.all(Iz >= -1e-14)  # numerical noise tolerance


def test_angular_spectrum_plane_wave_unchanged():
    """A uniform plane wave has flat intensity that should remain flat after propagation."""
    U0 = _plane_wave()
    _, Iz, _ = numerical_propagation(U0, dz=1e-4, pixelsize=PIXELSIZE,
                                      wavelength=WAVELENGTH, method="angular_spectrum")
    np.testing.assert_allclose(Iz, np.ones((N, N)), atol=1e-10)


# ---------------------------------------------------------------------------
# fresnel
# ---------------------------------------------------------------------------


def test_fresnel_identity_dz0():
    U0 = _gaussian_beam()
    Uz, _, _ = numerical_propagation(U0, dz=0.0, pixelsize=PIXELSIZE,
                                      wavelength=WAVELENGTH, method="fresnel")
    np.testing.assert_allclose(np.abs(Uz), np.abs(U0), atol=1e-12)


def test_fresnel_energy_conserved():
    U0 = _gaussian_beam()
    _, Iz, _ = numerical_propagation(U0, dz=1e-4, pixelsize=PIXELSIZE,
                                      wavelength=WAVELENGTH, method="fresnel")
    I0 = np.abs(U0) ** 2
    np.testing.assert_allclose(Iz.sum(), I0.sum(), rtol=1e-6)


# ---------------------------------------------------------------------------
# TIE linearised propagation
# ---------------------------------------------------------------------------


def test_tie_propagation_returns_nan_field_and_phase():
    U0 = _gaussian_beam()
    Uz, Iz, Phiz = numerical_propagation(U0, dz=1e-6, pixelsize=PIXELSIZE,
                                          wavelength=WAVELENGTH, method="tie")
    assert np.all(np.isnan(Uz))
    assert np.all(np.isnan(Phiz))
    assert not np.any(np.isnan(Iz))  # intensity should be real and finite


def test_tie_propagation_small_dz_matches_angular_spectrum():
    """For small dz, TIE-linearised and angular-spectrum intensities should agree."""
    U0 = _gaussian_beam(sigma_px=8.0)
    dz = 1e-7  # very small: linear approx is accurate

    _, Iz_tie, _ = numerical_propagation(U0, dz, PIXELSIZE, WAVELENGTH, method="tie")
    _, Iz_as, _ = numerical_propagation(U0, dz, PIXELSIZE, WAVELENGTH, method="angular_spectrum")

    np.testing.assert_allclose(Iz_tie, Iz_as, rtol=1e-3, atol=1e-6)


def test_tie_propagation_zero_phase_conserves_intensity():
    """Zero-phase uniform field: TIE dIdz = 0, so Iz = I0."""
    U0 = _plane_wave()
    _, Iz, _ = numerical_propagation(U0, dz=1e-5, pixelsize=PIXELSIZE,
                                      wavelength=WAVELENGTH, method="tie")
    I0 = np.abs(U0) ** 2
    np.testing.assert_allclose(Iz, I0, atol=1e-12)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_unknown_method_raises():
    U0 = _plane_wave()
    with pytest.raises(ValueError, match="Unknown method"):
        numerical_propagation(U0, dz=1e-5, pixelsize=PIXELSIZE,
                               wavelength=WAVELENGTH, method="dct")


def test_method_string_is_case_insensitive():
    U0 = _gaussian_beam()
    # Should not raise
    numerical_propagation(U0, dz=1e-5, pixelsize=PIXELSIZE,
                           wavelength=WAVELENGTH, method="Angular_Spectrum")
    numerical_propagation(U0, dz=1e-5, pixelsize=PIXELSIZE,
                           wavelength=WAVELENGTH, method="FRESNEL")
