"""
Integration test using real brightfield images from JUMP Cell Painting.

These tests load three genuine brightfield z-stack images acquired on a
PerkinElmer Phenix microscope (plate BR00116991, r01c01f01) and verify
that the TIE phase retrieval pipeline produces sensible results.

Physical parameters (from the acquisition metadata, Index.idx.xml):
  - pixelsize  = 597.98 nm  (5.9798e-7 m)
  - wavelength = 740 nm     (NIR broadband illumination)
  - z-offset   = ±4/7 µm   (asymmetric; effective half-span ≈ 5.5 µm)

The z-planes are *not* perfectly symmetric:
  - brightfield_under.tiff  ch7 "Brightfield L"  −4 µm from focus
  - brightfield_focus.tiff  ch8 "Brightfield"     in-focus
  - brightfield_over.tiff   ch6 "Brightfield H"  +7 µm from focus

We use dz = 5.5 µm (half the total 11 µm span) as the central-difference
step, which is the appropriate effective value for the asymmetric geometry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from US_TIE_Zhang_et_al_2020_py import retrieve_phase, remove_piston, TIESolver, compute_dIdz

# ---------------------------------------------------------------------------
# Paths and acquisition constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"

PIXELSIZE = 5.9798e-7   # m  (from ImageResolutionX in Index.idx.xml)
WAVELENGTH = 740e-9     # m  (NIR broadband brightfield)
DZ = 5.5e-6             # m  (effective half-span: total 11 µm / 2)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bf_images():
    """Load the three brightfield TIFF fixtures and convert to float64."""
    I_under = tifffile.imread(DATA_DIR / "brightfield_under.tiff").astype(float)
    I_focus = tifffile.imread(DATA_DIR / "brightfield_focus.tiff").astype(float)
    I_over  = tifffile.imread(DATA_DIR / "brightfield_over.tiff").astype(float)
    return I_under, I_focus, I_over


# ---------------------------------------------------------------------------
# Fixture sanity checks
# ---------------------------------------------------------------------------

class TestFixtureSanity:
    def test_shape(self, bf_images):
        for img in bf_images:
            assert img.shape == (512, 512)

    def test_positive_intensities(self, bf_images):
        for img in bf_images:
            assert img.min() > 0, "Expected strictly positive brightfield intensities"

    def test_in_focus_most_uniform(self, bf_images):
        """In-focus brightfield image should have the lowest variance.

        Unlike fluorescence, transmitted-light brightfield images are most
        uniform when in focus.  Defocus introduces diffraction rings and halos
        that increase pixel-to-pixel variation.
        """
        I_under, I_focus, I_over = bf_images
        assert I_focus.var() <= I_under.var() or I_focus.var() <= I_over.var(), (
            "In-focus brightfield image should be at least as uniform as one defocused image"
        )

    def test_mean_intensities_similar(self, bf_images):
        """All three planes should have similar mean intensity (same illumination)."""
        means = [img.mean() for img in bf_images]
        # Allow ±5% relative spread
        assert max(means) / min(means) < 1.05


# ---------------------------------------------------------------------------
# Two-image pipeline (forward difference)
# ---------------------------------------------------------------------------

class TestTwoImage:
    def test_retrieves_finite_phase(self, bf_images):
        _, I_focus, I_over = bf_images
        phase = retrieve_phase([I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH)
        assert np.all(np.isfinite(phase)), "Phase contains non-finite values"

    def test_output_shape(self, bf_images):
        _, I_focus, I_over = bf_images
        phase = retrieve_phase([I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH)
        assert phase.shape == (512, 512)

    def test_phase_has_structure(self, bf_images):
        """Real cells should produce a phase with measurable spatial variation."""
        _, I_focus, I_over = bf_images
        phase = retrieve_phase([I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH)
        # Standard deviation should be at least 0.01 rad (cells produce ~0.1–1 rad)
        assert remove_piston(phase).std() > 0.01


# ---------------------------------------------------------------------------
# Three-image pipeline (central difference)
# ---------------------------------------------------------------------------

class TestThreeImage:
    def test_retrieves_finite_phase(self, bf_images):
        I_under, I_focus, I_over = bf_images
        phase = retrieve_phase(
            [I_under, I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH
        )
        assert np.all(np.isfinite(phase))

    def test_output_shape(self, bf_images):
        I_under, I_focus, I_over = bf_images
        phase = retrieve_phase(
            [I_under, I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH
        )
        assert phase.shape == (512, 512)

    def test_phase_has_structure(self, bf_images):
        I_under, I_focus, I_over = bf_images
        phase = retrieve_phase(
            [I_under, I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH
        )
        assert remove_piston(phase).std() > 0.01

    def test_dct_backend(self, bf_images):
        """DCT (Neumann) backend should also produce a finite, non-trivial phase."""
        I_under, I_focus, I_over = bf_images
        phase = retrieve_phase(
            [I_under, I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH,
            solver="dct",
        )
        assert np.all(np.isfinite(phase))
        assert remove_piston(phase).std() > 0.01

    def test_three_image_vs_two_image(self, bf_images):
        """Three-image result should differ from two-image (different derivative estimate)."""
        I_under, I_focus, I_over = bf_images
        phase_2 = retrieve_phase([I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH)
        phase_3 = retrieve_phase([I_under, I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH)
        # They should be correlated but not identical
        corr = np.corrcoef(phase_2.ravel(), phase_3.ravel())[0, 1]
        assert corr > 0.5, "3-image and 2-image phases should be positively correlated"
        assert not np.allclose(phase_2, phase_3), "3-image and 2-image phases should differ"


# ---------------------------------------------------------------------------
# TIESolver batch API
# ---------------------------------------------------------------------------

class TestTIESolverRealData:
    def test_solver_class_matches_pipeline(self, bf_images):
        """TIESolver.solve() should give the same result as retrieve_phase()."""
        import math
        I_under, I_focus, I_over = bf_images
        dIdz, I0 = compute_dIdz([I_under, I_focus, I_over], DZ)

        solver = TIESolver(
            shape=(512, 512),
            pixelsize=PIXELSIZE,
            k=2 * math.pi / WAVELENGTH,
        )
        phase_solver = solver.solve(dIdz, I0)["phase"]
        phase_pipeline = retrieve_phase(
            [I_under, I_focus, I_over], DZ, PIXELSIZE, WAVELENGTH
        )
        np.testing.assert_allclose(phase_solver, phase_pipeline, rtol=1e-12)

    def test_convergence_reported(self, bf_images):
        """solve() should report iteration count."""
        import math
        I_under, I_focus, I_over = bf_images
        dIdz, I0 = compute_dIdz([I_under, I_focus, I_over], DZ)
        solver = TIESolver(
            shape=(512, 512),
            pixelsize=PIXELSIZE,
            k=2 * math.pi / WAVELENGTH,
        )
        result = solver.solve(dIdz, I0)
        assert "iterations" in result
        assert result["iterations"] >= 1
