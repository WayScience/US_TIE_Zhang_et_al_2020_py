# SPDX-License-Identifier: CC-BY-4.0
# Adapted from Zheng et al. (2020) MATLAB source (CC BY 4.0) by Way Science Lab (2026),
# with AI assistance from Claude (Anthropic). See LICENSE for full attribution.

"""
Numerical wave-field propagation methods.

When coherent light travels through space, its amplitude and phase evolve
according to the wave equation.  This module implements three standard
numerical approximations for propagating a complex optical field by a short
axial distance *dz*:

Angular Spectrum (default)
    The most accurate of the three methods.  Decomposes the field into plane
    waves (via FFT), advances each wave by its exact phase delay, and
    recombines.  Valid for any propagation distance, but assumes the field is
    band-limited (no spatial frequencies beyond the Nyquist limit).

Fresnel
    A paraxial approximation of the Angular Spectrum.  Uses a quadratic-phase
    transfer function instead of the exact square root.  Slightly faster,
    valid when all propagation angles are small (specimen features much larger
    than the wavelength).

TIE (linear approximation)
    Uses the Transport-of-Intensity Equation as a first-order approximation
    to predict how the intensity pattern changes with defocus.  Very fast and
    adequate for small defocus steps, but only computes the intensity (the
    complex field and phase at *z + dz* are not available).

These methods are provided primarily to let users **simulate** through-focus
image stacks for testing or validation — for example, to generate synthetic
``I_under`` / ``I0`` / ``I_over`` triplets from a known phase field.  For
*recovering* phase from real images, see :func:`US_TIE_Zhang_et_al_2020_py.retrieve_phase`.
"""
from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from .tie import tie_forward


def numerical_propagation(
    U0: np.ndarray,
    dz: float,
    pixelsize: float,
    wavelength: float,
    method: str = "angular_spectrum",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a complex optical field by axial distance *dz*.

    Starting from a complex field ``U0`` (amplitude × phase) at the focal
    plane, this function computes the field ``Uz``, intensity ``Iz``, and
    phase ``Phiz`` at a plane displaced by ``dz`` along the optical axis.

    Parameters
    ----------
    U0 : array_like, shape (Ny, Nx), complex
        Complex optical field at the starting plane.  The intensity is
        ``|U0|²`` and the phase is ``angle(U0)``.
    dz : float
        Propagation distance in metres.  Positive values propagate forward
        (away from the light source); negative values propagate backward.
    pixelsize : float
        Camera pixel size mapped to the sample plane, in metres.
        Example: a 40× objective with a 6.5 µm camera pixel gives
        6.5e-6 / 40 = 162.5 nm effective pixel size.
    wavelength : float
        Central wavelength of the illumination, in metres.
        Example: 532e-9 for green light.
    method : {'angular_spectrum', 'fresnel', 'tie'}, optional
        Propagation model to use.

        ``'angular_spectrum'`` *(default)*
            Exact within the scalar diffraction approximation.  Each spatial
            frequency component is phase-shifted by its correct propagation
            delay.  Evanescent waves (spatial frequencies beyond ``1/λ``) are
            suppressed.

        ``'fresnel'``
            Paraxial approximation.  Uses a quadratic transfer function
            ``exp(ikz(1 − λ²(u² + v²)/2))``.  Valid when all diffraction
            angles are small.

        ``'tie'``
            First-order approximation: ``I(z+dz) ≈ I(z) + dz·∂I/∂z``, where
            ``∂I/∂z`` is computed via the TIE forward model.  Only the
            intensity ``Iz`` is meaningful; ``Uz`` and ``Phiz`` are returned
            as NaN arrays.

    Returns
    -------
    Uz : ndarray, shape (Ny, Nx), complex
        Complex field at the propagated plane.  For ``method='tie'``, this
        is a NaN-filled array of the same shape and dtype as ``U0``.
    Iz : ndarray, shape (Ny, Nx), float
        Intensity at the propagated plane (``|Uz|²``).
    Phiz : ndarray, shape (Ny, Nx), float
        Phase at the propagated plane (``angle(Uz)``).  For ``method='tie'``,
        this is a NaN-filled array.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported values.

    Examples
    --------
    Simulate defocused intensity from a phase object:

    >>> import numpy as np
    >>> from US_TIE_Zhang_et_al_2020_py import numerical_propagation
    >>> N = 128
    >>> phi_true = np.zeros((N, N))
    >>> phi_true[40:80, 40:80] = 1.0      # square phase object (1 rad)
    >>> U0 = np.exp(1j * phi_true)         # unit-amplitude field
    >>> Uz, Iz, Phiz = numerical_propagation(
    ...     U0, dz=5e-6, pixelsize=100e-9, wavelength=532e-9
    ... )

    Notes
    -----
    The method string is normalised (lower-cased, spaces and hyphens replaced
    with underscores), so ``'Angular Spectrum'``, ``'angular-spectrum'``, and
    ``'angular_spectrum'`` are all accepted.

    See Also
    --------
    retrieve_phase : Recover phase from through-focus intensity images.
    """
    ny, nx = U0.shape
    k = 2 * np.pi / wavelength

    # Centered frequency grid — matches MATLAB convention of building the
    # transfer function on fftshift-ed frequencies.
    fx = np.fft.fftfreq(nx, d=pixelsize)
    fy = np.fft.fftfreq(ny, d=pixelsize)
    U_fft, V_fft = np.meshgrid(fx, fy)
    U_c = fftshift(U_fft)
    V_c = fftshift(V_fft)

    method = method.lower().replace(" ", "_").replace("-", "_")

    if method == "angular_spectrum":
        sq = 1.0 - (wavelength * U_c) ** 2 - (wavelength * V_c) ** 2
        H = np.exp(1j * k * dz * np.sqrt(np.abs(sq)))
        H[sq < 0] = 0.0          # suppress evanescent waves
        FU0 = fftshift(fft2(U0))
        Uz = ifft2(ifftshift(FU0 * H))

    elif method == "fresnel":
        H = np.exp(
            1j * k * dz
            * (1.0 - ((wavelength * U_c) ** 2 + (wavelength * V_c) ** 2) / 2.0)
        )
        FU0 = fftshift(fft2(U0))
        Uz = ifft2(ifftshift(FU0 * H))

    elif method == "tie":
        I0 = np.abs(U0) ** 2
        phi = np.angle(U0)
        dIdz = tie_forward(phi, I0, pixelsize, k)
        Iz = I0 + dz * dIdz
        nan_field = np.full_like(U0, np.nan, dtype=complex)
        return nan_field, Iz, np.full(U0.shape, np.nan)

    else:
        raise ValueError(
            f"Unknown method '{method}'.  "
            "Choose from: 'angular_spectrum', 'fresnel', 'tie'."
        )

    Iz = np.abs(Uz) ** 2
    Phiz = np.angle(Uz)
    return Uz, Iz, Phiz