# SPDX-License-Identifier: CC-BY-4.0
# Adapted from Zheng et al. (2020) MATLAB source (CC BY 4.0) by Way Science Lab (2026),
# with AI assistance from Claude (Anthropic). See LICENSE for full attribution.

"""
End-to-end phase retrieval pipeline.

:func:`retrieve_phase` is the primary user-facing entry point.  It accepts
2 or 3 brightfield images, a defocus step, pixel size, and wavelength, and
returns a quantitative phase image in radians.

Acquisition protocols
---------------------
**Two-image** (forward difference)::

    images = [I_focus, I_defocused]      # I_focus is in focus; I_defocused at +dz
    dIdz  ≈ (I_defocused - I_focus) / dz

**Three-image** (central difference — recommended)::

    images = [I_under, I_focus, I_over]  # symmetric about focus
    dIdz  ≈ (I_over - I_under) / (2·dz)

The three-image scheme is more accurate because the central difference
cancels first-order defocus artefacts that the forward difference retains.

See Also
--------
:class:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver` : reusable solver for batch processing.
:func:`~US_TIE_Zhang_et_al_2020_py.solvers.universal_solution` : lower-level functional wrapper.
"""
from __future__ import annotations

import numpy as np

from .solver import TIESolver


def compute_dIdz(
    images: list[np.ndarray] | np.ndarray,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the axial intensity derivative ``∂I/∂z`` from through-focus images.

    This is the first step in TIE phase retrieval.  The output ``dIdz``
    quantifies how the image brightness changes as you move through focus, which
    is what the TIE equation relates to the phase.

    Parameters
    ----------
    images : list of 2 or 3 arrays, or ndarray of shape (2|3, Ny, Nx)
        Through-focus intensity images, in axial order (lowest z first):

        * **2 images** ``[I_focus, I_defocused]`` — forward difference.
          ``I_focus`` is the in-focus frame; ``I_defocused`` is taken at
          axial distance ``+dz``.
        * **3 images** ``[I_under, I_focus, I_over]`` — central difference
          (recommended).  ``I_focus`` is the in-focus frame; ``I_under`` and
          ``I_over`` are equidistant at ``−dz`` and ``+dz`` respectively.

    dz : float
        Axial step between consecutive image planes in metres.  Must be
        positive.  For the 2-image case this is the distance between the two
        images; for the 3-image case it is the distance between consecutive
        planes (half the total axial span).

    Returns
    -------
    dIdz : ndarray, shape (Ny, Nx)
        Axial intensity derivative ``∂I/∂z`` in intensity units per metre.
    I0 : ndarray, shape (Ny, Nx)
        In-focus intensity image (the middle frame for 3 images, the first
        frame for 2 images).

    Raises
    ------
    ValueError
        If fewer than 2 or more than 3 images are supplied.

    Examples
    --------
    Three-image central difference (more accurate):

    >>> dIdz, I0 = compute_dIdz([I_under, I_focus, I_over], dz=1e-6)

    Two-image forward difference:

    >>> dIdz, I0 = compute_dIdz([I_focus, I_over], dz=1e-6)

    See Also
    --------
    retrieve_phase : Full pipeline that calls this function internally.
    """
    imgs = np.asarray(images, dtype=float)
    if imgs.ndim == 2:
        raise ValueError(
            "Pass at least 2 images as a list [I0, Iz] or a (2, Ny, Nx) array."
        )

    n = imgs.shape[0]
    if n == 2:
        I0 = imgs[0]
        dIdz = (imgs[1] - imgs[0]) / float(dz)
    elif n == 3:
        I0 = imgs[1]
        dIdz = (imgs[2] - imgs[0]) / (2.0 * float(dz))
    else:
        raise ValueError(
            f"Expected 2 or 3 images (got {n}).  "
            "For 2 images: [I0, I_defocused].  "
            "For 3 images: [I_under, I0, I_over]."
        )
    return dIdz, I0


def retrieve_phase(
    images: list[np.ndarray] | np.ndarray,
    dz: float,
    pixelsize: float,
    wavelength: float,
    reg: float = np.finfo(float).eps,
    max_iter: int = 500,
    tol: float = 1e-3,
    fft_workers: int = -1,
    solver: str = "fft",
) -> np.ndarray:
    """
    End-to-end quantitative phase retrieval from brightfield images.

    Accepts 2 or 3 through-focus intensity images and returns the recovered
    phase in radians using the Universal Solution to the Transport-of-Intensity
    Equation (US-TIE, Zheng et al. 2020).

    Parameters
    ----------
    images : list of 2 or 3 (Ny, Nx) arrays, or a (2|3, Ny, Nx) array.
        **2 images** ``[I0, I_defocused]``
            Forward-difference derivative.  Less accurate but requires only
            one extra image.  ``I0`` must be the in-focus frame.
        **3 images** ``[I_under, I0, I_over]``
            Central-difference derivative (recommended).  ``I0`` is the
            in-focus frame; ``I_under`` / ``I_over`` are equidistant.
    dz : float
        Defocus step in metres.  For 2 images this is the distance between
        ``I0`` and ``I_defocused``; for 3 images it is the step between
        consecutive images.
    pixelsize : float
        Camera pixel size mapped to the sample plane, in metres.
        Example: 6.5 µm camera pixel with 40× objective → 6.5e-6 / 40 = 162.5 nm.
    wavelength : float
        Central wavelength of the illumination in metres.
        Example: 532 nm → 532e-9.
    reg : float, optional
        Tikhonov regularisation parameter.  The default (≈ machine epsilon)
        gives essentially no regularisation while preventing DC division by
        zero.  Increase (e.g. ``1e-3``) to suppress low-frequency noise in
        images with a non-uniform background.
    max_iter : int, optional
        Maximum number of US-TIE iterations.  The algorithm usually converges
        in 10–50 iterations; 500 is a conservative ceiling.
    tol : float, optional
        Convergence threshold.  Iterations stop when the maximum absolute
        intensity-derivative residual falls below ``tol`` × its initial value.
    fft_workers : int, optional
        Number of CPU threads used for FFT computation.  ``-1`` (default) uses
        all available cores.  Set to ``1`` for reproducible single-threaded
        benchmarks.
    solver : ``'fft'`` or ``'dct'``, optional
        Poisson solver backend.

        ``'fft'`` *(default)*
            Spectral solver assuming **periodic** boundary conditions.
            Fastest option; may produce boundary artefacts when objects
            extend to the image edge.

        ``'dct'``
            DCT-II solver enforcing **Neumann** boundary conditions
            (∂u/∂n = 0).  Physically correct for images of finite objects;
            recommended when objects are near the image border or when
            the FFT result shows ringing artefacts.

    Returns
    -------
    phase : (Ny, Nx) float array
        Recovered quantitative phase in radians.

    Examples
    --------
    Three-image call (recommended):

    >>> import numpy as np
    >>> from US_TIE_Zhang_et_al_2020_py import retrieve_phase
    >>> # Load images — any 2-D float array works (tifffile, PIL, etc.)
    >>> # I_under = tifffile.imread("z_minus.tif").astype(float)
    >>> # I_focus = tifffile.imread("z_focus.tif").astype(float)
    >>> # I_over  = tifffile.imread("z_plus.tif").astype(float)
    >>> phase = retrieve_phase(
    ...     images=[I_under, I_focus, I_over],
    ...     dz=1e-6,            # 1 µm between consecutive planes
    ...     pixelsize=162.5e-9, # 6.5 µm camera pixel / 40× objective
    ...     wavelength=532e-9,  # 532 nm green LED
    ... )

    Two-image call:

    >>> phase = retrieve_phase(
    ...     images=[I_focus, I_over],
    ...     dz=1e-6,
    ...     pixelsize=162.5e-9,
    ...     wavelength=532e-9,
    ... )

    Using the DCT (Neumann) backend to reduce boundary ringing:

    >>> phase = retrieve_phase(
    ...     images=[I_under, I_focus, I_over],
    ...     dz=1e-6, pixelsize=162.5e-9, wavelength=532e-9,
    ...     solver='dct',
    ... )

    See Also
    --------
    compute_dIdz : Compute the axial intensity derivative separately.
    TIESolver : Reusable solver for batch / time-lapse processing.
    remove_piston : Remove the constant phase offset after reconstruction.
    """
    dIdz, I0 = compute_dIdz(images, dz)
    k = 2.0 * np.pi / float(wavelength)
    _solver = TIESolver(
        shape=I0.shape,
        pixelsize=float(pixelsize),
        k=k,
        reg=float(reg),
        backend=str(solver),
        fft_workers=int(fft_workers),
    )
    return _solver.solve(dIdz, I0, max_iter=max_iter, tol=tol)["phase"]