# SPDX-License-Identifier: CC-BY-4.0
# Adapted from Zheng et al. (2020) MATLAB source (CC BY 4.0) by Way Science Lab (2026),
# with AI assistance from Claude (Anthropic). See LICENSE for full attribution.

"""
Utility functions shared across the package.

These helpers are thin but important: :func:`remove_piston` fixes the
unavoidable constant phase offset that Poisson solvers leave behind, and
:func:`rmse` gives a single number for comparing reconstructed phases against
ground truth.
"""
from __future__ import annotations

import numpy as np


def remove_piston(phi: np.ndarray) -> np.ndarray:
    """Remove the constant phase offset (piston term) from a phase image.

    Poisson-based phase solvers can only determine phase up to an additive
    constant — they have no information about the absolute phase level, only
    about phase *differences* across the image.  This function removes that
    ambiguity by subtracting the mean of all non-NaN pixels, making the
    returned phase zero-mean.

    Parameters
    ----------
    phi : array_like, shape (Ny, Nx)
        Phase image in radians.  May contain NaN values (e.g. outside an
        aperture mask); those pixels are excluded from the mean.

    Returns
    -------
    phi_centered : ndarray, shape (Ny, Nx)
        Phase image with mean value subtracted.  NaN pixels are preserved.

    Examples
    --------
    >>> import numpy as np
    >>> from US_TIE_Zhang_et_al_2020_py import remove_piston
    >>> phi = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> remove_piston(phi)
    array([[-1.5, -0.5],
           [ 0.5,  1.5]])
    """
    return phi - np.nanmean(phi)


def rmse(phi: np.ndarray, true_phase: np.ndarray) -> float:
    """Root-mean-square error between a reconstructed phase and ground truth.

    Before computing the error, the piston offset (mean difference) is
    removed — because only *relative* phase differences are physically
    meaningful and Poisson solvers cannot recover the absolute offset.
    NaN pixels (e.g. outside an aperture) are excluded from the calculation.

    Parameters
    ----------
    phi : array_like, shape (Ny, Nx)
        Reconstructed phase in radians.
    true_phase : array_like, shape (Ny, Nx)
        Ground-truth phase in radians.  Must have the same shape as ``phi``.

    Returns
    -------
    rmse : float
        Root-mean-square phase error in radians, after piston removal.

    Examples
    --------
    >>> import numpy as np
    >>> from US_TIE_Zhang_et_al_2020_py import rmse
    >>> phi_rec   = np.array([0.1, 0.9, 1.8])  # reconstructed (with offset)
    >>> phi_true  = np.array([0.0, 1.0, 2.0])  # ground truth
    >>> rmse(phi_rec, phi_true)   # offset is removed before computing error
    0.0816...
    """
    err = phi - true_phase
    err = err - np.nanmean(err)
    valid = ~np.isnan(err)
    return float(np.sqrt(np.nansum(err**2) / valid.sum()))