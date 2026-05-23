.. _api_poisson:

Poisson solvers  —  ``US_TIE_Zhang_et_al_2020_py.poisson``
===========================================================

These functions solve the 2-D Poisson equation ∇²φ = f, which is the core
mathematical step in TIE phase retrieval.
They differ only in their boundary condition assumption at the image edge.

For most users these are called automatically by
:class:`~US_TIE_Zhang_et_al_2020_py.TIESolver`.
They are exposed here for users who need direct access — for example, to
apply a Poisson solve to a custom right-hand side.

.. currentmodule:: US_TIE_Zhang_et_al_2020_py

.. autofunction:: poisson_dct

.. currentmodule:: US_TIE_Zhang_et_al_2020_py.poisson

.. autofunction:: poisson_fft
