.. _api_algorithms:

Algorithms  —  ``US_TIE_Zhang_et_al_2020_py.algorithms``
=========================================================

Two TIE phase retrieval algorithms in a functional (stateless) API.
Each function constructs a :class:`~US_TIE_Zhang_et_al_2020_py.TIESolver`
internally on every call.
For repeated calls on images of the same size, construct
:class:`~US_TIE_Zhang_et_al_2020_py.TIESolver` once and call
:meth:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver.solve` directly to avoid
recomputing frequency grids.

.. currentmodule:: US_TIE_Zhang_et_al_2020_py

.. autofunction:: universal_solution

.. autofunction:: fft_tie_solution
