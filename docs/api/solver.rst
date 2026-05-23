.. _api_solver:

Solver  —  ``US_TIE_Zhang_et_al_2020_py.solver``
=================================================

:class:`TIESolver` is the iterative reconstruction engine.
It precomputes frequency grids and filter kernels once at construction time
and reuses them on every :meth:`~TIESolver.solve` call, making it efficient
for time-lapse or batch processing.

For a one-off reconstruction, :func:`~US_TIE_Zhang_et_al_2020_py.retrieve_phase`
wraps this class and is the recommended starting point.

.. currentmodule:: US_TIE_Zhang_et_al_2020_py

.. autoclass:: TIESolver
   :members:
   :special-members: __init__
   :show-inheritance:
