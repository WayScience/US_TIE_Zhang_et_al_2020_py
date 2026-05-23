.. _api:

API reference
=============

This section documents every public function and class in ``US_TIE_Zhang_et_al_2020_py``.

.. toctree::
   :maxdepth: 2

   pipeline
   solver
   algorithms
   poisson
   tie
   propagation
   utils

Summary
-------

.. rubric:: End-to-end pipeline

.. autosummary::

   US_TIE_Zhang_et_al_2020_py.retrieve_phase
   US_TIE_Zhang_et_al_2020_py.compute_dIdz

.. rubric:: Iterative solvers

.. autosummary::

   US_TIE_Zhang_et_al_2020_py.TIESolver
   US_TIE_Zhang_et_al_2020_py.universal_solution
   US_TIE_Zhang_et_al_2020_py.fft_tie_solution

.. rubric:: Poisson solvers (standalone)

.. autosummary::

   US_TIE_Zhang_et_al_2020_py.poisson_dct
   US_TIE_Zhang_et_al_2020_py.poisson.poisson_fft

.. rubric:: Propagation

.. autosummary::

   US_TIE_Zhang_et_al_2020_py.numerical_propagation

.. rubric:: Utilities

.. autosummary::

   US_TIE_Zhang_et_al_2020_py.remove_piston
   US_TIE_Zhang_et_al_2020_py.rmse
