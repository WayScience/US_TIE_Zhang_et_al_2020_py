.. _api_tie:

TIE operators  —  ``US_TIE_Zhang_et_al_2020_py.tie``
=====================================================

The TIE forward model and the single-step inverse operator.

:func:`tie_forward` predicts what the through-focus intensity change should
look like given a known phase — the "forward direction": phase → observable.

:func:`tie_max_solver` does one step in the *inverse* direction: given an
observed intensity change, it estimates the phase that produced it using the
maximum-intensity Poisson approximation of Zheng et al. (2020).

Both functions are primarily used internally by
:class:`~US_TIE_Zhang_et_al_2020_py.TIESolver` and are exposed here for
testing, validation, and custom pipelines.

.. currentmodule:: US_TIE_Zhang_et_al_2020_py.tie

.. autofunction:: tie_forward

.. autofunction:: tie_max_solver
