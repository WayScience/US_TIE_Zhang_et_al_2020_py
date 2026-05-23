.. _api_pipeline:

Pipeline  —  ``US_TIE_Zhang_et_al_2020_py.pipeline``
=====================================================

The pipeline module provides the primary user-facing entry points.
:func:`~US_TIE_Zhang_et_al_2020_py.retrieve_phase` is the recommended starting point for most
users: it accepts raw images and returns a phase image, handling the
``dIdz`` computation and solver construction internally.

.. currentmodule:: US_TIE_Zhang_et_al_2020_py

.. autofunction:: retrieve_phase

.. autofunction:: compute_dIdz
