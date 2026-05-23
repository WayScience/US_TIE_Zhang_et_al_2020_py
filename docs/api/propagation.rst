.. _api_propagation:

Wave propagation  —  ``US_TIE_Zhang_et_al_2020_py.propagation``
===============================================================

:func:`~US_TIE_Zhang_et_al_2020_py.numerical_propagation` simulates how a complex optical field
evolves as it travels along the optical axis.  It is primarily used to
**generate synthetic test images** from a known phase field — for example, to
validate the phase retrieval pipeline or to study the effect of different
defocus steps.

For *recovering* phase from real images, use :func:`~US_TIE_Zhang_et_al_2020_py.retrieve_phase`
instead.

.. currentmodule:: US_TIE_Zhang_et_al_2020_py

.. autofunction:: numerical_propagation
