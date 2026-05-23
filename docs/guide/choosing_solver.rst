.. _choosing_solver:

Choosing the right solver backend
===================================

``US_TIE_Zhang_et_al_2020_py`` provides two **Poisson solver backends** that differ in how they
handle the image boundaries.  For most use cases the default FFT backend is
fine.  This page explains the trade-offs and helps you decide when to switch.


The short version
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - ``solver=``
     - Boundary condition
     - Use when
   * - ``'fft'`` *(default)*
     - Periodic — left/right and top/bottom edges are stitched together
     - Specimen well away from the image edge; fastest option
   * - ``'dct'``
     - Neumann — no flux crosses the image border
     - Specimen extends to (or near) the edge; ringing artefacts visible in
       FFT result

.. code-block:: python

   phase = retrieve_phase(
       images=[I_under, I_focus, I_over],
       dz=1e-6, pixelsize=162.5e-9, wavelength=532e-9,
       solver='dct',   # switch to Neumann boundary conditions
   )

What are "boundary conditions"?
---------------------------------

The TIE Poisson equation has infinitely many solutions — you can add any
constant to ``φ`` without changing the equation.  To get a unique answer, the
solver must also know what happens at the edges of the image.  Two conventions
are supported:

**Periodic (FFT backend)**

The image is treated as if it were one tile in an infinite repeating pattern:
the right edge is "stitched" to the left edge, and the top to the bottom.
This works perfectly for signals that genuinely repeat (e.g. periodic
structures), and gives a good approximation when the specimen occupies only the
central portion of the image, leaving a uniform background at the edges.

When the specimen extends to the border, the periodic assumption is violated:
the phase may jump discontinuously from one edge to the other, and the FFT
solver sees this "seam" as a high-frequency artefact that creates ringing
across the whole image.

**Neumann (DCT backend)**

The solver enforces ``∂φ/∂n = 0`` at every edge — no optical flux crosses the
image boundary.  This is physically correct: a finite specimen in the centre of
the field generates no flux at a sufficiently distant boundary.  Even when the
specimen extends to the edge, there is no artificial jump.  The DCT-II
transform implements this condition exactly.

When should I use the DCT backend?
------------------------------------

Switch to ``solver='dct'`` when:

- The specimen (or a phase feature) touches or approaches the image border.
- You see "halo" or ringing artefacts in the FFT result that extend inward
  from the edges.
- You are imaging a specimen that fills the entire field of view (e.g. a
  monolayer of cells with no clear background region).

Keep the FFT backend (default) when:

- The specimen is small relative to the field of view and surrounded by
  uniform background.
- Speed is critical and images are large (the DCT backend is slightly slower
  due to an extra transform per iteration).

Technical note: self-consistency
----------------------------------

The FFT and DCT backends each use their own **forward model** internally —
the operation that predicts ``∂I/∂z`` from a candidate phase.  This is
important for numerical correctness:

- **FFT backend**: spectral (Fourier) differentiation — exact for periodic
  signals.
- **DCT backend**: cell-centred finite-difference divergence — exact for
  Neumann-BC signals.

This means the two backends solve *slightly different* discrete TIE equations
and will give somewhat different answers for the same input images.  The
difference is small for smooth, slowly varying phases but can be significant
for high-spatial-frequency structures near image boundaries.  In practice, each
backend is accurate for the type of image it is designed for.
