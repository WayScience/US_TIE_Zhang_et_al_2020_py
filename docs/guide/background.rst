.. _background:

Background: the physics and the algorithm
==========================================

This page explains the ideas behind ``US_TIE_Zhang_et_al_2020_py`` in plain terms before
introducing the relevant mathematics.


Why can't I just image the phase directly?
------------------------------------------

Cameras — including those in microscopes — measure **intensity**: how many
photons hit each pixel per unit time.  Photons carry both amplitude
(proportional to the square root of intensity) and phase (a wave delay), but
most detectors are sensitive only to amplitude.  The phase, which encodes
crucial information about transparent specimens such as cells, is thrown away.

Specialised instruments (interferometers, Zernike phase contrast microscopes,
differential interference contrast) can reveal phase information, but they
require extra hardware, calibration, and expertise.

The Transport-of-Intensity Equation approach requires **none of these
modifications**.  It extracts the phase from ordinary brightfield images by
exploiting a fundamental property of light: when a beam of light propagates
through space, its intensity evolves in a way that is directly tied to the
phase gradient.

From defocus to phase: the intuition
--------------------------------------

Imagine shining a flashlight through a converging lens.  The lens bends the
light — adding a phase gradient — and this bending causes the beam to
concentrate and then diverge as you move the detector along the optical axis.
The same principle applies at the microscopic scale: any phase structure in the
specimen (e.g. the slightly higher refractive index of a cell membrane) acts
as a tiny lens, causing the local intensity to increase slightly on one side of
focus and decrease on the other.

If you take images at slightly **under-focused** and **over-focused** planes and
subtract them, the difference reveals the local intensity gradient along the
optical axis — and that gradient is directly related to the phase.

The Transport-of-Intensity Equation
-------------------------------------

The mathematical statement of this relationship is the
**Transport-of-Intensity Equation (TIE)**:

.. math::

   k \frac{\partial I}{\partial z} = -\nabla \cdot \bigl[I(\mathbf{r})\,
   \nabla \phi(\mathbf{r})\bigr]

where:

- *k* = 2π/λ is the **optical wavenumber** (λ is the wavelength)
- *∂I/∂z* is the **axial intensity derivative** — how brightness changes as you
  move through focus (approximated from your images)
- ∇ is the 2-D gradient operator in the image plane
- *I*(**r**) is the **in-focus intensity** image
- *φ*(**r**) is the **phase** we want to find

The TIE was derived in the 1980s and has been extensively validated.  It is
exact within the paraxial approximation (small propagation angles), which is
virtually always satisfied in optical microscopy.

Solving the TIE: why it is hard
---------------------------------

If the intensity *I* were constant everywhere (uniform illumination), the TIE
would reduce to a simple **Poisson equation**:

.. math::

   k \frac{\partial I}{\partial z} = -I_0 \nabla^2 \phi

which can be solved in a single step via Fourier transform.

In reality, *I* varies across the image — there are bright and dark regions.
The variable-coefficient equation is much harder to solve.  Naive approaches
either require computationally expensive iterative methods or break down when
*I* is near zero (dark regions).

The Universal Solution to the TIE
------------------------------------

The **US-TIE algorithm** (Zheng et al. 2020) solves this problem by an elegant
iterative scheme.  It replaces the spatially varying *I*(**r**) with the scalar
maximum *I*\ :sub:`max` = max(*I*), which turns the equation into a simple
Poisson problem at each iteration:

.. math::

   k \frac{\partial I^{(n)}}{\partial z} = -I_\text{max} \nabla^2 \phi^{(n)}

**Algorithm (one iteration)**:

1. **Poisson solve**: given the current residual ``dIdz_curr``, solve
   ``∇²φ_est = −k · dIdz_curr / I_max`` for ``φ_est``.
2. **Forward model**: compute the intensity derivative ``dIdz_est`` that
   ``φ_est`` would produce in combination with the *actual* ``I₀``:
   ``dIdz_est = −∇·(I₀ ∇φ_est) / k``.
3. **Update residual**: ``dIdz_curr ← dIdz_curr − dIdz_est``.
4. **Accumulate phase**: ``φ ← φ + φ_est``.

This iteration converges because the residual shrinks at each step — the
algorithm "explains" the intensity derivative progressively.  For uniform
illumination, it converges in a single iteration.

Boundary conditions
--------------------

The Poisson equation has infinitely many solutions — you can add any constant
to the phase and the equation is still satisfied.  Two additional choices must
be made:

1. **DC fix** (constant offset): the solver always returns a zero-mean phase.
   After reconstruction, use :func:`~US_TIE_Zhang_et_al_2020_py.remove_piston` to apply your own
   reference.

2. **Spatial boundary condition**: what happens at the image edges?  Two
   conventions are supported:

   - **Periodic (FFT backend)**: the image is treated as if it tiles
     infinitely.  Fast, and works well when the specimen is away from the
     edges.  Can produce ringing artefacts at the border.

   - **Neumann (DCT backend)**: no optical flux crosses the image border
     (∂φ/∂n = 0).  Physically correct for images of finite objects; eliminates
     boundary ringing.  Implemented via the Discrete Cosine Transform (DCT-II).

Reference
----------

   Zheng, G., Horstmeyer, R., & Yang, C. (2020).
   "On a universal solution to the transport-of-intensity equation."
   *Optics Letters*, 45(7), 1607–1610.
   `arXiv:1912.07371 <https://arxiv.org/abs/1912.07371>`_
