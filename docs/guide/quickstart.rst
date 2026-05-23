.. _quickstart:

Quick start
===========

.. contents:: On this page
   :local:
   :depth: 2

Installation
------------

.. code-block:: bash

   pip install US_TIE_Zhang_et_al_2020_py

For development (editable install with test dependencies):

.. code-block:: bash

   git clone https://github.com/wayscience/US_TIE_Zhang_et_al_2020_py
   cd US_TIE_Zhang_et_al_2020_py
   pip install -e ".[dev]"

Your first phase image
-----------------------

The main entry point is :func:`~US_TIE_Zhang_et_al_2020_py.retrieve_phase`.  Pass your images, a
few physical parameters, and get the phase in radians.

**Three-image protocol (recommended)**

Take one in-focus image and two images at equal but opposite defocus steps:

.. code-block:: python

   import tifffile
   from US_TIE_Zhang_et_al_2020_py import retrieve_phase

   I_under = tifffile.imread("z_minus1um.tif").astype(float)
   I_focus = tifffile.imread("z_focus.tif").astype(float)
   I_over  = tifffile.imread("z_plus1um.tif").astype(float)

   phase = retrieve_phase(
       images=[I_under, I_focus, I_over],  # order matters!
       dz=1e-6,            # 1 µm between consecutive planes
       pixelsize=162.5e-9, # camera pixel / objective magnification
       wavelength=532e-9,  # illumination wavelength in metres
   )

The returned ``phase`` is a 2-D NumPy array in **radians**.

**Two-image protocol**

If you only have one in-focus and one defocused image:

.. code-block:: python

   phase = retrieve_phase(
       images=[I_focus, I_over],   # in-focus first, defocused second
       dz=1e-6,
       pixelsize=162.5e-9,
       wavelength=532e-9,
   )

.. note::

   The three-image protocol is more accurate because it uses a *central
   difference* for the defocus derivative.  The two-image protocol uses a
   *forward difference*, which introduces a small first-order error.

Computing physical quantities
------------------------------

The phase image ``φ(r)`` is related to the optical path length (OPL):

.. math::

   \phi(\mathbf{r}) = \frac{2\pi}{\lambda} \cdot \text{OPL}(\mathbf{r})

For a specimen of thickness *h* and refractive index *n* immersed in medium
of index *n*\ :sub:`m`:

.. math::

   \text{OPL}(\mathbf{r}) = (n - n_m) \cdot h(\mathbf{r})

So the **optical path length map** in metres is:

.. code-block:: python

   wavelength = 532e-9   # metres
   OPL = phase * wavelength / (2 * np.pi)   # metres

And the **dry mass** per pixel of a cell (using the specific refractive
increment α ≈ 0.18 mL/g for protein):

.. code-block:: python

   alpha = 0.18e-6         # m³/g (specific refractive increment)
   pixelsize = 162.5e-9    # metres
   dry_mass_per_pixel = OPL * pixelsize**2 / alpha   # grams
   total_dry_mass = dry_mass_per_pixel.sum()          # grams for whole image

Visualising the result
-----------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from US_TIE_Zhang_et_al_2020_py import remove_piston

   # Remove the constant phase offset (Poisson solvers cannot determine it)
   phase_centred = remove_piston(phase)

   fig, ax = plt.subplots()
   im = ax.imshow(phase_centred, cmap='RdBu_r')
   ax.set_title("Quantitative phase (rad)")
   plt.colorbar(im, ax=ax, label="Phase (rad)")
   plt.show()

.. seealso::

   - :doc:`acquisition` — detailed advice on acquiring images
   - :doc:`choosing_solver` — when to switch from the default FFT to the DCT solver
   - :func:`~US_TIE_Zhang_et_al_2020_py.retrieve_phase` — full parameter reference
