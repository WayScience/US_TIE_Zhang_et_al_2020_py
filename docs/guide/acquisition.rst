.. _acquisition:

Image acquisition guide
=======================

This page describes how to acquire images for phase retrieval and how to
translate microscope settings into the parameters required by ``US_TIE_Zhang_et_al_2020_py``.

.. contents:: On this page
   :local:
   :depth: 2

Overview
--------

You need **2 or 3 conventional brightfield (transmitted light) images** of the
same field of view at slightly different focus positions.  No special hardware,
fluorescence, or sample preparation is required beyond what you would use for
standard brightfield imaging.

Three-image protocol (recommended)
-----------------------------------

Acquire images at three equally spaced axial positions:

+-------------+--------------------------------------+
| Image       | Position                             |
+=============+======================================+
| ``I_under`` | Under-focus: focal plane - ``dz``    |
+-------------+--------------------------------------+
| ``I_focus`` | In focus (focal plane)               |
+-------------+--------------------------------------+
| ``I_over``  | Over-focus: focal plane + ``dz``     |
+-------------+--------------------------------------+

The **axial step** ``dz`` is the distance between consecutive planes (not the
total span).  Pass the images in the order ``[I_under, I_focus, I_over]``:

.. code-block:: python

   phase = retrieve_phase(
       images=[I_under, I_focus, I_over],
       dz=1e-6,   # metres
       ...
   )

This scheme uses a *central difference* approximation for the intensity
derivative, which cancels first-order defocus artefacts and gives more accurate
results than the two-image alternative.

Two-image protocol
-------------------

If you only have an in-focus image and one defocused image, pass them as
``[I_focus, I_over]``:

.. code-block:: python

   phase = retrieve_phase(
       images=[I_focus, I_over],
       dz=1e-6,   # distance from I_focus to I_over
       ...
   )

This uses a *forward difference* estimate, which is slightly less accurate but
still robust for most applications.

Choosing the defocus step ``dz``
---------------------------------

The defocus step controls the sensitivity of the measurement.

- **Too small** (< 0.1 µm): the intensity difference is dominated by camera
  noise, and the phase reconstruction will be noisy.
- **Too large** (> ~10 µm for a 40× objective): the TIE linear approximation
  breaks down, and the intensity patterns become complex (multiple fringes).

A good starting point is 1–3 µm for a standard 40× 0.75 NA objective with
visible light.  For higher NA objectives the diffraction-limited depth of field
is shallower, so smaller ``dz`` values are appropriate.

**Rule of thumb**: choose ``dz`` so that the defocused image looks "slightly
blurry but not radically different" from the in-focus image.

Computing ``pixelsize``
------------------------

The ``pixelsize`` parameter is the size of one camera pixel *mapped to the
sample plane* (also called the effective pixel size or the pixel pitch at the
sample).

.. math::

   \text{pixelsize} = \frac{\text{camera pixel size (m)}}{\text{objective magnification}}

Examples:

+---------------------------+---------------+-----------------------------------------------+
| Camera pixel              | Objective     | ``pixelsize``                                 |
+===========================+===============+===============================================+
| 6.5 µm                    | 20×           | 6.5e-6 / 20 = 325 nm                          |
+---------------------------+---------------+-----------------------------------------------+
| 6.5 µm                    | 40×           | 6.5e-6 / 40 = 162.5 nm                        |
+---------------------------+---------------+-----------------------------------------------+
| 6.5 µm                    | 100×          | 6.5e-6 / 100 = 65 nm                          |
+---------------------------+---------------+-----------------------------------------------+
| 11 µm                     | 10×           | 11e-6 / 10 = 1.1 µm                           |
+---------------------------+---------------+-----------------------------------------------+

If your microscope has a tube lens with a magnification factor other than 1×,
include that in the denominator as well.

Illumination wavelength
------------------------

Use the central wavelength of your illumination source in metres.  For
broadband (white light) sources, use the dominant wavelength (typically around
550 nm for the eye-weighted peak of a white LED):

+----------------------------+--------------------+
| Source                     | ``wavelength``     |
+============================+====================+
| Blue LED (405 nm)          | 405e-9             |
+----------------------------+--------------------+
| Violet LED (450 nm)        | 450e-9             |
+----------------------------+--------------------+
| Green LED (532 nm)         | 532e-9             |
+----------------------------+--------------------+
| Red LED (640 nm)           | 640e-9             |
+----------------------------+--------------------+
| White LED (broad)          | ~550e-9            |
+----------------------------+--------------------+

Practical tips
---------------

- **Flat field correction**: divide each image by a background (no-specimen)
  image to remove illumination non-uniformities before passing to ``US_TIE_Zhang_et_al_2020_py``.
- **Avoid saturation**: make sure no pixels are clipped at the camera's maximum
  value.  Reduce exposure or illumination intensity if needed.
- **Stabilise between images**: motion between the three focus planes (e.g.
  sample drift or vibration) will degrade the result.  Use motorised focus
  control and acquire images quickly.
- **Consistent exposure**: use the same exposure time and illumination intensity
  for all three planes so that the intensity differences reflect only defocus,
  not exposure variation.
