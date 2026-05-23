.. _batch:

Batch processing and performance
=================================

This page covers how to process many images efficiently: time-lapse movies,
multi-condition experiments, and large image stacks.

.. contents:: On this page
   :local:
   :depth: 2

Reusing the solver (fastest approach)
---------------------------------------

Every call to :func:`~US_TIE_Zhang_et_al_2020_py.retrieve_phase` constructs a fresh
:class:`~US_TIE_Zhang_et_al_2020_py.TIESolver` internally, which allocates and computes the
frequency grids and Poisson filter.  For a single image this overhead is
negligible.  For a time-lapse with hundreds of frames it adds up.

The solution is to construct a :class:`~US_TIE_Zhang_et_al_2020_py.TIESolver` **once** and call
:meth:`~US_TIE_Zhang_et_al_2020_py.solver.TIESolver.solve` in a loop:

.. code-block:: python

   import numpy as np
   from US_TIE_Zhang_et_al_2020_py import TIESolver, compute_dIdz

   # Build the solver once for this image size and wavelength
   solver = TIESolver(
       shape=(512, 512),
       pixelsize=162.5e-9,
       k=2 * np.pi / 532e-9,
       backend='fft',     # or 'dct'
   )

   # Process each time point
   phases = []
   for I_under, I_focus, I_over in frame_generator():
       dIdz, I0 = compute_dIdz([I_under, I_focus, I_over], dz=1e-6)
       result = solver.solve(dIdz, I0)
       phases.append(result['phase'])

Multithreaded FFT
------------------

``US_TIE_Zhang_et_al_2020_py`` uses ``scipy.fft`` for all FFT and DCT operations.  By default
(``fft_workers=-1``), SciPy uses **all available CPU cores** via
`pyfftw <https://github.com/pyFFTW/pyFFTW>`_ or its own multithreaded backend.
No extra configuration is needed.

To limit CPU usage or ensure reproducibility:

.. code-block:: python

   solver = TIESolver(
       shape=(512, 512),
       pixelsize=162.5e-9,
       k=2 * np.pi / 532e-9,
       fft_workers=4,   # use exactly 4 threads
   )

Expected performance
---------------------

For a 512 × 512 image on a modern 8-core workstation:

- **Per iteration**: ≈ 2–5 ms (FFT backend), ≈ 3–6 ms (DCT backend)
- **Typical convergence**: 10–50 iterations → 20–250 ms total
- **Construction cost**: ≈ 1–2 ms (amortised over all frames)

For a 1024 × 1024 image the per-iteration cost roughly quadruples.

Processing in parallel with ``multiprocessing``
------------------------------------------------

For embarrassingly parallel workloads (independent fields of view, independent
time points), use Python's ``multiprocessing`` module.  Each worker process
should construct its own :class:`~US_TIE_Zhang_et_al_2020_py.TIESolver` (they cannot be pickled
for inter-process sharing):

.. code-block:: python

   from multiprocessing import Pool
   from US_TIE_Zhang_et_al_2020_py import retrieve_phase

   def process_fov(args):
       images, dz, pixelsize, wavelength = args
       return retrieve_phase(images, dz, pixelsize, wavelength)

   with Pool(processes=4) as pool:
       all_phases = pool.map(process_fov, fov_list)

.. note::

   When using ``multiprocessing``, set ``fft_workers=1`` inside the worker
   functions to avoid oversubscribing the CPU (each process would otherwise
   try to use all cores):

   .. code-block:: python

       def worker(args):
           return retrieve_phase(*args, fft_workers=1)

Tracking convergence
---------------------

The ``solve`` method returns a dictionary that includes per-iteration
diagnostics when you pass a ground-truth phase (for validation) or when you
want to monitor convergence:

.. code-block:: python

   result = solver.solve(dIdz, I0, max_iter=500, tol=1e-3)
   print(f"Converged in {result['iterations']} iterations")

If you have a ground-truth phase (e.g. from a simulation), pass it as
``true_phase`` to track the RMSE over iterations:

.. code-block:: python

   result = solver.solve(dIdz, I0, true_phase=phi_true)
   print(f"Final RMSE: {result['rmse'][-1]:.4f} rad")

   # Plot convergence
   import matplotlib.pyplot as plt
   plt.semilogy(result['rmse'])
   plt.xlabel("Iteration")
   plt.ylabel("RMSE (rad)")
   plt.title("US-TIE convergence")
   plt.show()
