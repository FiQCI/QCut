QCut documentation
==================

QCut is a quantum circuit knitting package capable of efficiently partitioning
quantum circuits with wire and gate cuts using advanced LOCC and joint rotation decompositions
along with the standard local decompositions. It is designed to be compatible with Qiskit and should
be compatible with any Qiskit programmable backend but has been especially designed to be compatible
with IQM’s qpus and the Finnish Quantum Computing Infrastructure (`FiQCI <https://fiqci.fi/>`__).

QCut has been built at CSC - IT Center for Science (Finnish IT Center for Science)

.. toctree::
   :maxdepth: 1
   :caption: Guide

   GettingStarted
   Usage
   Options

.. toctree::
   :maxdepth: 1
   :caption: Further reading

   Notebooks
   Theory

.. toctree::
   :maxdepth: 1
   :caption: Reference

   QCut
   Changelog

.. toctree::
   :maxdepth: 1
   :caption: About

   Acknowledgement
   License

:doc:`GettingStarted` covers installation and a first run.

Why QCut
--------

``benchmarks/QCutVsAddon.ipynb`` in the repository compares QCut against IBM's
`qiskit-addon-cutting <https://github.com/Qiskit/qiskit-addon-cutting>`__ given the same
cuts: the gamma each achieves, how many subexperiments that comes to, how long they take
to generate, and what the two cut finders settle on under the same qubit budget. 
The benchmarks show QCut consistently matching or beating the Qiskit Cutting Addon.

.. code:: bash

   uv sync --group benchmark

Where one of QCut's joint decompositions applies, the same cuts cost less:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * -
     - QCut
     - addon
   * - three wires cut at the same point
     - gamma 15, 240 subexperiments
     - gamma 64, 1024
   * - three rotations crossing one split
     - gamma 7.16, 264 subexperiments
     - gamma 10.5, 432
   * - an ``rzz`` and an ``rxx`` on each of three crossing pairs
     - gamma 115
     - gamma 203

Where none applies, the QAOA layer at the sizes measured, the two agree exactly.
Turning consolidation, joint rotation cutting and communicating wire cuts off reproduces
the addon's gamma in every case the notebook measures, which is what says the difference
is those three and not something else.

