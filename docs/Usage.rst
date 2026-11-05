Basic Usage
===========

QCut is a quantum circuit knitting package capable of efficiently partitioning
quantum circuits with wire and gate cuts using advanced LOCC and joint rotation decompositions
along with the standard local decompositions. It is designed to be compatible with Qiskit and should
be compatible with any Qiskit programmable backend but has been especially designed to be compatible
with IQM’s qpus and the Finnish Quantum Computing Infrastructure (`FiQCI <https://fiqci.fi/>`__).

The pages below follow one example from start to finish, in the order a run goes: cut the
circuit, transpile it for a device, then run it and reconstruct the results. Each page
builds on the variables of the ones before it, and the imports are at the start of
:doc:`CuttingCircuits`. :doc:`Primitives` covers running several cut circuits in one
call.

Click :download:`here <notebooks/QCutBasicUsage.ipynb>` to download the example as a
notebook.

.. toctree::
   :maxdepth: 2

   CuttingCircuits
   Transpilation
   RunningExperiments
   Primitives
   Hardware
