Cutting circuits
================

Import needed packages
----------------------

.. code:: python

   import numpy as np
   import QCut as ck
   from QCut import cut, cutGate, CutOptions, find_cuts
   from qiskit import QuantumCircuit, transpile
   from qiskit.circuit.library import CXGate
   from qiskit.quantum_info import SparsePauliOp
   from qiskit_aer import AerSimulator
   from qiskit.primitives import StatevectorEstimator as Estimator, BackendEstimatorV2 as BackendEstimator
   from iqm.qiskit_iqm import IQMFakeAdonis

Define a QuantumCircuit just like in Qiskit
-------------------------------------------

.. code:: python

   circuit = QuantumCircuit(5)

   for qubit in range(5):
       circuit.ry(0.4 + 0.3*qubit, qubit)
   circuit.cx(0,1)
   circuit.cx(1,2)
   circuit.cx(2,3)
   circuit.cx(3,4)

   circuit.draw("mpl")

.. image:: _static/images/circ1.png

Insert cuts to denote where to cut the circuit
----------------------------------------------

Note that here we don’t insert any measurements. Measurements will be
automatically handled by QCut.

.. code:: python

   marked_circuit = QuantumCircuit(5)

   for qubit in range(5):
       marked_circuit.ry(0.4 + 0.3*qubit, qubit)
   marked_circuit.cx(0,1)
   marked_circuit.append(**cutGate(CXGate(), 1, 2))
   marked_circuit.cx(2,3)
   marked_circuit.append(cut(), [3])
   marked_circuit.cx(3,4)

   marked_circuit.decompose(gates_to_decompose=["CutGate"]).draw("mpl")

.. image:: _static/images/circ2.png

:code:`cutGate()` marks a gate cut and :code:`cut()` a wire cut. Any two-qubit gate can
be cut, with the decomposition derived from the gate's KAK coordinates, so the CX above
is a single cut rather than a CZ with fix-up gates around it. See
:doc:`gate cuts <GateCuts>` and :doc:`wire cuts <WireCuts>` for more on
placing them, and :doc:`Theory` for where the decompositions come from.

Extract cut locations and split into independent subcircuits
------------------------------------------------------------

.. code:: python

   cut_circuit = ck.get_locations_and_subcircuits(marked_circuit)

Now we can draw our subcircuits.

.. code:: python

   cut_circuit.subcircuits[0].draw("mpl")

.. image:: _static/images/circ3.png

.. code:: python

   cut_circuit.subcircuits[1].draw("mpl")

.. image:: _static/images/circ4.png

.. code:: python

   cut_circuit.subcircuits[2].draw("mpl")

.. image:: _static/images/circ11.png

Define observables and generate experiment circuits
---------------------------------------------------

Observables are given the way Qiskit's estimator takes them: a Pauli label, a
:code:`Pauli`, a :code:`SparsePauliOp`, a :code:`SparseObservable` (qiskit 2.1+), a
:code:`{label: coefficient}` mapping, or any nested sequence of those. The expectation
values come back shaped like what you pass, so a list of four observables gives four
values, in order.

.. code:: python

   observables = ["IIIIZ", "IIIZI", "IIZII", "IIIZZ"]

   cut_experiment = ck.get_experiment_circuits(cut_circuit, observables)

   print(cut_experiment.num_groups)

``48``

:code:`get_experiment_circuits()` does not modify the :code:`CutCircuit` it is given, so
the same split can be reused for several observable sets. Both :code:`CutCircuit` and
:code:`CutExperiment` implement the :code:`assign_parameters()` function of
:code:`Qiskit.QuantumCircuit`.

What a cut costs
----------------

.. code:: python

   print(cut_circuit.gamma, cut_circuit.optimal_gamma)

``12.0 9.0``

:code:`gamma` is the sampling overhead of this split and :code:`optimal_gamma` the least
those same cuts could cost with every decomposition available. Shot cost goes as
:code:`gamma` squared. Both are closed form, so a plan can be costed before any
experiment circuits are built, and :code:`CutExperiment` carries both forward.

The gap here is the lone wire cut: on its own it does not communicate under the default
:code:`wire_cut_communication="auto"`, so it costs 4 rather than 3. Passing
:code:`"always"` closes it and this split reaches 9.

Cheaper decompositions
----------------------

Cuts are not decomposed one at a time where a cheaper joint decomposition exists. All
three of the below are on by default and can be turned off through :doc:`Options`.

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * -
     - default
     - turned off
   * - A run of gates on one qubit pair costs a single cut
     - 1 cut, gamma 1.59, 6 subexperiments
     - 2 cuts, gamma 9, 36
   * - Parallel single-axis rotations share one decomposition
       (:doc:`derivation <theory/Joint_rotation_derivation>`)
     - gamma 5.50, 30 subexperiments
     - gamma 6.77, 36
   * - A block of parallel wire cuts exchanges its measured outcome
       (:doc:`derivation <theory/LOCC_wire_derivation>`)
     - gamma 7, 28 subexperiments
     - gamma 16, 64

A separate circuit, to show the last of those on its own. Two parallel wire cuts cost
:code:`2**(n+1) - 1` rather than :code:`4**n`:

.. code:: python

   pair = QuantumCircuit(4)
   for qubit in range(4):
       pair.ry(0.4 + 0.2 * qubit, qubit)
   pair.cx(0, 1)
   pair.cx(0, 2)
   pair.append(cut(), [1])
   pair.append(cut(), [2])
   pair.cx(1, 2)
   pair.cx(2, 3)

   block = ck.get_locations_and_subcircuits(pair)
   local = ck.get_locations_and_subcircuits(
       pair, options=CutOptions(wire_cut_communication="never")
   )

   print(block.gamma, local.gamma, local.optimal_gamma)

``7.0 16.0 7.0``

Those cuts run in waves, since one side has to be measured before the other can prepare
what it measured. :code:`submit_experiments()` handles that itself.

Options
-------

:code:`CutOptions` is collected once and carried through the run. It covers the three
decompositions above, the expansion strategy, sampling and the cut finder. Above 1000
groups the experiment is sampled from the quasiprobability distribution rather than
enumerated. Every option and its default is listed under :doc:`Options`.

Automatic cuts
--------------

QCut comes with functionality for automatically finding good cut locations that can place both wire and gate cuts.

.. code:: python

   options = CutOptions(
      finder_num_partitions=3,
      finder_cut_mode="both",
   )

   found = find_cuts(circuit, options=options)

   print(len(found.cut_locations), found.gamma)

``2 9.0``

Here the finder reaches the :code:`optimal_gamma` the hand-placed cuts above did not. See
:doc:`automatic cuts <AutomaticCuts>` for the finder's own options and how it
chooses.

In depth
--------

Each kind of cut has a page of its own, with its own setup.

.. toctree::
   :maxdepth: 1

   WireCuts
   GateCuts
   AutomaticCuts
