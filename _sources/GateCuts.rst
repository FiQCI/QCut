Gate cuts
=========

Gate cuts can be used to cut two-qubit gates instead of cutting wires. This is done by inserting special gate cut instructions into the circuit.

**Any** two-qubit gate can be cut. CZ, SWAP and iSWAP use hand-derived decompositions. For every other gate a quasiprobability decomposition is generated from the gate's own matrix via its KAK decomposition (see :doc:`Theory`). Gates on more than two qubits are still transpiled down first, which turns them into several cuts.

Generating the decomposition rather than transpiling to CZ matters most for parametrised gates, whose sampling overhead depends on the angle. ``rzz(0.3)`` costs :math:`\gamma = 1 + 2|\sin\theta| \approx 1.59` over 6 subexperiments as itself, against :math:`\gamma = 9` over 36 subexperiments as two CZ cuts. The shot count scales as :math:`\gamma^2`, so that is a factor of roughly 32.

.. code:: python

   from qiskit.circuit.library import CXGate
   from qiskit import QuantumCircuit
   from QCut import cutGate

   cut_circuit = QuantumCircuit(3)
   cut_circuit.h(0)
   cut_circuit.append(**cutGate(CXGate(), 0, 1))
   cut_circuit.cx(1,2)

   cut_circuit.decompose(["CutGate"]).draw("mpl")

After this the circuit can be processed as usual with QCut (take a look at the Usage documentation for more details).

Cutting parallel rotations together
-----------------------------------

Cutting two gates separately costs the product of their overheads. Rotation gates that
are *parallel*, meaning they act on disjoint qubits and can be scheduled in the same
moment, have a joint decomposition that costs strictly less than that product, in both
sampling overhead and circuit count.

It applies to any gate locally equivalent to a single-axis rotation, so ``rzz``, ``rxx``
and ``ryy`` all qualify, as do the gates that differ from them only by single-qubit
operations. For angles :math:`\theta_s` the joint overhead is
:math:`\gamma = 2\prod_s(1 + |\sin\theta_s|) - 1` against
:math:`\prod_s(1 + 2|\sin\theta_s|)` for cutting them one at a time. The derivation is
in :doc:`Theory`.

.. code:: python

   from qiskit import QuantumCircuit
   from qiskit.circuit.library import RZZGate
   import QCut as ck
   from QCut import cutGate, CutOptions

   circuit = QuantumCircuit(4)
   for qubit in range(4):
       circuit.ry(0.3 + 0.2 * qubit, qubit)
   circuit.rzz(0.4, 0, 1)
   circuit.rzz(0.5, 2, 3)

   # Both of these cross the same split, and neither touches the other's qubits.
   circuit.append(**cutGate(RZZGate(0.7), 1, 2))
   circuit.append(**cutGate(RZZGate(0.9), 0, 3))

   observables = ["IIIZ", "IIZI", "IZII", "ZIII"]

   cut_circuit = ck.get_locations_and_subcircuits(circuit)
   experiment = ck.get_experiment_circuits(cut_circuit, observables)

   print(experiment.num_circuits)

The two cuts are bundled automatically, giving 60 circuits at
:math:`\gamma \approx 4.86`. Passing ``CutOptions(joint_rotation_cuts=False)`` cuts them
one at a time instead, for 72 circuits at :math:`\gamma \approx 5.87`.

Bundling needs both gates to end up between the same pair of subcircuits. In the example
the ``rzz`` gates on ``(0, 1)`` and ``(2, 3)`` are what hold each half together; without
them the circuit falls apart into four single-qubit pieces and there is no pair of
subcircuits for a bundle to span.

Merging gates on the same pair
------------------------------

Where the previous section is about gates side by side, this one is about gates one
after another. A run of gates acting on the same qubit pair is a single two-qubit
unitary, so it can be cut once rather than once per gate, and QCut merges such runs
before cutting.

Merging is not always the cheaper choice. A run of rotations about different axes
multiplies out to a generic unitary, which is no longer a single-axis rotation and so
can no longer be bundled with a parallel neighbour. ``consolidate`` therefore defaults
to ``"auto"``, which costs both plans and keeps the cheaper one; ``"always"`` merges
wherever it lowers the cost of that pair on its own, and ``"never"`` leaves every gate
alone. See :doc:`Options`.

Note that merging is what makes a run whose product is the identity free: two ``cx``
gates on the same pair merge into an identity that needs no cut at all.
