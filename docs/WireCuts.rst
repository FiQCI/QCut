Wire cuts
=========

Wire cuts can be used to cut the wires of qubits in a circuit. This is done by inserting special cut instructions into the circuit.

A wire cut marks a point on a qubit's wire where the qubit is measured and a fresh state
prepared, so everything before the cut and everything after it can run as separate
circuits. Unlike a gate cut it does not need a gate to attach to, which makes it the
only option where a qubit has to be handed from one partition to another without an
interaction to cut.

.. code:: python

   from qiskit import QuantumCircuit
   from QCut import cut

   cut_circuit = QuantumCircuit(3)
   cut_circuit.h(0)
   cut_circuit.cx(0,1)
   cut_circuit.append(cut(), [1])
   cut_circuit.cx(1,2)

   cut_circuit.draw("mpl")

After this the circuit can be processed as usual with QCut (take a look at the Usage documentation for more details).

Cutting several wires at once
-----------------------------

Cutting :math:`n` wires separately costs :math:`4^n`, since each wire cut costs 4 and
the overheads multiply. Wires cut at the same point in the circuit can do better than
that if the two sides are allowed to exchange the measured outcome: the side that
measures tells the side that prepares what it saw, and the block then costs
:math:`2^{n+1} - 1`.

.. list-table::
   :header-rows: 1

   * - wires in the block
     - cut separately
     - exchanging outcomes
   * - 1
     - 4
     - 3
   * - 2
     - 16
     - 7
   * - 3
     - 64
     - 15

The shot count scales as :math:`\gamma^2`, so a block of two is around five times
cheaper and a block of three around eighteen times. The construction is ancilla-free and
is derived in :doc:`Theory`.

.. code:: python

   from qiskit import QuantumCircuit
   import QCut as ck
   from QCut import cut, CutOptions

   circuit = QuantumCircuit(4)
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.cx(0, 2)

   # Two wires cut at the same point, so they can share one decomposition.
   circuit.append(cut(), [1])
   circuit.append(cut(), [2])

   circuit.cx(1, 2)
   circuit.cx(2, 3)

   observables = ["IIIZ", "IIZI", "IZII", "IIZZ"]

   cut_circuit = ck.get_locations_and_subcircuits(circuit)
   experiment = ck.get_experiment_circuits(cut_circuit, observables)

   print(experiment.num_circuits)

That gives 56 circuits. ``CutOptions(wire_cut_communication="never")`` cuts the two
wires independently instead, for 128.

What it costs to run
--------------------

Exchanging the outcome means one side cannot be run until the other has been, so the
experiment runs in waves rather than all at once: every measuring subcircuit first, then
the preparing ones, with each wave's shots allocated according to what the previous wave
actually measured. Chains take more than two waves, since a subcircuit can be a
preparing side and a measuring side at the same time.

The advertised :math:`\gamma` assumes the prepared state follows the measured outcome
shot by shot. Emulating that with batched runs and post-selection costs extra, and at a
single wire that extra outweighs the gain, which is why ``wire_cut_communication``
defaults to ``"auto"`` and applies only to blocks of two or more. ``"always"`` uses it
for any block including single wires, and ``"never"`` for none. See :doc:`Options`.
