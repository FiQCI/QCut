Automatic cuts
==============

QCut comes with functionality for automatically finding good cut locations that can place both wire and gate cuts.
The number of partitions, the size of the partitions, and the type of cuts can be specified.

Under the hood QCut uses `pymetis <https://github.com/inducer/pymetis>`__  to find good cut locations based on the circuit's graph representation.

.. code:: python

   from qiskit import QuantumCircuit
   from qiskit_aer import AerSimulator
   import QCut as ck
   from QCut import find_cuts, CutOptions

   circuit = QuantumCircuit(6)
   for qubit in range(6):
       circuit.h(qubit)
   for control, target in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
       circuit.rzz(0.8, control, target)
   for qubit in range(6):
       circuit.rx(0.6, qubit)

   observables = ["IIIIZZ", "IIIZZI", "IZZIII", "ZZIIII"]

   options = CutOptions(
      finder_num_partitions=2,
      finder_cut_mode="both",
   )

   cut_circuit = find_cuts(circuit, options=options)

   print(cut_circuit.cut_locations)
   print([subcircuit.num_qubits for subcircuit in cut_circuit.subcircuits])

   estimated_expectation_values = ck.run_cut_circuit(cut_circuit, observables, AerSimulator())

Nothing here says where to cut. The finder settles on the single ``rzz`` in the middle of
the chain, splitting the six qubits into two halves of three for 12 experiment circuits.
Cutting a wire there instead would cost :math:`\gamma = 4` against roughly 2.4 for that
gate, which is the sort of choice ``finder_cut_mode="both"`` exists to make; restricting
it to ``"wire"`` or ``"gate"`` forces one kind.

``finder_num_partitions`` asks for a number of pieces, and ``finder_max_qubits`` asks
instead for a size limit per piece, either as one number for all of them or as a list
with one entry per partition. At least one of the two has to be given.

How cuts are costed
-------------------

METIS minimises the **sum** of the weights of the edges it cuts, but the sampling
overhead of a set of cuts is the **product** of their :math:`\gamma` values. Taking the
logarithm turns one into the other, so each edge is weighted by
:math:`\log\gamma` rather than by :math:`\gamma`, and the partitioner then optimises
the cost that actually matters.

A wire cut weighs :math:`\log 4`, and a gate cut weighs the logarithm of its own
:math:`\gamma`, so an ``rzz`` with a small angle is correctly treated as nearly free.
A cut whose :math:`\gamma` is exactly 1 costs no shots at all, but it still adds
circuits, so its weight floors at one rather than zero.

One thing the weights cannot express is that parallel rotation gates get cheaper when
cut together, since that is a property of a set of edges rather than of any one edge.
Choosing between candidate partitions
--------------------------------------

METIS minimises the weighted edge cut. With ``log gamma`` weights that is exactly the
true cost of a plan, right up until two cuts share a decomposition. A joint rotation
bundle or a communicating block of wires costs less than the product of its parts, and
METIS cannot know that. It also returns only the partitioning that scored best by its
own measure, discarding the rest.

So the finder asks for one partitioning per seed, builds each one out in full, and costs
the finished plans with :func:`~QCut.qpd_operations.plan_cost`, which does see the
bundles. The cheapest wins. ``CutOptions.finder_candidates`` (default 5) sets how many to try, and
because the seeds are consecutive a larger set contains the smaller one, so raising it
cannot give a worse answer.

Meeting a qubit budget
-----------------------

A budget has to be asked for rather than repaired for. The graph METIS partitions has
one node per wire segment, so several nodes belong to one qubit and balancing node
counts says nothing about how many qubits a partition ends up holding. Each of a
qubit's nodes therefore carries an equal share of one qubit's worth, which makes a
partition's weight its qubit count, and ``max_qubits`` becomes the share each partition
may hold. A straddling qubit counts in both, which is what a wire cut costs anyway.

The weighting only applies when there is a budget. With none, an unbalanced split is
often cheaper, so forcing balance would make that case worse.

``find_cuts`` still applies joint cutting to whatever cuts it settles on, it just does
not actively steer the partitioner towards cut sets that would bundle well.