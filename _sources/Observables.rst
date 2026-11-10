Defining observables
====================

Observables are given the way Qiskit's estimator takes them, written on the qubits of
the original uncut circuit. :func:`~QCut.get_experiment_circuits`, :func:`~QCut.run`,
:func:`~QCut.run_cut_circuit` and :class:`~QCut.CutEstimator` all take them the same way.

.. code:: python

   from QCut import CutEstimator
   from qiskit.quantum_info import SparsePauliOp, SparseObservable

   sim = AerSimulator()
   fake = IQMFakeAdonis()

Pauli strings
-------------

A Pauli label spans the whole uncut circuit, with qubit 0 rightmost as in Qiskit:

.. code:: python

   "IIIIZ"   # Z on qubit 0
   "ZIIII"   # Z on qubit 4
   "IIIXX"   # X on qubits 0 and 1

The values come back shaped like the observables, so a list of four gives four values
and a nested list a nested array:

.. code:: python

   observables = [["IIIIZ", "IIIZI"], ["IIZII", "IZIII"]]   # values of shape (2, 2)

SparsePauliOp
-------------

A weighted sum of Paulis. :code:`from_sparse_list()` names only the qubits each term acts
on, which is easier than writing out labels on a wide circuit:

.. code:: python

   op = SparsePauliOp.from_sparse_list(
       [("ZZ", [0, 1], 0.5), ("XX", [3, 4], 0.5)], num_qubits=5
   )

A :code:`{label: coefficient}` dict is the same thing written out, here
:code:`{"IIIZZ": 0.5, "XXIII": 0.5}`.

SparseObservable
----------------

On Qiskit 2.1 or newer, a :code:`SparseObservable` can also carry projectors:
:code:`0`, :code:`1`, :code:`+`, :code:`-`, :code:`r` and :code:`l`. QCut expands each
into Paulis, :math:`|0\rangle\langle 0| = (I + Z)/2` and so on, so the probability that
qubits 0 and 1 both read 0 is:

.. code:: python

   so = SparseObservable.from_sparse_list([("00", [0, 1], 1.0)], num_qubits=5)

A term on :math:`k` projectors becomes :math:`2^k` Pauli terms, all measured in one
setting, up to 16 projectors per term. For the whole distribution over many qubits, use
:code:`qubits` or :class:`~QCut.CutQuasiSampler` instead, see
:ref:`probability distributions <probability-distributions>`.

All three in one run, on the cut circuit from :doc:`CuttingCircuits`:

.. code:: python

   job = CutEstimator(sim).run(cut_circuit, ["IIIIZ", op, so], shots=8192)
   print(job.expectation_values(0))

``[0.9247 0.8691 0.8501]``, against exact values of ``[0.9211 0.8642 0.8476]``.

No layout to apply
------------------

With Qiskit's own estimator, an observable for a transpiled circuit has to be moved onto
the physical qubits the transpiler chose, with :code:`observable.apply_layout(layout)`.
QCut does not need this. Observables stay on the uncut circuit's qubits throughout:
QCut keeps track of which subcircuit each qubit went to and, once transpiled, which
physical qubit holds it, and measures through both. The same observables work before and
after :doc:`transpilation <Transpilation>`:

.. code:: python

   transpiled = ck.transpile_subcircuits(cut_circuit, fake)
   job = CutEstimator(fake).run(transpiled, ["IIIIZ", op, so])

Do not apply a layout yourself. On a device larger than the circuit, the observable then
spans the device rather than the circuit and is refused:

``ValueError: observable 'IIIIIIIIIIIZIIIIZIII' spans 20 qubits, but every observable
must match the number of qubits in the original uncut circuit (5). If a transpiled layout
was applied to it with apply_layout(), pass it as written for the uncut circuit instead.
QCut handles the layout itself.``

On a device the same size as the circuit it is accepted and silently measures other
qubits. On :code:`IQMFakeAdonis`, the :code:`XX` on qubits 3 and 4 above became an
:code:`XX` on qubits 2 and 4.
