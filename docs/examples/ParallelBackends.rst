Running on several backends
===========================

A cut experiment is a collection of many independent circuits, so it does not have to run on one
machine. :class:`~QCut.ParallelBackend` takes several backends and
hands them the batches in turn.

.. code:: python

   from qiskit import QuantumCircuit
   from qiskit.circuit.library import CXGate
   import QCut as ck
   from QCut import ParallelBackend, cutGate

   circuit = QuantumCircuit(6)
   for qubit in range(6):
       circuit.ry(0.4 + 0.2 * qubit, qubit)
   circuit.cx(0, 1)
   circuit.cx(3, 4)
   circuit.append(**cutGate(CXGate(), 2, 3))
   circuit.cx(4, 5)

   cut_circuit = ck.get_locations_and_subcircuits(circuit)
   experiment = ck.get_experiment_circuits(cut_circuit, ["IIIIIZ"])

   backend = ParallelBackend([first_device, second_device])
   job = ck.submit_experiments(experiment, shots=4096, backend=backend)

   expectation_values = ck.estimate_expectation_values(job.result())

   backend.submitted  # circuits given to each, e.g. [1728, 1728]

Why it can be shared
--------------------

QCut submits every batch of a wave before collecting any of it, so the machines queue at
the same time rather than one after another. Communicating wire cuts still run in waves,
because a preparing side needs the measuring side's outcome, but the circuits *within* a
wave are independent and are shared like any others.

The experiment is not transpiled beforehand. Each batch is transpiled for the backend
about to run it, so only one batch is ever held in its transpiled form and the
experiment itself stays as it was built.

Machines of different sizes
---------------------------

A batch only goes to a backend with room for it, so the pieces that fit anywhere are
shared while the widest go to the backends that can fit them. Cutting a circuit into a
wide piece and a narrow one and giving it a large device and a small one uses both:

.. code:: python

   backend = ParallelBackend([five_qubit_device, twenty_qubit_device])
   job = ck.submit_experiments(experiment, shots=4096, backend=backend)

If nothing is wide enough for a subcircuit, QCut says so rather than letting the device
refuse the job. Cut into smaller pieces with ``max_qubits``, or pass a larger backend.

Writing your own
----------------

``ParallelBackend`` is not special: QCut asks a backend only for
``run(circuits, shots=..., **options)`` returning a job whose ``result()`` gives counts
by position. Anything answering that can decide for itself where circuits go.

.. code:: python

   import QCut as ck

   class MyParallelBackend:
       def __init__(self, backends):
           self.backends = list(backends)
           self._turn = 0

       def run(self, circuits, shots=1024, **options):
           backend = self.backends[self._turn % len(self.backends)]
           self._turn += 1
           ready = ck.transpile_circuits(list(circuits), backend)
           return backend.run(ready, shots=shots, **options)

:func:`~QCut.transpile_circuits` is what QCut transpiles with
itself, so an IQM device goes through IQM's own transpiler and a resonator machine gets
its MOVE routing. It takes a ``CutCircuit``, a ``CutExperiment`` or plain circuits.

Four things keep it correct:

- **Return the backend's own job.** Its ``result()`` already answers ``get_counts(index)``
  for that batch, which is what QCut reads. Splitting one batch across machines instead
  means merging the results back into submission order yourself.
- **Pass ``shots`` through untouched.** It is what the estimate divides by, and the waves
  allocate it themselves.
- **Submit one batch per call and do not merge them.** The batches are already sized, and
  waves are grouped so that circuits wanting very different shot counts do not share
  a job.
- **Do not wait for results inside** ``run``. Returning immediately is what leaves the
  machines running at the same time.

A backend that declares ``max_shots`` has it respected; ``ParallelBackend`` reports the
smallest of its own, since a batch may land on any of them. Add ``max_batch_size`` to
``run``'s signature to use the value QCut used.
