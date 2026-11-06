Primitives
==========

For running several cut circuits in one call, QCut has two primitives shaped like
`FiQCI-EMS <https://github.com/FiQCI/fiqci-ems>`__'s estimator and sampler:
:class:`~QCut.CutEstimator` gives expectation values and :class:`~QCut.CutQuasiSampler`
quasi-probability distributions. Each is made once for a backend, and its :code:`run()`
submits every circuit and hands back a job at once, without waiting for results.

They take cut circuits, as :func:`~QCut.get_locations_and_subcircuits` or
:func:`~QCut.find_cuts` make them, and each runs with the :doc:`options <Options>` it was
made with.

Estimating expectation values
-----------------------------

Continuing the example from :doc:`CuttingCircuits`, here with two observables on the
hand-cut circuit and a second, unrelated circuit whose cuts are found automatically:

.. code:: python

   from QCut import CutEstimator

   ghz = QuantumCircuit(6)
   ghz.h(0)
   for qubit in range(5):
       ghz.cx(qubit, qubit + 1)
   found_ghz = find_cuts(ghz, options=CutOptions(finder_num_partitions=2))

   estimator = CutEstimator(sim)
   job = estimator.run(
       [cut_circuit, found_ghz], [["IIIIZ", "IIZII"], "ZZZZZZ"], shots=4096
   )
   print(job.result().expectation_values())

``[array([0.9227, 0.3818]), array(1.0001)]``

The observables pair with the circuits the way FiQCI's estimator pairs them:

- One circuit takes its observables exactly as given.
- With a list of circuits, a list of observables has one entry per circuit, each an
  observable or a list of them, as above.
- Anything else is a single observable used for every circuit.

Each entry of the result belongs to one input circuit, in the order given.
:code:`result[i].expectation_values` is shaped like that circuit's observables,
:code:`result[i].standard_errors` bounds their shot noise as
:func:`~QCut.estimate_standard_errors` does, and :code:`result[i].raw` is the
:class:`~QCut.RawResult` they were estimated from. :code:`job.expectation_values(i)` and
:code:`job.standard_errors(i)` are shortcuts for the first two.

Reconstructing distributions
----------------------------

The sampler reconstructs the distribution over the qubits the circuit measured at its
end. Measure before cutting: splitting strips the final measurements, and the cut circuit
remembers where they were.

.. code:: python

   from QCut import CutQuasiSampler

   measured = marked_circuit.copy()
   measured.measure_all()

   sampler = CutQuasiSampler(sim)
   job = sampler.run(ck.get_locations_and_subcircuits(measured), shots=4096)
   distribution = job.result()[0]

   print(distribution.quasi_probabilities.top(4))

``{'10000': 0.2252, '00000': 0.1881, '01000': 0.1184, '11000': 0.1178}``

The bitstrings follow Qiskit's convention, classical bit zero rightmost, whatever qubit it
was measured from. Every classical bit has to be measured into exactly once.

:code:`quasi_probabilities` is a :class:`~QCut.QuasiProbabilities`, so everything in
:ref:`probability distributions <probability-distributions>` applies, and it never
expands the whole distribution unless asked. :code:`quasi_dist()` gives a Qiskit
:code:`QuasiDistribution` instead, which does expand all :code:`2**n` outcomes unless
:code:`top` limits it:

.. code:: python

   distribution.quasi_dist(top=3)

``{16: 0.2252, 0: 0.1881, 8: 0.1184}``

Running on a device
-------------------

The primitives do not transpile, much as Qiskit's primitives take circuits already
written for their target. On a backend with a coupling map, transpile each cut circuit
for that same backend first:

.. code:: python

   fake = IQMFakeAdonis()

   transpiled = [ck.transpile_subcircuits(c, fake) for c in [cut_circuit, found]]
   job = CutEstimator(fake).run(transpiled, observables)

A cut circuit transpiled for a different backend object, or not transpiled at all, is
refused there, before anything is submitted. A backend without a coupling map, such as an
ideal simulator or a V2 sampler, runs the subcircuits as they are.

What a run costs
----------------

:code:`estimate()` takes the same arguments as :code:`run()` and submits nothing:

.. code:: python

   print(estimator.estimate([cut_circuit, found_ghz], [["IIIIZ", "IIZII"], "ZZZZZZ"], shots=4096))

``Experiment will run 41 circuits in 2 jobs with a total of 167936 shots. See the
returned object for the breakdown.``

The totals cover the whole run, and indexing gives one circuit's share, as a
:class:`~QCut.RunEstimate`. It is exact wherever :func:`~QCut.estimate_run` is.

The job
-------

The job answers for every circuit at once. :code:`job_ids()`, :code:`status()`,
:code:`cancel()` and :code:`wait_for_final_state()` work as on a Qiskit job,
:code:`statuses()` and :code:`partial_results()` give one entry per circuit, and
:code:`job.jobs` holds each circuit's own :class:`~QCut.QCutJob`.

Each circuit is submitted in batches of its own, so circuits from different inputs never
share a backend job.
