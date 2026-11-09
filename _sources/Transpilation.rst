Transpilation
=============

Two helpers, :code:`transpile_subcircuits()` and :code:`transpile_experiments()`,
differing in when they run:

.. code:: python

   fake = IQMFakeAdonis() #noisy
   sim = AerSimulator() #ideal

:code:`transpile_circuits()` does whichever of the two the thing it is given calls for.
Each subcircuit once, before the experiment circuits are built:

.. code:: python

   transpiled = ck.transpile_circuits(cut_circuit, fake, optimization_level=3)
   cut_experiment = ck.get_experiment_circuits(transpiled, observables)

Or every experiment circuit, afterwards:

.. code:: python

   cut_experiment = ck.transpile_circuits(
       ck.get_experiment_circuits(cut_circuit, observables), fake, optimization_level=3
   )

Given a cut circuit it is much the faster of the two, but the subcircuits still carry the
cut and observable placeholders, so the transpiler is working on a circuit it cannot see
all of. It therefore disallows :code:`remove_final_rzs` (:code:`False` by default on :code:`iqm_client>=36.0.0`) and
:code:`optimize_single_qubits` and raises if you pass them. Given an experiment there are
no placeholders left, so it optimises further.

A cut circuit is transpiled as its subcircuits, never as the whole circuit: the pieces run
separately, so each is placed on the device on its own, and the whole circuit need not
fit the device at all. The transpiled :code:`CutCircuit` remembers the backend it was
transpiled for, which is what :doc:`Primitives` check before running it.

Plain circuits are transpiled as they are, which is what a backend handed circuits rather
than an experiment needs. See :ref:`several-backends`.

On an IQM backend both use IQM's own transpiler. Pass :code:`use_iqm_transpiler=False`
for the ordinary Qiskit path. On a resonator device such as :code:`IQMFakeDeneb` the MOVE
gates are routed in for you.
