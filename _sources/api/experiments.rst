Experiments and execution
=========================

.. currentmodule:: QCut

See :doc:`../RunningExperiments` for how they fit together.

Building experiments
--------------------

.. autofunction:: get_experiment_circuits

.. data:: ObservablesLike

   What observables can be given as, the same as Qiskit's estimator takes: a Pauli
   label, a :code:`Pauli`, a :code:`SparsePauliOp`, a :code:`SparseObservable`, a
   :code:`{label: coefficient}` mapping, or any nested sequence of those.

What a run costs
----------------

.. autofunction:: estimate_run

.. autoclass:: RunEstimate

.. autoclass:: JobEstimate

Submitting
----------

.. autofunction:: submit_experiments

.. autofunction:: run_experiments

.. autoclass:: QCutJob

.. autoclass:: PartialBatch

.. autoclass:: ParallelBackend

The whole sequence in one call
------------------------------

.. autofunction:: run

.. autofunction:: run_cut_circuit
