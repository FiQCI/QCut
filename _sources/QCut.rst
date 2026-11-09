QCut API reference
==================

.. toctree::
   :hidden:

   api/cutting
   api/options
   api/transpilation
   api/primitives
   api/experiments
   api/results

.. currentmodule:: QCut

Cutting
-------

Marking cuts by hand, finding them automatically, and splitting the circuit.

.. autosummary::

   cut
   cutGate
   cutCZ
   cutSWAP
   cutISWAP
   get_locations_and_subcircuits
   find_cuts
   consolidate_two_qubit_blocks

Options
-------

.. autosummary::

   CutOptions

and :data:`DEFAULT_OPTIONS`, used when none are given.

Transpilation
-------------

.. autosummary::

   transpile_subcircuits
   transpile_experiments
   transpile_circuits

Primitives
----------

Several cut circuits in one call, and the jobs and results that call hands back.

.. autosummary::

   CutEstimator
   CutQuasiSampler
   primitives.CutEstimatorJob
   primitives.CutEstimatorResult
   primitives.CutEstimatorEntry
   primitives.CutQuasiSamplerJob
   primitives.CutQuasiSamplerResult
   primitives.CutQuasiSamplerEntry
   primitives.QCutPrimitiveJob
   primitives.CutRunEstimate

Experiments and execution
-------------------------

One cut circuit step by step: build its experiment, check the cost, submit it.

.. autosummary::

   get_experiment_circuits
   estimate_run
   RunEstimate
   JobEstimate
   submit_experiments
   run_experiments
   QCutJob
   PartialBatch
   ParallelBackend
   run
   run_cut_circuit

and :data:`ObservablesLike`, what observables can be given as.

Results and post-processing
---------------------------

.. autosummary::

   RawResult
   CircuitResult
   estimate_expectation_values
   estimate_standard_errors
   estimate_probabilities
   QuasiProbabilities
