Primitives
==========

See :doc:`../Primitives` for how they are used. Each primitive's :code:`run()` hands
back a job, the job's :code:`result()` a result with one entry per input circuit.

Estimator
---------

.. autoclass:: QCut.CutEstimator

.. autoclass:: QCut.primitives.CutEstimatorJob

.. autoclass:: QCut.primitives.CutEstimatorResult

.. autoclass:: QCut.primitives.CutEstimatorEntry

Sampler
-------

.. autoclass:: QCut.CutQuasiSampler

.. autoclass:: QCut.primitives.CutQuasiSamplerJob

.. autoclass:: QCut.primitives.CutQuasiSamplerResult

.. autoclass:: QCut.primitives.CutQuasiSamplerEntry

Shared
------

.. autoclass:: QCut.primitives.QCutPrimitiveJob

.. autoclass:: QCut.primitives.CutRunEstimate
