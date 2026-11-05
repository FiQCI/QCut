Running on hardware
===================

Running on FiQCI
----------------

For running on real hardware through the Lumi supercomputer’s FiQCI
partition follow the instructions
`here <https://docs.csc.fi/computing/quantum-computing/helmi/running-on-helmi/>`__.
If you are used to using Qiskit on jupyter notebooks it is recommended
to use the `Lumi web
interface <https://docs.lumi-supercomputer.eu/runjobs/webui/>`__.

`fiqci-ems <https://github.com/FiQCI/fiqci-ems>`__ can run the experiments too, with its
error mitigation applied: see :doc:`RunningExperiments`.

Running on other hardware
-------------------------

Running on other providers such as IBM is untested at the moment but as
long as the hardware can be accessed with Qiskit version > 1.0 QCut
should be compatible.
