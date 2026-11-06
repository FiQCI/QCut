Getting started
===============

Installation
------------

For installation a UNIX-like system is currently needed due to
`PyMetis <https://github.com/inducer/pymetis>`__ being used for automatic cut finding.
On Windows use WSL.

Pip
~~~

Installation should be done via `uv <https://docs.astral.sh/uv/>`__:

.. code:: bash

   uv pip install QCut
   #or
   uv add QCut

If using other than the default Qiskit version (newest) it is recommended to install
Qiskit first before installing QCut.

QCut does not install a simulator. The examples run on
`Qiskit Aer <https://github.com/Qiskit/qiskit-aer>`__, which needs
``uv pip install qiskit-aer``. For drawing circuits you might also need
``uv pip install pylatexenc``.

uv itself can be installed with:

.. code:: bash

   #Linux / mac
   curl -LsSf https://astral.sh/uv/install.sh | sh

IQM hardware and fake backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QCut ships an extra for this, so the adapter comes with it:

.. code:: bash

   uv pip install "QCut[iqm]"
   #or
   uv add "QCut[iqm]"

That installs `IQM client <https://docs.meetiqm.com/iqm-client>`__ with its Qiskit
adapter, which covers Qiskit 1.0 up to but not including 2.2. Installing the extra will
therefore hold Qiskit below 2.2, put it in an environment whose Qiskit is already in
range if you would rather the resolver left it alone.

After installation just import the backend you want to use:

.. code:: python

   from iqm.qiskit_iqm import IQMFakeAdonis
   backend = IQMFakeAdonis()

Install from source
~~~~~~~~~~~~~~~~~~~

It is also possible to use QCut by cloning the repository and including it in your
project folder:

.. code:: bash

   cd QCut
   uv pip install .
   #or
   uv sync --no-dev

   #or with dev deps
   uv sync

Quick start
-----------

Mark where to cut, split the circuit, build the experiment for the observables wanted,
run it, and put the expectation values back together:

.. code:: python

   import QCut as ck
   from QCut import cut, cutGate
   from qiskit import QuantumCircuit
   from qiskit.circuit.library import CXGate
   from qiskit_aer import AerSimulator

   circuit = QuantumCircuit(5)
   for qubit in range(5):
       circuit.ry(0.4 + 0.3 * qubit, qubit)
   circuit.cx(0, 1)
   circuit.append(**cutGate(CXGate(), 1, 2))  # a gate cut
   circuit.cx(2, 3)
   circuit.append(cut(), [3])  # a wire cut
   circuit.cx(3, 4)

   observables = ["IIIIZ", "IIZII", "ZZIII"]

   cut_circuit = ck.get_locations_and_subcircuits(circuit)
   # On a device with a coupling map, transpile the subcircuits for it here:
   # cut_circuit = ck.transpile_subcircuits(cut_circuit, backend)
   experiment = ck.get_experiment_circuits(cut_circuit, observables)

   job = ck.submit_experiments(experiment, backend=AerSimulator(), shots=4096)
   print(ck.estimate_expectation_values(job.result()))

``[0.9204 0.3866 -0.0315]``

against exact values of ``[0.9211 0.3806 -0.0292]`` for the uncut circuit.

:doc:`Usage` goes through each step, :doc:`Primitives` runs several cut circuits in one
call, and :doc:`Options` lists what can be configured.
