"""Tests for QCutError."""

from qiskit import QuantumCircuit

from QCut.errors.qcuterror import QCutError

# import QCut as ck
from QCut.pre_process.circuit_preparation import get_locations_and_subcircuits

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)


def test_qcuterror() -> None:
    """Test QCutError.

    This function tests whether the QCutError is raised when the cut locations
    are not correctly identified in the provided test circuits by comparing the
    result to the pre-defined solutions.
    """
    try:
        _cut_circuit = get_locations_and_subcircuits(qc)
    except Exception as e:
        assert isinstance(e, QCutError), "Expected a QCutError to be raised."

    try:
        raise QCutError("Test error message", code=123)
    except QCutError as e:
        assert str(e) == "[Error 123] Test error message", (
            "QCutError string representation is incorrect."
        )  # noqa: E501
    try:
        raise QCutError("Test error message without code")
    except QCutError as e:
        assert str(e) == "Test error message without code", (
            "QCutError string representation is incorrect when no code is provided."
        )  # noqa: E501
