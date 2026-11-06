"""Storage class for storing cut locations."""

from __future__ import annotations

from qiskit import QuantumRegister
from qiskit.circuit import Gate


def _bind(param, parameters: dict):
    """Substitute ``parameters`` into one gate parameter, keyed by object or by name.

    Both key forms are accepted because ``QuantumCircuit.assign_parameters`` accepts
    both.
    """
    free = getattr(param, "parameters", None)
    if not free:
        return param
    for symbol in list(free):
        if symbol in parameters:
            param = param.assign(symbol, parameters[symbol])
        elif symbol.name in parameters:
            param = param.assign(symbol, parameters[symbol.name])
    if getattr(param, "parameters", None):
        return param
    return float(param)


class CutLocation:
    """Storage class for storing cut locations."""

    def __init__(
        self,
        cut_location: tuple[list, int],
        gate_name: str = "cz",
        gate: Gate | None = None,
    ) -> None:
        """Init.

        ``gate`` is only needed for gates without a hand-written QPD, where one is
        generated from the gate's matrix.
        """
        self.qubits = cut_location[0]
        self.control = cut_location[0][0][1]
        self.target = cut_location[0][1][1]
        self.index = cut_location[1]
        self.gate_name = gate_name
        self.gate = gate
        # Cached by QCut.qpd_operations.qpd_for_location, since generating a QPD costs a
        # KAK decomposition and it is looked up once per experiment group.
        self._qpd: list[dict] | None = None

    def assign_parameters(self, parameters: dict) -> CutLocation:
        """Return a copy with ``parameters`` bound into the stored gate.

        A parametrised cut gate survives the split unbound, so binding has to reach it
        here as well as in the subcircuits. Returns ``self`` if there is nothing to do.
        """
        if self.gate is None or not self.gate.params:
            return self
        bound_params = [_bind(param, parameters) for param in self.gate.params]
        if bound_params == list(self.gate.params):
            return self
        gate = self.gate.copy()
        gate.params = bound_params
        return CutLocation((self.qubits, self.index), self.gate_name, gate)

    def __eq__(self, other) -> bool:
        """Equality."""
        if not isinstance(other, CutLocation):
            return NotImplemented

        return (
            self.control == other.control
            and self.target == other.target
            and self.index == other.index
            and self.gate_name == other.gate_name
        )

    def __str__(self) -> str:
        """Format string."""
        msg = (
            f"control qubit: {self.control}, target qubit: {self.target}, "
            f"gate: {self.gate_name}, cut index: {self.index}"
        )
        return msg

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


class SingleQubitCutLocation:
    """Storage class for storing cut locations."""

    def __init__(self, cut_location: tuple[tuple[QuantumRegister, int], int]) -> None:
        """Init."""
        self.qubits = cut_location[0]
        self.meas = cut_location[0][1]
        self.init = cut_location[0][1]
        self.index = cut_location[1]

    def __eq__(self, other) -> bool:
        """Equality."""
        if not isinstance(other, SingleQubitCutLocation):
            return NotImplemented

        return (
            self.meas == other.meas
            and self.init == other.init
            and self.index == other.index
        )

    def __str__(self) -> str:
        """Format string."""
        msg = (
            f"meas qubit: {self.meas}, init qubit: {self.init}, cut index: {self.index}"
        )
        return msg

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)
