"""What a backend can do, however it describes itself."""

from __future__ import annotations

import logging

from qiskit.circuit.library import get_standard_gate_name_mapping

from QCut.errors.qcuterror import QCutError

logger: logging.Logger = logging.getLogger(__name__)


def is_resonator_backend(backend) -> bool:
    """
    Whether the device couples its qubits only through a resonator.
    """
    has_resonators = getattr(backend, "has_resonators", None)
    return bool(has_resonators and has_resonators())


def _backend_gate_names(backend) -> list[str]:
    """The gate names a backend accepts, however that backend spells them.

    A BackendV1 answers through its configuration, an IQM device through its
    architecture, and any other BackendV2 through its target. IQM is asked before the
    target because its architecture is what the prx rename below was written against.

    Raises:
        QCutError: the backend names its gates in none of the three ways.
    """
    # TODO(3.0): BackendV1 is gone from qiskit 2.0, so this branch can go.
    configuration = getattr(backend, "configuration", None)
    if configuration is not None:
        try:
            return list(configuration().basis_gates)
        except Exception:  # noqa: BLE001 - a V2 backend may carry a stub that raises
            pass

    architecture = getattr(backend, "architecture", None)
    if architecture is not None:
        return list(architecture.gates.keys())

    target = getattr(backend, "target", None)
    if target is not None:
        return list(target.operation_names)

    raise QCutError(
        f"cannot tell which gates {type(backend).__name__} accepts, because it has no "
        "configuration(), no architecture and no target. Please raise an issue so this "
        "backend can be supported. In the meantime, build the experiment circuits from "
        "an untranspiled cut circuit and transpile them with transpile_experiments."
    )


def _device_basis(backend) -> list[str]:
    """The backend's gates as qiskit basis gate names.

    What the experiment's own operations -- the observable basis changes and the QPD
    terms -- are transpiled to when the subcircuits were transpiled for a device.

    Raises:
        QCutError: the backend names its gates in none of the ways
            :func:`_backend_gate_names` knows.
    """
    basis = _backend_gate_names(backend)
    basis = ["r" if gate == "prx" else gate for gate in basis]
    # A device's own gate list can name operations qiskit does not know, and it
    # refuses those through ``basis_gates`` rather than ignoring them. IQM's
    # resonator machines list ``move``, which shifts a state between a qubit and a
    # resonator, so a Deneb-class backend used to fail here. The basis changes
    # translated with this are single-qubit, so anything qiskit cannot name is of no
    # use to them and is dropped.
    known = set(get_standard_gate_name_mapping())
    known.update(("measure", "reset", "delay", "barrier", "id"))
    dropped = [gate for gate in basis if gate not in known]
    if dropped:
        logger.debug(
            f"ignoring backend gate(s) {dropped}, which qiskit cannot take as a "
            "basis gate, while translating the observable basis changes"
        )
    return [gate for gate in basis if gate in known]


def _backend_shot_cap(backend) -> int | None:
    """Return the most shots a backend takes in one job, if it says."""
    for probe in (
        lambda: backend.max_shots,
        lambda: backend.configuration().max_shots,
    ):
        try:
            cap = probe()
        except Exception:
            continue
        if isinstance(cap, int) and cap > 0:
            return cap
    return None
