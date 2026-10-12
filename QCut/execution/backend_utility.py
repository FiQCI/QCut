"""
Utility functions for running on real backends.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Instruction
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.transpiler import PassManager, Target

from QCut.cutcircuit import CutCircuit, CutExperiment
from QCut.cutlocation import CutLocation, SingleQubitCutLocation
from QCut.errors.qcuterror import QCutError
from QCut.execution.move_routing import is_resonator_backend, move_route
from QCut.qpd.bundle import locate_placeholders, plan_bundles
from QCut.utils.circuit_utils import (
    MarkerSpan,
    _drop_barriers,
    _fence_markers,
    _record_layout,
    barriers_to_markers,
    fuse_markers,
    markers_to_barriers,
    split_markers,
)

logger: logging.Logger = logging.getLogger(__name__)

try:
    from qiskit.transpiler.passes import RemoveIdentityEquivalent

    #: None on qiskit 1.2 and older, which do not have the pass.
    _REMOVE_IDENTITIES: PassManager | None = PassManager([RemoveIdentityEquivalent()])
except ImportError:  # pragma: no cover - depends on the installed qiskit
    _REMOVE_IDENTITIES = None


#: How wide a block one marker will stand in for. Two covers the case that pays: a pair
#: of wire cuts communicating costs gamma 7 against the 16 of cutting them locally.
#: Wider blocks need all of their wires mutually coupled, which no device this targets
#: has, so they are left to fall back to narrower ones.
WIDEST_BLOCK_MARKER: int = 2

#: Prefix of the marker that stands in for a whole block while transpiling.
BLOCK_MARKER: str = "qcut_block"


def _block_spans(
    cut_circuit,
) -> tuple[dict[int, list[MarkerSpan]], dict[str, list[str]]]:
    """Find the placeholder runs that should be transpiled as one wider marker.

    Planning happens here, on the logical subcircuits, because the whole point is to
    tell the transpiler something before it runs. The experiment builder plans again
    afterwards and vetoes any block the device does not couple, but a block routed as
    one marker passes that veto, and its placeholders come back consecutive, so it is
    found again.
    """
    bundles = plan_bundles(
        cut_circuit.cut_locations,
        cut_circuit.subcircuits,
        cut_circuit.options,
        announce=False,
        fits=lambda bundle: bundle.size <= WIDEST_BLOCK_MARKER,
    )
    placeholders = locate_placeholders(cut_circuit.subcircuits)
    spans: dict[int, list[MarkerSpan]] = defaultdict(list)
    members: dict[str, list[str]] = {}
    for order, bundle in enumerate(bundles):
        if bundle.size < 2:
            continue
        for side, group in enumerate(bundle.layout):
            owner = placeholders[group[0]].subcircuit
            subcircuit = cut_circuit.subcircuits[owner]
            indices = [placeholders[member].index for member in group]
            name = f"{BLOCK_MARKER}_{order}_{side}"
            spans[owner].append(
                MarkerSpan(
                    name,
                    tuple(placeholders[member].qubit for member in group),
                    tuple(indices),
                    min(indices) if bundle.place[side] == "first" else max(indices),
                )
            )
            members[name] = [subcircuit.data[index].operation.name for index in indices]
    return spans, members


def _add_block_gates(target, coupling_map, members: dict[str, list[str]]) -> set[str]:
    """Declare each block marker on both directions of every coupled pair.

    Both directions, because a device's coupling map names each pair once and the layout
    is free to hand the marker its qubits either way round. Qiskit can turn a ``cz`` the
    right way because it knows the gate is symmetric; a marker is opaque, so an
    unsupported orientation is a transpiler error rather than something it can fix.

    Without a coupling map every pair is coupled, so the marker is declared global.
    """
    loci = None
    if coupling_map is not None:
        loci = {}
        for first, second in coupling_map.get_edges():
            loci[(first, second)] = None
            loci[(second, first)] = None
    for name, group in members.items():
        target.add_instruction(
            Gate(num_qubits=len(group), name=name, params=[], label=name),
            loci,
            name=name,
        )
    return set(members)


def _markers_in(circuit, names: set[str]) -> list[str]:
    """The placeholders a circuit carries, sorted, for comparing before with after."""
    return sorted(
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name in names
    )


def _translate_subcircuit(subcircuit, spans, translate, marker_names):
    """Transpile one subcircuit with its blocks fused, or plainly if that loses a cut.

    Standing a wider marker in for a block is an optimisation: it costs at most a swap
    and saves far more in shots. Losing a placeholder to it would not be a worse
    optimisation but a wrong answer, and one nothing downstream could notice -- the cut
    would simply not be there. So the fused attempt is checked against the placeholders
    that went in, and anything short of all of them coming back falls back.
    """
    if not spans:
        return translate(subcircuit, [])

    expected = _markers_in(subcircuit, marker_names)
    try:
        fused = translate(subcircuit, spans)
    except Exception as error:  # noqa: BLE001 - a failed optimisation must not propagate
        logger.info("could not transpile this subcircuit's blocks as one: %s", error)
        return translate(subcircuit, [])

    if _markers_in(fused, marker_names) == expected:
        return fused

    logger.info(
        "the wider marker standing in for a block did not survive transpilation, so "
        "this subcircuit is transpiled without it. Its cuts are then decomposed "
        "separately, which costs more shots but is what the device can run."
    )
    return translate(subcircuit, [])


#: What each of IQM's own options does to a circuit that still carries placeholders,
#: and therefore the value :func:`transpile_subcircuits` has to hold it at.
_IQM_ENFORCED: dict[str, tuple[bool, str]] = {
    "remove_final_rzs": (
        False,
        "drops the trailing Z rotations, which are unobservable before a Z measurement "
        "but not before the X and Y basis changes QCut adds afterwards",
    ),
    "optimize_single_qubits": (
        False,
        "commutes Z rotations along each wire, and a barrier does not stop it, so a "
        "rotation written before a cut ends up applied after the wire has been "
        "measured and re-prepared",
    ),
}


def _check_iqm_options(transpile_options: dict | None) -> None:
    """Raise if an option would break a circuit that still carries placeholders.

    Raises:
        QCutError: one of :data:`_IQM_ENFORCED` was overridden.
    """
    for name, (required, why) in _IQM_ENFORCED.items():
        asked = (transpile_options or {}).get(name, required)
        if asked == required:
            continue
        raise QCutError(
            f"transpile_subcircuits cannot honour {name}={asked!r} on an IQM "
            f"backend: it {why}. Subcircuits still carry the cut and observable "
            "placeholders, so the transpiler is working on a circuit it cannot see "
            "all of. Transpile the finished experiment circuits with "
            f"transpile_experiments instead, which keeps {name}={asked!r}."
        )


def _drop_identities(circuit):
    """Remove the identity gates a QPD table writes out.

    IQM's single-qubit pass refuses to see one: it is not in the basis, and unlike the
    other gates in an experiment circuit it has no translation into it. They stay in
    until here because the insertion sites count a table entry's instructions when they
    shift the placeholder indices behind it, and ``id-meas``, ``0-init`` and ``I`` are
    nothing but the ``id``.
    """
    if "id" not in circuit.count_ops():
        return circuit

    if _REMOVE_IDENTITIES is not None:
        out = _REMOVE_IDENTITIES.run(circuit)
        # The pass goes through a DAG, which does not carry the layout, and a resonator
        # backend reads it to tell a qubit wire from its resonator.
        out._layout = circuit.layout
        return out

    out = circuit.copy_empty_like()
    for instruction in circuit.data:
        if instruction.operation.name != "id":
            out.append(instruction.operation, instruction.qubits, instruction.clbits)
    return out


def _placeholder_gates(cut_circuit) -> dict[str, Gate]:
    """One opaque one-qubit gate per placeholder the subcircuits carry.

    The transpiler is told about them through the target, so it leaves them alone
    instead of refusing to synthesise something it has never heard of.
    """
    gates: dict[str, Gate] = {}

    def add(name: str, width: int = 1) -> None:
        gates[name] = Gate(num_qubits=width, name=name, params=[], label=name)

    for ind, location in enumerate(cut_circuit.cut_locations):
        if isinstance(location, CutLocation):
            add(f"cut{location.gate_name.upper()}_t_{ind}")
            add(f"cut{location.gate_name.upper()}_c_{ind}")
        elif isinstance(location, SingleQubitCutLocation):
            add(f"Meas_{ind}")
            add(f"Init_{ind}")

    for index in range(sum(x.num_qubits for x in cut_circuit.subcircuits)):
        add(f"obs_{index}")
    return gates


def _target_of(backend) -> Target:
    """A backend's target, read through the public ``target``.

    Raises:
        QCutError: the backend has no target to transpile for.
    """
    target = getattr(backend, "target", None)
    if target is None:
        raise QCutError(
            f"{backend} has no target to transpile for. Pass a backend that has one."
        )
    return target


def _basis(target: Target) -> tuple[list[str], dict[str, Instruction]]:
    """The operations a target names, and those of them qiskit would not recognise."""
    standard = get_standard_gate_name_mapping()
    names: set[str] = set()
    custom: dict[str, Instruction] = {}
    for operation in target.operations:
        if not isinstance(operation, Instruction):
            continue
        names.add(operation.name)
        if operation.name not in standard:
            custom[operation.name] = operation
    return sorted(names), custom


def has_topology(backend) -> bool:
    """Whether running on ``backend`` needs its circuits transpiled first."""
    target = getattr(backend, "target", None)
    return target is not None and target.build_coupling_map() is not None


def _iqm_transpiler_for(backend):
    """Return IQM's transpiler if it is installed and ``backend`` is one of its own.

    The adapter comes with ``pip install "QCut[iqm]"``. Without it, or for any other
    backend, this returns None and the ordinary qiskit path is used. The backend check
    is what ``IQMBackendBase`` is imported for: a fake IQM device and a real one both
    derive from it, and nothing else does.
    """
    try:
        from iqm.qiskit_iqm import transpile_to_IQM  # type: ignore
        from iqm.qiskit_iqm.iqm_backend import IQMBackendBase  # type: ignore
    except ImportError:
        return None
    return transpile_to_IQM if isinstance(backend, IQMBackendBase) else None


def _transpiler_for(backend, use_iqm_transpiler: bool):
    """IQM's transpiler if it applies, or None for the ordinary qiskit path.

    Raises:
        QCutError: the device needs MOVE routing, which only IQM's transpiler does.
    """
    iqm_transpile = _iqm_transpiler_for(backend) if use_iqm_transpiler else None
    if iqm_transpile is None and is_resonator_backend(backend):
        raise QCutError(
            "this backend needs MOVE gates around every two-qubit gate, which only "
            "IQM's transpiler inserts. Leave use_iqm_transpiler at True for it."
        )
    return iqm_transpile


def transpile_subcircuits(
    cut_circuit: CutCircuit,
    backend,
    optimization_level: int = 3,
    transpile_options: dict | None = None,
    use_iqm_transpiler: bool = True,
) -> CutCircuit:
    """
    Transpile subcircuits for a given backend. More efficient than transpiling
    experiment circuits as it only transpiles each subcircuit once instead of
    each experiment circuit. However, may lead to suboptimal transpilation results as
    the transpiler cannot use the backend object directly due to need to retain
    some placeholder gates for cuts and observables. `transpile_options` can be used
    to pass additional options to the transpiler. For more control over transpilation
    of experiment circuits, use `transpile_experiments` or manually transpile them.

    On an IQM backend this gives up IQM's single-qubit optimisation, which would
    otherwise commute Z rotations across the cut placeholders and put them on the wrong
    side of a cut. Nothing can be told to leave the placeholders alone -- a barrier is a
    scheduling directive there, not an algebraic wall -- so circuits come out perhaps a
    fifth deeper than they need to be. :func:`transpile_experiments` has no placeholders
    left to protect and keeps the optimisation, so it is the one to use when the depth
    matters more than the transpilation time.

    Args:
        cut_circuit (CutCircuit): The split circuit whose subcircuits are transpiled.
        backend: Backend to transpile to.
        optimization_level (int): Optimization level for transpilation (0-3).
        transpile_options (dict): Arguments passed to the transpiler.
        use_iqm_transpiler (bool): Whether an IQM backend may use IQM's transpiler.
            Pass False for the ordinary qiskit path.

    Returns:
        CutCircuit: Transpiled subcircuits wrapped in CutCircuit class.
    """

    if not isinstance(cut_circuit, CutCircuit):
        raise ValueError("cut_circuit must be of type CutCircuit.")

    custom_gates = _placeholder_gates(cut_circuit)

    marker_names = set(custom_gates)
    # Where several placeholders will be replaced by one block, so it can be routed as
    # the multi-qubit operation it is rather than as unrelated single-qubit ones.
    spans, block_members = _block_spans(cut_circuit)

    backend_target = _target_of(backend)
    coupling_map = backend_target.build_coupling_map()
    basis_gates, backend_gates = _basis(backend_target)
    target = Target.from_configuration(
        num_qubits=backend.num_qubits,
        coupling_map=coupling_map,
        basis_gates=basis_gates + list(custom_gates.keys()),
        custom_name_mapping={**backend_gates, **custom_gates},
    )
    block_names = _add_block_gates(target, coupling_map, block_members)

    # Fenced first: a placeholder is a single-qubit gate as far as the transpiler is
    # concerned, so without barriers it gets commuted past the others and the experiment
    # builder, which reads them in order, misreads the result.

    iqm_transpile = _transpiler_for(backend, use_iqm_transpiler)
    if iqm_transpile is not None:
        # IQM's own transpiler produces markedly shallower circuits than the generic
        # path, and it will accept the placeholders once they are barriers carrying
        # their name. What it must not be allowed to do to a circuit that still has
        # placeholders is listed in _IQM_ENFORCED, and asking for any of it is refused
        # rather than quietly ignored.
        _check_iqm_options(transpile_options)
        options: dict[str, object] = {
            name: value for name, (value, _why) in _IQM_ENFORCED.items()
        }
        options["optimization_level"] = optimization_level
        options.update(transpile_options or {})

        move_routing = bool(
            options.pop("perform_move_routing", is_resonator_backend(backend))
        )

        def route(subcircuit):
            marked = markers_to_barriers(subcircuit, marker_names)
            if move_routing:
                return move_route(marked, backend, iqm_transpile, **options)
            # Explicitly off, not merely absent: this transpiler defaults it on, and
            # then round-trips the circuit through IQM's own format, which is what
            # loses the labels and the classical registers.
            return _record_layout(
                iqm_transpile(marked, backend, perform_move_routing=False, **options),
                subcircuit.num_qubits,
            )

        transpiled = [
            barriers_to_markers(route(subcircuit), marker_names)
            for subcircuit in cut_circuit.subcircuits
        ]
        return CutCircuit(
            subcircuits=transpiled,
            cut_locations=cut_circuit.cut_locations,
            map_qubit=cut_circuit.map_qubit,
            options=cut_circuit.options,
            uncut_num_qubits=cut_circuit.uncut_num_qubits,
            backend=backend,
        )

    def qiskit_translate(subcircuit, spans_for):
        translated = transpile(
            _fence_markers(
                fuse_markers(
                    subcircuit,
                    spans_for,
                    lambda span: target.operation_from_name(span.name),
                ),
                marker_names | block_names,
            ),
            target=target,
            optimization_level=optimization_level,
            **(transpile_options or {}),
        )

        return split_markers(
            _drop_barriers(_record_layout(translated, subcircuit.num_qubits)),
            block_members,
        )

    transpiled = [
        _translate_subcircuit(
            subcircuit, spans.get(index, []), qiskit_translate, marker_names
        )
        for index, subcircuit in enumerate(cut_circuit.subcircuits)
    ]

    return CutCircuit(
        subcircuits=transpiled,
        cut_locations=cut_circuit.cut_locations,
        map_qubit=cut_circuit.map_qubit,
        options=cut_circuit.options,
        uncut_num_qubits=cut_circuit.uncut_num_qubits,
        backend=backend,
    )


def transpile_experiments(
    cut_experiment: CutExperiment,
    backend,
    optimization_level: int = 3,
    transpile_options: dict | None = None,
    use_iqm_transpiler: bool = True,
) -> CutExperiment:
    """
    Transpile experiment circuits. Transpiles all generated experiment circuits for
    a given backend. Most often one should use `transpile_subcircuits` instead, as that
    only transpiles subcircuits before experiment generation which is a lot more
    efficient. This function is mainly provided for special cases where one needs/wants
    extra control over the transpilation of experiment circuits.

    As with :func:`transpile_subcircuits`, an IQM backend is handed to IQM's own
    transpiler when the adapter is installed, with the same defaults inverted and for
    the same reasons. No placeholders are left at this point, so nothing has to be
    hidden from it as barriers -- and nothing has to give up IQM's single-qubit
    optimisation either, which is why this route produces the shallower circuits of the
    two, at the cost of transpiling every experiment circuit rather than each subcircuit
    once. The layout still has to be recorded afterwards.

    Args:
        cut_experiment: (CutExperiment): Experiment circuits to be transpiled.
        backend (str): Backend to transpile to.
        optimization_level (int): Optimization level for transpilation (0-3).
        transpile_options (dict): Arguments passed to the transpiler.
        use_iqm_transpiler (bool): Whether an IQM backend may use IQM's transpiler.
            Pass False for the ordinary qiskit path.

    Returns:
        CutExperiment: Transpiled experiment circuits wrapped in CutExperiment class.
    """

    if not isinstance(cut_experiment, CutExperiment):
        raise ValueError("cut_experiment must be of type CutExperiment.")

    translated = iter(
        transpile_circuits(
            [
                circuit
                for exps in cut_experiment.experiments
                for exp in exps
                for circuit in exp.values()
            ],
            backend,
            optimization_level,
            transpile_options,
            use_iqm_transpiler,
        )
    )
    subexperiments = [
        [{ind: next(translated) for ind in exp} for exp in exps]
        for exps in cut_experiment.experiments
    ]

    return CutExperiment(
        subexperiments,
        cut_locations=cut_experiment.cut_locations,
        map_qubit=cut_experiment.map_qubit,
        coefficients=cut_experiment.coefficients,
        observables=cut_experiment._observables,
        qubits=cut_experiment.qubits,
        options=cut_experiment.options,
        backend=backend,
        num_draws=cut_experiment._num_draws,
        plan=cut_experiment.plan,
        qpd_bits=cut_experiment.qpd_bits,
        gamma=cut_experiment.gamma,
        optimal_gamma=cut_experiment.optimal_gamma,
        can_reconstruct_probabilities=cut_experiment.can_reconstruct_probabilities,
        uncut_num_qubits=cut_experiment._uncut_num_qubits,
    )


def transpile_circuits(
    circuits: CutCircuit | CutExperiment | list[QuantumCircuit],
    backend,
    optimization_level: int = 3,
    transpile_options: dict | None = None,
    use_iqm_transpiler: bool = True,
):
    """Transpile for a backend, whatever stage the circuits have reached.

    A :class:`CutCircuit` goes to :func:`transpile_subcircuits`, which has placeholders
    to protect, and a :class:`CutExperiment` to :func:`transpile_experiments`, which
    does not. Plain circuits are transpiled as they are, which is what a backend given
    circuits rather than an experiment needs -- see
    :class:`~QCut.execution.parallel.ParallelBackend`.

    Args:
        circuits: a cut circuit, an experiment, or circuits with nothing left to hide.
        backend: backend to transpile to.
        optimization_level (int): optimization level for transpilation (0-3).
        transpile_options (dict): arguments passed to the transpiler.
        use_iqm_transpiler (bool): whether an IQM backend may use IQM's transpiler.

    Returns:
        The same kind of thing it was given, transpiled.
    """
    if isinstance(circuits, CutCircuit):
        return transpile_subcircuits(
            circuits,
            backend,
            optimization_level,
            transpile_options,
            use_iqm_transpiler,
        )
    if isinstance(circuits, CutExperiment):
        return transpile_experiments(
            circuits,
            backend,
            optimization_level,
            transpile_options,
            use_iqm_transpiler,
        )

    iqm_transpile = _transpiler_for(backend, use_iqm_transpiler)
    if iqm_transpile is not None:
        options = {
            "remove_final_rzs": False,
            "perform_move_routing": is_resonator_backend(backend),
            "optimization_level": optimization_level,
        }
        options.update(transpile_options or {})
        return [
            _record_layout(
                iqm_transpile(_drop_identities(circuit), backend, **options),
                circuit.num_qubits,
            )
            for circuit in circuits
        ]

    # An ideal simulator has no topology to build a target from, and needs none: it
    # takes the circuits as they are, so it is handed itself instead.
    target = None
    if has_topology(backend):
        backend_target = _target_of(backend)
        basis_gates, backend_gates = _basis(backend_target)
        target = Target.from_configuration(
            num_qubits=backend.num_qubits,
            coupling_map=backend_target.build_coupling_map(),
            basis_gates=basis_gates,
            custom_name_mapping=backend_gates,
        )

    return [
        _record_layout(
            transpile(
                circuit,
                backend=None if target is not None else backend,
                target=target,
                optimization_level=optimization_level,
                **(transpile_options or {}),
            ),
            circuit.num_qubits,
        )
        for circuit in circuits
    ]
