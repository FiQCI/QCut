"""
Building the experiment circuits: inserting a QPD term at every cut placeholder.
"""

from __future__ import annotations

import logging

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    Qubit,
)
from qiskit.converters import circuit_to_dag, dag_to_circuit

from QCut.errors.qcuterror import QCutError
from QCut.model.cutcircuit import CutCircuit, CutExperiment
from QCut.qpd.bundle import (
    SIDE_1,
    communication_waves,
    locate_placeholders,
    parse_placeholder,
    plan_bundles,
)
from QCut.qpd.qpd_locc import CommunicationPlan
from QCut.qpd.qpd_operations import (
    _insert_2qubit_gate_cut_qpd,
    _insert_bundle_qpd,
    _insert_wire_cut_qpd,
    coupling_filter,
    get_qpd_combinations,
    qpd_for_bundle,
    sample_qpd_combinations,
)
from QCut.transpilation.basis_transform import (
    _get_obs_subcircuits,
    _measurement_basis_ops,
)
from QCut.utils.backend import _device_basis
from QCut.utils.circuit import (
    _remove_obsm,
    _remove_obsm_2,
    compact_qpd_register,
)
from QCut.utils.observables import (
    ObservablesLike,
    _combine_pauli_ops,
    _validate_qubits,
    coerce_observables,
)

logger: logging.Logger = logging.getLogger(__name__)


def _finalize_subcircuit(
    subcircuit: QuantumCircuit, qpd_qubits: list[int]
) -> QuantumCircuit:
    """Finalize the subcircuit by measuring remaining qubits and decomposing."""

    # Transpiling puts a subcircuit's qubits on physical wires of the backend's
    # choosing, and _record_layout notes which wire holds which. Measuring through that
    # map, in the subcircuit's own qubit order, puts the bits where everything
    # downstream expects them without moving a single gate -- which matters, because the
    # placement is what makes the two-qubit gates land on pairs the device couples.
    #
    # Wires the layout never used, and wires routing borrowed, are absent from the map,
    # so nothing measures them.
    layout = (subcircuit.metadata or {}).get("qcut_layout")
    if layout is None:
        layout = list(range(subcircuit.num_qubits))

    # ``qcut_spent`` names the subcircuit's own qubits, which is what the observable
    # register was sized against. ``qpd_qubits`` names the wires the placeholders sat
    # on, and routing may have put two of them on one wire, so it cannot be counted.
    spent = (subcircuit.metadata or {}).get("qcut_spent")
    if spent is None:
        meas_qubits = [wire for wire in layout if wire not in qpd_qubits]
    else:
        meas_qubits = [wire for qubit, wire in enumerate(layout) if qubit not in spent]

    dag = circuit_to_dag(subcircuit)
    idle = list(dag.idle_wires())

    # The observable register by name. Positions are not fixed, because a register of
    # no bits is not created, so "the second one" is not reliably the right one.
    creg_to_use = next(
        (register for register in subcircuit.cregs if register.name == "meas"),
        subcircuit.cregs[0] if subcircuit.cregs else None,
    )
    if creg_to_use is None:
        return subcircuit

    for wire in idle:
        if not isinstance(wire, Qubit):
            continue
        index = subcircuit.find_bit(wire).index
        if index in meas_qubits and len(meas_qubits) > len(creg_to_use):
            meas_qubits.remove(index)

    if len(meas_qubits) == 0:
        return subcircuit

    subcircuit.measure(meas_qubits, creg_to_use)

    return subcircuit


def _get_placeholder_locations(subcircuits: list[QuantumCircuit]) -> list:
    """
    Identify the placeholder operations in a list of quantum subcircuits: the
    measurement ("Meas") and initialization ("Init") sides of wire cuts, and the halves
    of gate cuts. Only the instructions are kept, not their indices, because the
    experiment circuits are built from copies whose placeholders transpilation may have
    moved, and :func:`_build_experiment_circuit` finds them again by name.

    Args:
        subcircuits (list[QuantumCircuit]):
            A list of QuantumCircuit objects to be analyzed.
    Returns:
        list:
            A list of lists, where each sublist holds the placeholder instructions of
            the corresponding subcircuit, in circuit order.

    """
    ops = []
    for circ in subcircuits:
        subops = []
        for op in circ.data:
            name = op.operation.name
            if (
                name.startswith("Meas")
                or name.startswith("Init")
                or (name.startswith("cut") and "_" in name)
            ):
                subops.append(op)
        ops.append(subops)

    return ops


def _transpiled(circuit: QuantumCircuit, basis, cache: dict) -> QuantumCircuit:
    """Transpile a QPD operation once and reuse it across experiment groups.

    A bundle's operations are shared between groups, so without a cache each group would
    transpile them again. Caching also leaves the hand-written tables untouched, which
    transpiling in place did not.
    """
    key = id(circuit)
    hit = cache.get(key)
    if hit is not None and hit[0] is circuit:
        return hit[1]
    transpiled = transpile(circuit, basis_gates=basis)
    cache[key] = (circuit, transpiled)
    return transpiled


def _build_experiment_circuit(
    circ: QuantumCircuit,
    placeholder_ops: list,
    qpd,
    bundle_of_cut: dict,
    placeholders: dict,
) -> tuple[QuantumCircuit, tuple[int, int], list]:
    """Insert one group's QPD terms into one observable subcircuit.

    Every placeholder is replaced by the operation ``qpd`` chooses for its cut, the
    qubits left over are measured into the observable register, and the qpd bits the
    terms never write to are dropped.

    Args:
        circ (QuantumCircuit): the subcircuit with its observable basis change in place.
            It is left untouched, and the experiment circuit is built on a copy.
        placeholder_ops (list): its placeholder instructions, as
            :func:`_get_placeholder_locations` found them.
        qpd: the group's choice of term for every cut.
        bundle_of_cut (dict): the bundle each cut belongs to.
        placeholders (dict): where each placeholder sits, from
            :func:`QCut.qpd.bundle.locate_placeholders`.

    Returns:
        The experiment circuit, its ``(qpd bits written, qpd bits dropped)``, and a
        ``(bundle, clbits)`` pair for each communicating wire cut it measures.

    Raises:
        QCutError: a placeholder is missing from the subcircuit.
        ValueError: a placeholder is of no kind the builder knows.
    """
    subcircuit = circ.copy_empty_like()
    subcircuit._data = circ._data.copy(copy_instructions=False)
    offset = 0
    classical_bit_index = 0
    qpd_qubits = []  # store the qubit indices of qubits used for qpd
    # measurements
    labels_here: list = []  # (bundle, clbits) per communicating wire cut
    # Placeholders are found by name in the subcircuit as it stands, not by
    # the index they had when it was built. Transpiling against a backend
    # commutes them past each other, since each looks like an ordinary
    # single-qubit gate, and the walk below both reads them and accumulates
    # ``offset`` on the assumption that it meets them front to back. Every
    # placeholder name is unique within its subcircuit, so the lookup is
    # exact, and sorting by it restores the order the walk needs.
    position = {
        instruction.operation.name: index
        for index, instruction in enumerate(subcircuit.data)
    }
    placeholders_here = []
    for op in placeholder_ops:
        if op.operation.name not in position:
            raise QCutError(
                f"Placeholder '{op.operation.name}' is missing from its "
                "subcircuit. Transpilation is not allowed to remove or "
                "rename it, so the experiment cannot be built."
            )
        placeholders_here.append((position[op.operation.name], op))
    placeholders_here.sort(key=lambda pair: pair[0])

    for ind, op in placeholders_here:
        parsed = parse_placeholder(op.operation.name)
        bundle = bundle_of_cut.get(parsed[0]) if parsed is not None else None
        if bundle is not None and bundle.kind != "single":
            offset, classical_bit_index = _insert_bundle_qpd(
                ind,
                op,
                subcircuit,
                offset,
                qpd_qubits,
                qpd,
                bundle,
                placeholders,
                classical_bit_index,
                labels_here,
            )

        elif "cut" in op.operation.name:
            offset, classical_bit_index = _insert_2qubit_gate_cut_qpd(
                ind,
                op,
                subcircuit,
                offset,
                qpd_qubits,
                qpd,
                classical_bit_index,
            )

        elif "Meas" in op.operation.name or "Init" in op.operation.name:
            offset, classical_bit_index = _insert_wire_cut_qpd(
                ind,
                op,
                subcircuit,
                offset,
                qpd_qubits,
                qpd,
                classical_bit_index,
            )
        else:
            raise ValueError(
                f"""Unknown placeholder operation: {op.operation.name}.
                Actual operation: {subcircuit.data[ind + offset]}"""
            )

    subcircuit = _finalize_subcircuit(subcircuit, qpd_qubits)
    # Drop the qpd bits this term never writes to, so no circuit carries an
    # unused classical register, and remember how many went so the sign they
    # stood for can be put back during post-processing.
    subcircuit, dropped = compact_qpd_register(subcircuit)
    remaining = next((r.size for r in subcircuit.cregs if r.name == "qpd_meas"), 0)
    return subcircuit, (remaining, dropped), labels_here


def _check_measurement_spec(
    observables: ObservablesLike | None, qubits: list[int] | None
) -> None:
    """Check that exactly one of observables or qubits was given.

    Args:
        observables: the observables to estimate, if given.
        qubits (list[int]): the qubits to reconstruct a distribution over, if given.

    Raises:
        ValueError: neither was given, or both were.
    """
    if observables is not None and qubits is not None:
        raise ValueError("Only one of observables or qubits can be provided.")

    if observables is None and qubits is None:
        raise ValueError("Either observables or qubits must be provided.")


def get_experiment_circuits(  # noqa: C901
    cut_circuit: CutCircuit,
    observables: ObservablesLike | None = None,
    qubits: list[int] | None = None,
) -> CutExperiment:
    """Generate experiment circuits by inserting QPD operations on
    measure/initialize/cutCZ nodes.

    Args:
        cut_circuit (CutCircuit): The cut circuit to generate experiment circuits for.
        observables: the observables to estimate, taken the way qiskit's estimator
            takes them -- a label, a ``Pauli``, a ``SparsePauliOp``, a
            ``SparseObservable``, a ``{label: coefficient}`` mapping, or any nested
            sequence of those. The expectation values come back shaped like what is
            given here.
        qubits (list[int]): The qubits to measure.

        One of observables or qubits must be provided.

    Returns:
        CutExperiment: An object containing the generated experiment circuits and
        related information.

    """

    _check_measurement_spec(observables, qubits)

    if qubits is not None:
        _validate_qubits(cut_circuit.uncut_num_qubits, qubits)

    num_qubits = 0
    for subcircuit in cut_circuit.subcircuits:
        crs = subcircuit.cregs
        for cr in crs:
            if cr.name == "meas":
                num_qubits += cr.size

    # The circuits are built for the Pauli terms behind the observables, not for the
    # observables themselves: an observable is a weighted sum of terms, and separate
    # observables share terms, so measuring the distinct terms once covers all of them.
    # The weights that put the observables back together are kept on the spec, and
    # applied by the estimator. A ``qubits`` experiment builds no spec at all: its terms
    # are every Z string over those qubits, and there are ``2**k`` of them, which
    # nothing below needs. The experiment builds them on demand if it is ever asked.
    spec = None if qubits is not None else coerce_observables(observables, num_qubits)

    # Planning a bundle and building the experiment circuits both read where the
    # placeholders sit, and they have to read the same positions. Building the
    # observable subcircuits goes through a DAG, which re-serialises the instructions in
    # topological order, so a placeholder with nothing left on its own qubit can move
    # ahead of a gate that was written before it. Planning against the order as written
    # would then put a block where the builder cannot: it would sit before a gate that
    # one of the block's other wires still has to go through, and that wire would be
    # measured too early. Normalising first makes the two orders the same.
    # Everything below works on this copy, so the caller's cut circuit comes back
    # exactly as it went in.
    subcircuits = []
    for subcircuit in cut_circuit.subcircuits:
        normalised = dag_to_circuit(circuit_to_dag(subcircuit))

        normalised._layout = subcircuit.layout
        subcircuits.append(normalised)

    if qubits is not None:
        # Z strings all commute, so every one of them is covered by this single
        # setting so we can just use that directly.
        measurement_settings = [dict.fromkeys(qubits, "Z")]
    else:
        # _check_measurement_spec has already refused the case where neither the
        # observables nor the qubits were given, so there is a spec here.
        assert spec is not None
        measurement_settings = _combine_pauli_ops(spec.terms)

    if len(measurement_settings) > 1:
        logger.info(
            f"Found {len(measurement_settings)} conflicting observables. Extra"
            f" circuits will be generated to evaluate all expectation values."
        )

    # With a backend, the subcircuits were transpiled for it, so what goes into them
    # here has to be written in the device's gates too.
    backend = cut_circuit.backend
    basis = None if backend is None else _device_basis(backend)
    ops = None if basis is None else _measurement_basis_ops(basis)
    obs_subcircuits = _get_obs_subcircuits(subcircuits, measurement_settings, ops)

    _remove_obsm(obs_subcircuits)

    _remove_obsm_2(subcircuits)

    # Enumerating every combination costs the product of the per-bundle term counts, so
    # past a threshold the decomposition is sampled instead. Both paths must agree with
    # qpd_for_bundle on the term counts, or the coefficients and the combinations would
    # not line up.
    options = cut_circuit.options
    # A bundle wider than one cut has two-qubit gates in its terms, and those go in
    # after transpilation, so they are only runnable where the device couples the wires
    # the cuts landed on. Vetoing during planning lets a block too wide to route fall
    # back to narrower blocks instead of to no bundling at all.
    bundles = plan_bundles(
        cut_circuit.cut_locations,
        subcircuits,
        options,
        fits=coupling_filter(
            cut_circuit.cut_locations, subcircuits, cut_circuit.backend
        ),
    )
    bundle_of_cut = {cut: bundle for bundle in bundles for cut in bundle.cuts}
    placeholders = locate_placeholders(subcircuits)
    exact_groups = 1
    for bundle in bundles:
        exact_groups *= len(qpd_for_bundle(bundle, cut_circuit.cut_locations))

    if options.should_sample(exact_groups):
        if options.seed is None:
            # Drawn here and kept on the experiment, so it can be rebuilt exactly.
            options = options.replace(seed=int(np.random.default_rng().integers(2**32)))
        qpd_combinations, coefficients, num_draws = sample_qpd_combinations(
            cut_circuit.cut_locations, options.sample_count, options.seed, bundles
        )
        logger.info(
            f"Sampling {num_draws} draws over {exact_groups} possible groups, "
            f"giving {len(qpd_combinations)} distinct circuit groups."
        )
    else:
        qpd_combinations = get_qpd_combinations(cut_circuit.cut_locations, bundles)
        coefficients = np.empty(exact_groups)
        num_draws = None
        logger.debug(
            "expanding %d bundle(s) over %d cut(s) into %d experiment group(s)",
            len(bundles),
            len(cut_circuit.cut_locations),
            exact_groups,
        )

    experiment_circuits = []
    transpile_cache: dict[int, tuple[QuantumCircuit, QuantumCircuit]] = {}
    # Which classical bits carry a communicating wire cut's measured outcome, keyed by
    # (group, observable setting, subcircuit). Execution reads the label back from it.
    label_clbits: dict[tuple[int, int, int], list] = {}
    # (group, observable set, subcircuit) -> (qpd bits written, qpd bits dropped)
    qpd_bits: dict[tuple[int, int, int], tuple[int, int]] = {}
    communicating = [bundle for bundle in bundles if bundle.kind == "cc_wire"]
    group_labels: list[dict] = []
    group_keys: list[tuple] = []
    placeholder_locations = _get_placeholder_locations(subcircuits)
    for id_meas_experiment_index, qpd in enumerate(
        qpd_combinations
    ):  # loop through all
        # QPD combinations
        if num_draws is None:
            coefficients[id_meas_experiment_index] = np.prod([op["c"] for op in qpd])

        if communicating:
            group_labels.append(
                {bundle: qpd[bundle.anchor]["label"] for bundle in communicating}
            )
            # A communicating bundle contributes only its channel, so groups differing
            # only in the label they answer share a key. Every other bundle contributes
            # its whole choice, or a group would be handed operations meant for another.
            group_keys.append(
                tuple(
                    (id(bundle), qpd[bundle.anchor]["channel"])
                    if bundle.kind == "cc_wire"
                    else (
                        id(bundle),
                        id(qpd[bundle.anchor]["op_0"]),
                        id(qpd[bundle.anchor]["op_1"]),
                    )
                    for bundle in bundles
                )
            )

        if basis is not None:
            qpd = tuple(
                sub
                if sub["op_0"] is None
                else {
                    **sub,
                    "op_0": _transpiled(sub["op_0"], basis, transpile_cache),
                    "op_1": _transpiled(sub["op_1"], basis, transpile_cache),
                }
                for sub in qpd
            )

        # circuits
        obs_set_circuits = []
        for obs_index, obs_set in enumerate(obs_subcircuits):
            cur_set_circuits = {}
            for id_meas_subcircuit_index, circ in obs_set.items():
                subcircuit, bits, labels_here = _build_experiment_circuit(
                    circ,
                    placeholder_locations[id_meas_subcircuit_index],
                    qpd,
                    bundle_of_cut,
                    placeholders,
                )
                key = (id_meas_experiment_index, obs_index, id_meas_subcircuit_index)
                qpd_bits[key] = bits
                cur_set_circuits[id_meas_subcircuit_index] = subcircuit
                if labels_here:
                    label_clbits[key] = labels_here
            obs_set_circuits.append(cur_set_circuits)
        experiment_circuits.append(obs_set_circuits)

    plan = None
    if communicating:
        waves, bundle_waves = communication_waves(
            bundles, placeholders, len(subcircuits)
        )
        plan = CommunicationPlan(
            label_clbits,
            group_labels,
            group_keys,
            frozenset(
                placeholders[(cut, SIDE_1)].subcircuit
                for bundle in communicating
                for cut in bundle.cuts
            ),
            waves,
            bundle_waves,
        )
    cut_experiment = CutExperiment(
        experiment_circuits=experiment_circuits,
        cut_locations=cut_circuit.cut_locations,
        map_qubit=cut_circuit.map_qubit,
        coefficients=coefficients,
        observables=spec,
        qubits=qubits,
        can_reconstruct_probabilities=qubits is not None,
        backend=backend,
        options=options,
        num_draws=num_draws,
        plan=plan,
        qpd_bits=qpd_bits,
        gamma=cut_circuit.gamma,
        optimal_gamma=cut_circuit.optimal_gamma,
        uncut_num_qubits=cut_circuit.uncut_num_qubits,
    )

    logger.info(f"Generated {cut_experiment.num_circuits} circuits for the experiment.")

    return cut_experiment
