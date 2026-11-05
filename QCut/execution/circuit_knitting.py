"""
A module for the main circuit knitting workflow.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field, replace

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    Qubit,
)
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.primitives import BaseSamplerV2

from QCut.errors.qcuterror import QCutError
from QCut.job.cutresult import CircuitResult, RawResult
from QCut.job.handle import QCutJob, _Batch, _job_id
from QCut.model.cutcircuit import CutCircuit, CutExperiment
from QCut.options import CutOptions
from QCut.post_process.postprocess import estimate_expectation_values
from QCut.post_process.probabilities import (
    QuasiProbabilities,
    estimate_probabilities,
)
from QCut.pre_process.circuit_preparation import get_locations_and_subcircuits
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
)
from QCut.transpilation.transpile_cut_circuits import (
    has_topology,
    transpile_subcircuits,
)
from QCut.utils.backend import _backend_gate_names, _backend_shot_cap
from QCut.utils.circuit import (
    _remove_obsm,
    _remove_obsm_2,
    circuit_fingerprint,
    compact_qpd_register,
)
from QCut.utils.deprecation import deprecated
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


def _has_measurements(circuit: QuantumCircuit) -> bool:
    return "measure" in circuit.count_ops()


def _get_placeholder_locations(subcircuits: list[QuantumCircuit]) -> list:
    """
    Identify the locations of placeholder operations in a list of quantum subcircuits.
    This function scans through each quantum subcircuit provided in the input list and
    identifies the indices and operations where either measurement ("Meas") or
    initialization ("Init") operations occur. It returns a list of lists, where each
    sublist corresponds to  a subcircuit and contains tuples of the
    form (index, operation).

    Args:
        subcircuits (list[QuantumCircuit]):
            A list of QuantumCircuit objects to be analyzed.
    Returns:
        list:
            A list of lists, where each sublist contains tuples (index, operation)
            indicating the positions of measurement or initialization operations in
            the corresponding subcircuit.

    """
    ops = []
    for circ in subcircuits:
        subops = []
        for ind, op in enumerate(circ.data):
            name = op.operation.name
            if (
                name.startswith("Meas")
                or name.startswith("Init")
                or (name.startswith("cut") and "_" in name)
            ):
                subops.append((ind, op))
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

    check_circuit_type = cut_circuit.backend is not None

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

    backend = None
    if check_circuit_type:
        backend = cut_circuit.backend
        basis = _backend_gate_names(backend)
        basis = ["r" if gate == "prx" else gate for gate in basis]
        # A device's own gate list can name operations qiskit does not know, and it
        # refuses those through ``basis_gates`` rather than ignoring them. IQM's
        # resonator machines list ``move``, which shifts a state between a qubit and a
        # resonator, so a Deneb-class backend used to fail here. The operations being
        # translated below are single-qubit basis changes, so anything qiskit cannot
        # name is of no use to them and is dropped.
        known = set(get_standard_gate_name_mapping())
        known.update(("measure", "reset", "delay", "barrier", "id"))
        dropped = [gate for gate in basis if gate not in known]
        if dropped:
            logger.debug(
                f"ignoring backend gate(s) {dropped}, which qiskit cannot take as a "
                "basis gate, while translating the observable basis changes"
            )
        basis = [gate for gate in basis if gate in known]

    obs_subcircuits = None

    if check_circuit_type:
        x_meas_ops = QuantumCircuit(1)
        x_meas_ops.h(0)
        x_meas_ops.name = "X-meas"
        x_meas_ops = transpile(x_meas_ops, basis_gates=basis)

        y_meas_ops = QuantumCircuit(1)
        y_meas_ops.sdg(0)
        y_meas_ops.h(0)
        y_meas_ops.name = "Y-meas"
        y_meas_ops = transpile(y_meas_ops, basis_gates=basis)

        ops = {"X-meas": x_meas_ops, "Y-meas": y_meas_ops}

        obs_subcircuits = _get_obs_subcircuits(subcircuits, measurement_settings, ops)
    else:
        obs_subcircuits = _get_obs_subcircuits(subcircuits, measurement_settings)

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

        if check_circuit_type:
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
        inserted_operations = 0
        obs_set_circuits = []
        for obs_set in obs_subcircuits:
            cur_set_circuits = {}
            for id_meas_subcircuit_index, circ in obs_set.items():
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
                for recorded_ind, op in placeholder_locations[id_meas_subcircuit_index]:
                    if op.operation.name not in position:
                        raise QCutError(
                            f"Placeholder '{op.operation.name}' is missing from its "
                            "subcircuit. Transpilation is not allowed to remove or "
                            "rename it, so the experiment cannot be built."
                        )
                    placeholders_here.append((position[op.operation.name], op))
                placeholders_here.sort(key=lambda pair: pair[0])

                for op_ind in placeholders_here:
                    ind, op = op_ind

                    parsed = parse_placeholder(op.operation.name)
                    bundle = (
                        bundle_of_cut.get(parsed[0]) if parsed is not None else None
                    )
                    if bundle is not None and bundle.kind != "single":
                        (
                            offset,
                            classical_bit_index,
                            inserted_operations,
                        ) = _insert_bundle_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            bundle,
                            placeholders,
                            classical_bit_index,
                            inserted_operations,
                            labels_here,
                        )

                    elif "cut" in op.operation.name:
                        (
                            offset,
                            classical_bit_index,
                            inserted_operations,
                        ) = _insert_2qubit_gate_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            classical_bit_index,
                            inserted_operations,
                        )

                    elif "Meas" in op.operation.name or "Init" in op.operation.name:
                        (
                            offset,
                            classical_bit_index,
                            inserted_operations,
                        ) = _insert_wire_cut_qpd(
                            ind,
                            op,
                            subcircuit,
                            offset,
                            qpd_qubits,
                            qpd,
                            classical_bit_index,
                            inserted_operations,
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
                remaining = next(
                    (r.size for r in subcircuit.cregs if r.name == "qpd_meas"), 0
                )
                qpd_bits[
                    (
                        id_meas_experiment_index,
                        len(obs_set_circuits),
                        id_meas_subcircuit_index,
                    )
                ] = (remaining, dropped)
                cur_set_circuits[id_meas_subcircuit_index] = subcircuit
                if labels_here:
                    label_clbits[
                        (
                            id_meas_experiment_index,
                            len(obs_set_circuits),
                            id_meas_subcircuit_index,
                        )
                    ] = labels_here
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


def _apply_communication(cut_experiment, results) -> None:
    """Record which outcome each group takes from its measuring run.

    The selection is attached to the group's own result rather than applied to it, since
    groups sharing a measuring circuit each keep a different label out of the same
    counts.
    """
    plan = cut_experiment.plan
    for (group, obs, sub), entries in plan.label_clbits.items():
        leaf = results[group][obs][sub]
        results[group][obs][sub] = replace(
            leaf,
            label_filter=leaf.label_filter
            + tuple(
                (clbits, plan.labels[group][bundle], 2**bundle.size)
                for bundle, clbits in entries
            ),
        )


def _takes_max_batch_size(run) -> bool:
    """Whether this ``run`` batches on its own account, so it can be told the size."""
    try:
        return "max_batch_size" in inspect.signature(run).parameters
    except (TypeError, ValueError):
        return False


def _submit(backend, circuits, shots, max_batch_size, run_options):
    """Run one batch, whether ``backend`` is a backend, a sampler, or a wrapper.

    Both hand back a job rather than a result, so either way the caller can submit
    everything before collecting any of it.

    A wrapper that batches on its own account is given the size QCut used, since this
    batch is already sized both for the device and for the shots its circuits asked for,
    and splitting it again would only add jobs. Anything else in ``run_options`` is
    passed through untouched.
    """
    options = dict(run_options or {})
    if isinstance(backend, BaseSamplerV2):
        return backend.run([(circuit,) for circuit in circuits], shots=shots, **options)
    if "max_batch_size" not in options and _takes_max_batch_size(backend.run):
        options["max_batch_size"] = max_batch_size
    return backend.run(circuits, shots=shots, **options)


def _submit_batches(job: QCutJob, wave: int, planned) -> None:
    """Submit batches to the job's target, every one before any is collected.

    ``planned`` holds ``(circuits, targets, shots, scale)`` per batch. A target can
    refuse a batch after accepting the ones before it, and raising there would lose the
    jobs already queued. The refusal is kept on its batch instead, nothing after it is
    submitted, and the job raises it when its results are asked for.
    """
    refused = False
    for circuits, targets, shots, scale in planned:
        batch = _Batch(wave, targets, shots, scale)
        if not refused and not job._cancelled.is_set():
            try:
                batch.job = _submit(
                    job.backend,
                    circuits,
                    shots,
                    job.max_batch_size,
                    job._run_options,
                )
            except Exception as error:  # noqa: BLE001 - kept on the batch, raised later
                refused = True
                batch.error = error
                logger.warning(
                    f"The target refused a batch of {len(circuits)} circuits: {error}. "
                    "Nothing after it is submitted, and the job raises this when its "
                    "results are asked for."
                )
            else:
                logger.info(
                    f"Submitted {len(circuits)} circuits at {shots} shots each, job "
                    f"{_job_id(batch.job) or 'unknown'}"
                )
        job._add(batch)
        # Cancelled while this one was going out, so it missed the sweep.
        if job._cancelled.is_set():
            batch.cancel()


def _seeded(cut_experiment) -> list[list[dict[int, CircuitResult]]]:
    """An empty result per slot, ready for the batches to fill in.

    Seeded in subcircuit order, because the estimator reads a group's subcircuits back
    in the order these were filled, and both deduplicated circuits and the waves fill
    them out of order.
    """
    return [
        [{sub: CircuitResult({}) for sub in sorted(obs_group)} for obs_group in group]
        for group in cut_experiment.experiments
    ]


#: Widest ratio of requested shot counts allowed to share one job. Circuits in a batch
#: all run at the same number of shots, so a batch spanning a wide range starves the
#: circuits at its top end. Holding the ratio to two keeps the variance cost of that
#: within a few percent while still leaving only a handful of batches.
SHOT_SPREAD: float = 2.0


#: Shots per circuit when the caller does not say. Used by :func:`submit_experiments`
#: and every wrapper around it, so they cannot drift apart.
DEFAULT_SHOTS: int = 2**12


#: Share of the shot budget spent on the measuring wave. A measuring circuit is shared
#: between every group that differs only in which label it answers, so a shot spent
#: there buys precision for all of them at once, while a preparing shot buys it for one
#: group alone. Splitting the budget evenly over the subcircuits therefore over-funds
#: the measuring side. Scans at one, two and three wires, over cuts into two and three
#: pieces, all put the best share near a sixth, independent of the block width and of
#: how many pieces the circuit was cut into.
MEASURE_SHARE: float = 1 / 6


def _batches(runnable, max_batch_size):
    """Group jobs, already sorted by requested shots, into runs that can share a job.

    A batch closes when it is full or when the next circuit wants more than
    :data:`SHOT_SPREAD` times what the batch's smallest asked for. Both limits matter.
    Without the size limit a batch could exceed what the backend takes in one job, and
    without the spread limit a generous ``max_batch_size`` would put everything in one
    batch at one shot count, which is uniform allocation and throws away the whole point
    of splitting the shots in proportion to the labels.
    """
    batch = []
    for job in runnable:
        if batch and (
            len(batch) >= max_batch_size or job[2] > batch[0][2] * SHOT_SPREAD
        ):
            yield batch
            batch = []
        batch.append(job)
    if batch:
        yield batch


def _batch_shots(batch, cap) -> int:
    """The shots a batch runs at: the mean of what its own circuits asked for."""
    shots = max(1, round(sum(job[2] for job in batch) / len(batch)))
    return min(shots, cap) if cap is not None else shots


def _dispatch(job: QCutJob, wave: int, jobs, cap, results) -> None:
    """Submit one wave, and fill in the circuits with nothing to measure.

    Circuits in a wave no longer want the same number of shots, and only circuits asking
    for the same number can share a job. They are sorted and grouped by
    :func:`_batches`, and each batch runs at the mean of what its own circuits asked
    for. Sorting first is what keeps that mean close to every request in the batch.

    Each batch carries the factor that rescales its counts to the job's nominal shots
    when they are collected, so that everything downstream can keep dividing by the one
    number it was given.
    """
    job._wave = wave
    runnable = []
    for targets, circuit, wanted in jobs:
        if not _has_measurements(circuit):
            synthetic = {" " + "0" * circuit.num_clbits: job.shots}
            for group, obs, sub in targets:
                results[group][obs][sub] = CircuitResult(dict(synthetic))
            continue
        runnable.append((targets, circuit, wanted))

    runnable.sort(key=lambda entry: entry[2])

    planned = []
    for batch in _batches(runnable, job.max_batch_size):
        shots = _batch_shots(batch, cap)
        planned.append(
            (
                [circuit for _t, circuit, _w in batch],
                [targets for targets, _c, _w in batch],
                shots,
                job.shots / shots,
            )
        )
    _submit_batches(job, wave, planned)


def _label_fraction(counts: dict, clbits, label) -> float:
    """Return how often a measuring run came out with this label."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    kept = sum(
        value
        for key, value in counts.items()
        if all(
            key.split(" ")[-1][len(key.split(" ")[-1]) - 1 - clbit] == str(wanted)
            for clbit, wanted in zip(clbits, label)
        )
    )
    return kept / total


def _label_weights(cut_experiment, results, wave: int) -> list[float]:
    """Return how likely each group's path was, using the waves already run.

    A bundle's outcome is only known once its preparing side has been reached, so at
    wave ``w`` the weight is the product over the bundles settled by then. That is just
    the information the protocol itself has at that point.
    """
    plan = cut_experiment.plan
    settled = {bundle for bundle, at in plan.bundle_waves.items() if at <= wave}
    seen: dict[tuple[int, object], float] = {}
    for (group, obs, sub), entries in plan.label_clbits.items():
        for bundle, clbits in entries:
            if bundle not in settled or (group, bundle) in seen:
                continue
            seen[(group, bundle)] = _label_fraction(
                results[group][obs][sub].raw_counts(),
                clbits,
                plan.labels[group][bundle],
            )
    weights = []
    for group in range(len(cut_experiment.experiments)):
        weight = 1.0
        for bundle in plan.labels[group]:
            if bundle in settled:
                weight *= seen.get((group, bundle), 0.0)
        weights.append(weight)
    return weights


def _allocate(weights: list[float], total: int) -> list[int]:
    """Split ``total`` shots between groups in proportion to ``weights``.

    Proportional allocation is what minimises the variance of the sum, and it is the
    whole point of communicating: a label that rarely comes up needs correspondingly few
    shots spent on the state it asks for. A group whose label did come up must still get
    at least one shot, or its contribution goes missing and the estimate is biased, so
    the split uses largest remainders on top of a floor of one.
    """
    live = [index for index, weight in enumerate(weights) if weight > 0]
    allocation = [0] * len(weights)
    if not live:
        return allocation
    spare = max(total - len(live), 0)
    scale = sum(weights[index] for index in live)
    exact = [spare * weights[index] / scale for index in live]
    for position, index in enumerate(live):
        allocation[index] = 1 + int(exact[position])
    remainder = spare - sum(int(value) for value in exact)
    order = sorted(range(len(live)), key=lambda p: -(exact[p] % 1))
    for position in order[: max(remainder, 0)]:
        allocation[live[position]] += 1
    return allocation


def _first_wave_jobs(cut_experiment, shots, sharing, scale=1.0):
    """Return the jobs for wave zero, one per distinct label-independent circuit.

    ``scale`` is :data:`MEASURE_SHARE` rewritten as a per-subcircuit factor, so that
    the measuring wave takes that share of the run rather than an even one.
    """
    plan = cut_experiment.plan
    experiments = cut_experiment.experiments
    jobs = []
    for members in sharing.values():
        budget = max(1, round(shots * len(members) * scale))
        for obs, obs_group in enumerate(experiments[members[0]]):
            for sub, circuit in obs_group.items():
                if plan.waves.get(sub, 0) != 0:
                    continue
                jobs.append(([(group, obs, sub) for group in members], circuit, budget))
    return jobs


def _later_wave_jobs(cut_experiment, wave, allocation, results):
    """Return one wave's jobs, blanking groups whose path never came up."""
    plan = cut_experiment.plan
    jobs = []
    for group, obs_groups in enumerate(cut_experiment.experiments):
        for obs, obs_group in enumerate(obs_groups):
            for sub, circuit in obs_group.items():
                if plan.waves.get(sub, 0) != wave:
                    continue
                if allocation[group] <= 0:
                    results[group][obs][sub] = CircuitResult({})
                else:
                    jobs.append(([(group, obs, sub)], circuit, allocation[group]))
    return jobs


def _submit_communicating(job: QCutJob) -> None:
    """Submit an experiment whose wire cuts exchange their measured outcome.

    The state one side prepares depends on what the other measured, so the circuits run
    in waves. Wave zero holds everything no label can affect, and its circuits are
    shared between the groups differing only in which label they answer, so they run
    once for the whole set. Each later wave has its shots split in proportion to how
    often the labels it depends on actually came up.

    The budget is not divided evenly between the waves. Sharing makes a measuring shot
    worth more than a preparing one, so wave zero takes :data:`MEASURE_SHARE` of the run
    and the later waves divide the rest. The total is unchanged either way.

    Wave zero is submitted here, so the job has ids to report as soon as it is handed
    back. Every later wave needs the one before it collected first, so those are left
    to a thread of the job's own, which submits each as soon as it can rather than when
    someone next polls.

    TODO: the waves after the first do not share circuits at all. ``_later_wave_jobs``
    gives every group a job of its own, where :func:`_deduplicate` would collapse the
    groups holding the same circuit as wave zero's ``plan.keys`` already does. It is not
    the same change as the plain path's, because each group carries its own shot
    allocation by then, so the shared job would have to run at their sum and be scaled
    back per group.
    """
    cut_experiment = job.experiment
    shots = job.shots
    plan = cut_experiment.plan
    results = _seeded(cut_experiment)
    cap = _backend_shot_cap(job.backend)

    # Budget-neutral reweighting of the waves. Every subcircuit would otherwise get
    # shots * groups whatever wave it fell in, which hands the measuring wave a share
    # of one over the number of pieces the circuit was cut into. Rewriting
    # MEASURE_SHARE as a per-subcircuit factor leaves the run's total untouched, so
    # ``shots`` keeps meaning what it always did.
    subcircuits = list(cut_experiment.experiments[0][0])
    measuring = sum(1 for sub in subcircuits if plan.waves.get(sub, 0) == 0)
    preparing = len(subcircuits) - measuring
    first_scale = MEASURE_SHARE * len(subcircuits) / measuring
    later_scale = (
        (1 - MEASURE_SHARE) * len(subcircuits) / preparing if preparing else 0.0
    )

    sharing: dict[tuple, list[int]] = {}
    for group in range(len(cut_experiment.experiments)):
        sharing.setdefault(plan.keys[group], []).append(group)

    logger.info(
        f"Wave 1 of {plan.last_wave + 1}: {len(sharing)} distinct measuring run(s) "
        f"shared across {len(cut_experiment.experiments)} group(s)."
    )
    _dispatch(
        job,
        0,
        _first_wave_jobs(cut_experiment, shots, sharing, first_scale),
        cap,
        results,
    )

    def drive():
        job._collect(results, 0)
        for wave in range(1, plan.last_wave + 1):
            if job._cancelled.is_set():
                raise QCutError(f"cancelled before wave {wave + 1}")
            weights = _label_weights(cut_experiment, results, wave)
            allocation = _allocate(
                weights, round(shots * len(cut_experiment.experiments) * later_scale)
            )
            logger.info(
                f"Wave {wave + 1} of {plan.last_wave + 1}: preparing circuits for "
                f"{sum(1 for value in allocation if value > 0)} of "
                f"{len(cut_experiment.experiments)} group(s), shots split in "
                "proportion to how often each label came up."
            )
            _dispatch(
                job,
                wave,
                _later_wave_jobs(cut_experiment, wave, allocation, results),
                cap,
                results,
            )
            job._collect(results, wave)

        _apply_communication(cut_experiment, results)
        return results

    job._start(drive, background=True)


@dataclass(frozen=True)
class JobEstimate:
    """One ``run`` call: how many circuits it carries, at how many shots each."""

    circuits: int
    shots: int


@dataclass(frozen=True)
class RunEstimate:
    """What running an experiment will cost, before anything is submitted."""

    breakdown: dict[str, JobEstimate] = field(default_factory=dict)
    exact: bool = True

    @property
    def jobs(self) -> int:
        """How many ``run`` calls the experiment takes."""
        return len(self.breakdown)

    @property
    def circuits(self) -> int:
        """How many circuits are submitted, over every job."""
        return sum(job.circuits for job in self.breakdown.values())

    @property
    def shots(self) -> int:
        """How many shots the experiment spends in total."""
        return sum(job.circuits * job.shots for job in self.breakdown.values())

    def __str__(self) -> str:
        """Format string."""
        line = (
            f"Experiment will run {self.circuits} circuits in {self.jobs} jobs with a "
            f"total of {self.shots} shots."
        )
        if not self.exact:
            line += (
                " Past the first wave the shots each group gets depend on what the wave"
                " before it measured. The circuit count is the most it can submit, the"
                " job count the fewest it can take, since circuits wanting very"
                " different shot counts cannot share a job, and the shot total is right"
                " to rounding."
            )
        return f"{line} See the returned object for the breakdown."

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


def _named(jobs: list[JobEstimate]) -> dict[str, JobEstimate]:
    return {f"job{index}": job for index, job in enumerate(jobs, start=1)}


def _estimated_batches(jobs, max_batch_size, cap) -> list[JobEstimate]:
    """Batch a wave's jobs the way :func:`_dispatch` will, without running them."""
    runnable = [job for job in jobs if _has_measurements(job[1])]
    runnable.sort(key=lambda job: job[2])
    return [
        JobEstimate(len(batch), _batch_shots(batch, cap))
        for batch in _batches(runnable, max_batch_size)
    ]


def _projected_wave_jobs(cut_experiment, wave: int, shots: int):
    """A later wave's jobs as they would be if every group got an equal share."""
    plan = cut_experiment.plan
    return [
        ([(group, obs, sub)], circuit, shots)
        for group, obs_groups in enumerate(cut_experiment.experiments)
        for obs, obs_group in enumerate(obs_groups)
        for sub, circuit in obs_group.items()
        if plan.waves.get(sub, 0) == wave
    ]


#: One slot of an experiment: which group, which measurement setting, which subcircuit.
Slot = tuple[int, int, int]


def _runnable(cut_experiment) -> tuple[list, list]:
    """Split an experiment's slots into the ones worth submitting and the ones not.

    A QPD term that measures nothing on a subcircuit leaves a circuit with no
    measurements at all. There is nothing to run, and its counts are known: every shot
    reads zero. Those are filled in directly rather than submitted.

    Returns:
        The slots to submit, as ``(slot, circuit)``, and the empty ones, as
        ``(slot, num_clbits)``. Both in group, setting, subcircuit order.
    """
    runnable: list[tuple[Slot, QuantumCircuit]] = []
    empty: list[tuple[Slot, int]] = []
    for group_idx, circuit_group in enumerate(cut_experiment.experiments):
        for obs_idx, obs_group in enumerate(circuit_group):
            for sub_idx, subcircuit in obs_group.items():
                slot = (group_idx, obs_idx, sub_idx)
                if _has_measurements(subcircuit):
                    runnable.append((slot, subcircuit))
                else:
                    empty.append((slot, subcircuit.num_clbits))
    return runnable, empty


def _deduplicate(runnable: list, enabled: bool) -> list[tuple[list, QuantumCircuit]]:
    """Group slots filled by the same circuit, so it is only submitted once.

    A subcircuit's experiment circuit depends on the QPD terms of the cuts touching that
    subcircuit and on the measurement setting restricted to its own qubits, neither of
    which most of an experiment's groups change. The slot count grows exponentially in
    the number of pieces and the distinct count linearly, so almost all of a large run
    is the same handful of circuits submitted over and over.

    The key carries the subcircuit index as well as the fingerprint. The estimator takes
    one factor per subcircuit from each group and multiplies them, so two factors of a
    single product sharing a sample would give ``E[X**2]`` where ``E[X]**2`` is meant.
    Keying on the subcircuit makes that impossible, since a group holds each subcircuit
    once. Sharing between groups is a different matter: groups are summed rather than
    multiplied, so it leaves the estimate unbiased and moves only its variance.

    Two subcircuits of a symmetric circuit really can produce identical experiment
    circuits, so this is not a theoretical concern.

    Returns:
        ``(slots, circuit)`` per distinct circuit, in the order first seen, so two runs
        of one experiment submit the same circuits in the same order.
    """
    if not enabled:
        return [([slot], circuit) for slot, circuit in runnable]

    cache: dict = {}
    distinct: dict[tuple, tuple[list, QuantumCircuit]] = {}
    for slot, circuit in runnable:
        key = (slot[2], circuit_fingerprint(circuit, cache))
        hit = distinct.get(key)
        if hit is None:
            distinct[key] = ([slot], circuit)
        else:
            hit[0].append(slot)
    return list(distinct.values())


def _submission(cut_experiment) -> tuple[list[tuple[list, QuantumCircuit]], list]:
    """What a run of this experiment actually submits, and what it fills in itself."""
    runnable, empty = _runnable(cut_experiment)
    jobs = _deduplicate(runnable, cut_experiment.options.deduplicate)
    if len(jobs) < len(runnable):
        logger.info(
            f"{len(runnable)} experiment circuits hold {len(jobs)} distinct ones"
        )
    return jobs, empty


def estimate_run(
    cut_experiment: CutExperiment,
    shots: int = DEFAULT_SHOTS,
    backend=None,
    max_batch_size: int = 100,
) -> RunEstimate:
    """Work out what :func:`submit_experiments` will cost, without submitting anything.

    Exact for an experiment that runs in one go. One whose wire cuts communicate runs in
    waves, and every wave after the first spends its shots in proportion to what the one
    before it measured, so only that first wave is known in advance. The rest is worked
    out as if every group got an equal share. That gives the most circuits a wave can
    submit, since a group whose label never came up submits none, and the fewest jobs it
    can take, since an uneven share puts circuits wanting very different shot counts in
    jobs of their own.

    Args:
        cut_experiment (CutExperiment): the experiment circuits that would be run.
        shots (int): shots per circuit, as passed to :func:`submit_experiments`.
        backend: the backend or sampler that would run it, read only for the most shots
            it takes in one job.
        max_batch_size (int): maximum number of circuits submitted per ``run`` call.

    Returns:
        RunEstimate: the circuits, jobs and shots the run takes, and a breakdown by job.

    """
    cap = _backend_shot_cap(backend) if backend is not None else None

    if cut_experiment.plan is None:
        # Counted through the same pair the run uses, so an estimate cannot drift from
        # what is submitted.
        jobs, _empty = _submission(cut_experiment)
        return RunEstimate(
            _named(
                [
                    JobEstimate(len(jobs[start : start + max_batch_size]), shots)
                    for start in range(0, len(jobs), max_batch_size)
                ]
            ),
            exact=True,
        )

    plan = cut_experiment.plan
    groups = len(cut_experiment.experiments)
    subcircuits = list(cut_experiment.experiments[0][0])
    measuring = sum(1 for sub in subcircuits if plan.waves.get(sub, 0) == 0)
    preparing = len(subcircuits) - measuring
    first_scale = MEASURE_SHARE * len(subcircuits) / measuring
    later_scale = (
        (1 - MEASURE_SHARE) * len(subcircuits) / preparing if preparing else 0.0
    )

    sharing: dict[tuple, list[int]] = {}
    for group in range(groups):
        sharing.setdefault(plan.keys[group], []).append(group)

    estimates = _estimated_batches(
        _first_wave_jobs(cut_experiment, shots, sharing, first_scale),
        max_batch_size,
        cap,
    )
    for wave in range(1, plan.last_wave + 1):
        even = max(1, round(round(shots * groups * later_scale) / groups))
        estimates += _estimated_batches(
            _projected_wave_jobs(cut_experiment, wave, even), max_batch_size, cap
        )

    return RunEstimate(_named(estimates), exact=plan.last_wave == 0)


@deprecated(
    "submit_experiments() is planned to be removed in QCut 3.0. run_experiments() will "
    "return the QCutJob itself, and the job's result() will give expectation values "
    "or a distribution rather than raw counts. The raw counts will stay available on "
    "the job."
)
def submit_experiments(
    cut_experiment: CutExperiment,
    shots: int = DEFAULT_SHOTS,
    backend=None,
    max_batch_size: int = 100,
    run_options: dict | None = None,
) -> QCutJob:
    """Submit experiment circuits and return without waiting for their results.

    Every batch is submitted before this returns, and the returned job collects them
    when its :meth:`QCutJob.result` is asked for. An experiment whose wire cuts
    communicate runs in waves: the first is submitted before this returns, and the
    job submits the rest itself as each wave before them comes back.

    Args:
        cut_experiment (CutExperiment): the experiment circuits to run
        shots (int): number of shots per circuit run (optional). Communicating wire
            cuts spend it differently: the waves split a total of ``shots`` times the
            number of groups per subcircuit between them, in proportion to how often
            each label came up, so individual circuits run at very different counts and
            only that total is fixed. See :func:`_submit_communicating`.
        backend: backend or V2 sampler used for running the circuits.
            A sampler is run through its own interface and its results are read
            per register, so the circuits must already be in its target's basis.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call. Larger batches reduce per-job overhead on real hardware. A target that
            batches on its own account, as e.g. fiqci-ems does, is given this size
            too, so it does not split a batch QCut has already sized.
        run_options (dict): passed on to every ``run`` call. Use it for whatever the
            target takes per run, since QCut sets only ``shots`` itself.

    Returns:
        QCutJob: a handle on the submitted experiment, whose :meth:`QCutJob.result`
        returns what :func:`run_experiments` does.

    """
    if backend is None:
        raise QCutError("a backend or sampler to run the circuits on is needed")

    plan = cut_experiment.plan
    job = QCutJob(
        cut_experiment,
        backend,
        shots,
        max_batch_size,
        run_options,
        num_waves=1 if plan is None else plan.last_wave + 1,
    )

    if plan is not None:
        _submit_communicating(job)
        return job

    jobs, empty_locations = _submission(cut_experiment)

    num_batches = len(jobs) // max_batch_size + (1 if len(jobs) % max_batch_size else 0)
    logger.info(
        f"Running {len(jobs)} circuits on the backend {backend} with {shots} shots each"
    )
    logger.info(
        "Circuits will be split into "
        f"{num_batches}"
        f" batches of size {max_batch_size} for execution."
    )
    _submit_batches(
        job,
        0,
        [
            (
                [circuit for _slots, circuit in batch],
                [slots for slots, _circuit in batch],
                shots,
                1.0,
            )
            for batch in (
                jobs[start : start + max_batch_size]
                for start in range(0, len(jobs), max_batch_size)
            )
        ],
    )

    def collect():
        results = _seeded(cut_experiment)
        job._collect(results)
        for (group_idx, obs_idx, sub_idx), num_clbits in empty_locations:
            results[group_idx][obs_idx][sub_idx] = CircuitResult(
                {" " + "0" * num_clbits: shots}
            )
        return results

    job._start(collect, background=False)
    return job


@deprecated(
    "run_experiments() is planned to change in QCut 3.0: it will return a QCutJob "
    "without waiting, as submit_experiments() does now, and the job's result() will "
    "give expectation values or a distribution rather than a RawResult. The RawResult "
    "will stay available on the job."
)
def run_experiments(
    cut_experiment: CutExperiment,
    shots: int = DEFAULT_SHOTS,
    backend=None,
    max_batch_size: int = 100,
    run_options: dict | None = None,
) -> RawResult:
    """Run experiment circuits and wait for their results.

    The same as :func:`submit_experiments` followed by :meth:`QCutJob.result`. The
    counts come back as ``[group][observable][subcircuit]``, each key ``"xxx yy"``
    where ``xxx`` is the end-of-circuit measurements on the ``meas`` register and
    ``yy`` the QPD basis measurements on the ``qpd_meas`` register.

    Args:
        cut_experiment (CutExperiment): the experiment circuits to run
        shots (int): number of shots per circuit run (optional), as in
            :func:`submit_experiments`.
        backend: backend or V2 sampler used for running the circuits.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call, as in :func:`submit_experiments`.
        run_options (dict): passed on to every ``run`` call.

    Returns:
        RawResult:
            the raw counts, carrying the experiment they came from so that
            :func:`QCut.estimate_expectation_values` can be called on them alone

    """
    return submit_experiments(
        cut_experiment,
        shots=shots,
        backend=backend,
        max_batch_size=max_batch_size,
        run_options=run_options,
    ).result()


@deprecated(
    "run_cut_circuit() is planned to change in QCut 3.0: it will return a QCutJob "
    "without waiting, whose result() gives the expectation values or the "
    "distribution. The raw counts will stay available on the job."
)
def run_cut_circuit(
    cut_circuit: CutCircuit,
    observables: ObservablesLike | None = None,
    backend=None,
    max_batch_size: int = 100,
    options: CutOptions | None = None,
    shots: int = DEFAULT_SHOTS,
    qubits: list[int] | None = None,
    run_options: dict | None = None,
) -> np.ndarray | QuasiProbabilities:
    """After splitting the circuit run the rest of the circuit knitting sequence.

    Args:
        cut_circuit (CutCircuit): the split circuit, carrying its placeholder
            operations and cut locations
        observables: the observables to estimate, taken as qiskit's estimator takes
            them. See :func:`get_experiment_circuits`.
        backend: backend or V2 sampler to run the experiment circuits on. A device
            with a coupling map gets the subcircuits transpiled for it first. A
            sampler is not transpiled for, so it must take the circuits as built.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call (optional)
        options (CutOptions): configuration, overriding what the CutCircuit carries
            (optional)
        shots (int): number of shots per circuit run, as in :func:`submit_experiments`
            (optional)
        qubits (list[int]): the qubits to reconstruct a distribution over, instead of
            estimating observables (optional)
        run_options (dict): passed on to the backend or sampler, as in
            :func:`submit_experiments` (optional)

        One of observables or qubits must be provided.

    Returns:
        np.ndarray | QuasiProbabilities: one expectation value per observable, shaped
        like the observables given. A single observable comes back as a
        zero-dimensional array or, when given ``qubits``, the reconstructed
        distribution over them.

    """
    _check_measurement_spec(observables, qubits)

    if options is not None:
        cut_circuit.options = options

    if backend is None:
        raise QCutError("a backend or sampler to run the circuits on is needed")

    if has_topology(backend):
        cut_circuit = transpile_subcircuits(cut_circuit, backend, optimization_level=3)

    cut_experiment = get_experiment_circuits(cut_circuit, observables, qubits)

    results = submit_experiments(
        cut_experiment,
        shots=shots,
        backend=backend,
        max_batch_size=max_batch_size,
        run_options=run_options,
    ).result()

    if qubits is not None:
        return estimate_probabilities(results)

    return estimate_expectation_values(results)


@deprecated(
    "run() is planned to change in QCut 3.0: it will return a QCutJob without "
    "waiting, whose result() gives the expectation values or the distribution. The "
    "raw counts will stay available on the job."
)
def run(
    circuit: QuantumCircuit,
    observables: ObservablesLike | None = None,
    backend=None,
    max_batch_size: int = 100,
    options: CutOptions | None = None,
    shots: int = DEFAULT_SHOTS,
    qubits: list[int] | None = None,
    run_options: dict | None = None,
) -> np.ndarray | QuasiProbabilities:
    """Run the whole circuit knitting sequence with one function call.

    Args:
        circuit (QuantumCircuit): circuit with cut experiments
        observables: the observables to estimate, taken as qiskit's estimator takes
            them. See :func:`get_experiment_circuits`.
        backend: backend or V2 sampler to run the experiment circuits on. A device
            with a coupling map gets the subcircuits transpiled for it first. A
            sampler is not transpiled for, so it must take the circuits as built.
        max_batch_size (int): maximum number of circuits submitted per backend.run
            call (optional)
        options (CutOptions): configuration for the run (optional)
        shots (int): number of shots per circuit run, as in :func:`submit_experiments`
            (optional)
        qubits (list[int]): the qubits to reconstruct a distribution over, instead of
            estimating observables (optional)
        run_options (dict): passed on to the backend or sampler, as in
            :func:`submit_experiments` (optional)

        One of observables or qubits must be provided.

    Returns:
        np.ndarray | QuasiProbabilities: one expectation value per observable, shaped
        like the observables given.A single observable comes back as a
        zero-dimensional array or, when given ``qubits``, the reconstructed
        distribution over them.

    """
    _check_measurement_spec(observables, qubits)

    # circuit = circuit.copy()
    cut_circuit = get_locations_and_subcircuits(circuit, options=options)

    return run_cut_circuit(
        cut_circuit,
        observables,
        backend,
        max_batch_size,
        shots=shots,
        qubits=qubits,
        run_options=run_options,
    )
