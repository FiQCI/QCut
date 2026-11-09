"""End-to-end checks against a backend with a coupling map.

Nothing else in the suite runs a cut experiment against a backend that has a real
topology, and that gap hid a run of defects: transpilation lays subcircuits out on
physical qubits and pads them to the device width, so reading measurement bits by
position attributed results to the wrong qubits; placeholders were looked up by the
index they had before transpiling; routing borrowed wires that were then measured; a
resonator machine's gate list broke the basis translation; and unused classical
registers made real hardware refuse the job outright. Every one of those passed
``pytest`` while returning wrong numbers or none at all, because the only backend
test asserted gate names.

So these assert values, and that the circuits would be accepted.
"""

from copy import deepcopy

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator

import QCut as ck
from QCut import CutOptions, cut, cutGate
from QCut.errors.qcuterror import QCutError
from QCut.qpd.bundle import plan_bundles
from QCut.qpd.qpd_operations import coupling_filter
from QCut.transpilation.transpile_cut_circuits import (
    _IQM_ENFORCED,
    _markers_in,
    _placeholder_gates,
)
from QCut.utils.backend import is_resonator_backend

iqm = pytest.importorskip(
    "iqm.qiskit_iqm", reason='needs the IQM adapter: pip install "QCut[iqm]"'
)
validate_circuit = iqm.iqm_circuit_validation.validate_circuit

#: Loose, because these run few shots on purpose: the failures being guarded against are
#: wrong-qubit and wrong-register ones, which miss by order one, not by a fraction.
TOLERANCE = 0.15
SHOTS = 2**12


def _backends():
    return [
        pytest.param(iqm.IQMFakeAdonis(), id="adonis_star"),
        pytest.param(iqm.IQMFakeDeneb(), id="deneb_resonator"),
        pytest.param(iqm.IQMFakeApollo(), id="apollo"),
    ]


def _direct_backends():
    """The backends that couple qubits to each other, so a plain two-qubit gate runs.

    A resonator machine is excluded on purpose: it couples qubits only through its
    resonator, so a block inserted after transpilation has no locus to land on however
    the MOVEs are routed. :func:`test_a_resonator_device_gets_its_moves_inserted` and
    :func:`test_a_block_is_never_bundled_on_a_resonator_device` cover it separately.
    """
    return [
        pytest.param(iqm.IQMFakeAdonis(), id="adonis_star"),
        pytest.param(iqm.IQMFakeApollo(), id="apollo"),
    ]


def _ideal(subcircuits, backend):
    """An ideal simulator that can run whatever came out of transpilation.

    Aer has no MOVE gate, so a move-routed circuit cannot go through it at all. The fake
    device can, and clearing its noise model leaves it ideal -- and still checking every
    locus, which Aer never did.
    """
    if not any(sub.count_ops().get("move") for sub in subcircuits):
        return AerSimulator()

    from qiskit_aer.noise import NoiseModel

    ideal = deepcopy(backend)
    ideal.noise_model = NoiseModel(basis_gates=list(ideal.noise_model.basis_gates))
    return ideal


def _gate_cut():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.ry(0.3 + 0.2 * qubit, qubit)
        circuit.rzz(0.4, 0, 1)
        circuit.rzz(0.5, 2, 3)
    marked.append(**cutGate(RZZGate(0.9), 1, 2))
    plain.rzz(0.9, 1, 2)
    return marked, plain, ["IIIZ", "IIZI", "IZII", "ZIII"]


def _wire_cut():
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.ry(0.3 + 0.2 * qubit, qubit)
        circuit.cx(0, 1)
    marked.append(cut(), [1])
    for circuit in (marked, plain):
        circuit.cx(1, 2)
        circuit.cx(2, 3)
    return marked, plain, ["IIIZ", "IIZI", "IZII", "ZIII"]


def _locc_block():
    marked, plain = QuantumCircuit(6), QuantumCircuit(6)
    for circuit in (marked, plain):
        for qubit in range(6):
            circuit.ry(0.3 + 0.13 * qubit, qubit)
        circuit.cx(0, 1)
    marked.append(cut(), [0])
    marked.append(cut(), [1])
    for circuit in (marked, plain):
        circuit.cx(0, 2)
        circuit.cx(1, 3)
    return marked, plain, ["IIIIIZ", "IIIIZI"]


def _locc_pair():
    """Two cuts that really do share one block, unlike :func:`_locc_block`.

    That one bundles each wire on its own, so its terms are one-qubit operations and it
    never exercised the case this file exists for: a block spanning two wires puts a
    two-qubit gate in after transpilation, at whatever wires the layout happened to give
    the cuts, with nothing routing it.
    """
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
    marked.append(cut(), [1])
    marked.append(cut(), [2])
    for circuit in (marked, plain):
        circuit.cx(1, 2)
        circuit.cx(2, 3)
    return marked, plain, ["IIIZ", "IIZI", "IZII", "IIZZ"]


def _rotations_across_a_wire_cut():
    """A cut with single-qubit rotations either side of it, as a QAOA layer has.

    The other cases put their cuts where almost nothing surrounds them, which hid a
    defect: IQM's transpiler commutes Z rotations along a wire, and a barrier does not
    stop it, so a rotation written before the cut was applied after it -- on the far
    side of a wire that has been measured and re-prepared in between. Nothing catches
    that unless there is a rotation there to move.
    """
    marked, plain = QuantumCircuit(4), QuantumCircuit(4)
    for circuit in (marked, plain):
        for qubit in range(4):
            circuit.h(qubit)
        circuit.rzz(0.7, 0, 1)
        circuit.rzz(0.9, 1, 2)
    marked.append(cut(), [2])
    for circuit in (marked, plain):
        circuit.rzz(1.1, 2, 3)
        for qubit in range(4):
            circuit.rx(0.5, qubit)
    return marked, plain, ["IIZZ", "IZZI", "ZZII", "IIIZ"]


CASES = {
    "gate_cut": _gate_cut,
    "wire_cut": _wire_cut,
    "locc_block": _locc_block,
    "locc_pair": _locc_pair,
    "rotations_across_a_cut": _rotations_across_a_wire_cut,
}


def _spans_two_wires(bundles):
    return any(bundle.size > 1 for bundle in bundles)


def _every_register_is_written(circuit):
    """IQMBackend.run refuses a job whose circuit has a register nothing writes to."""
    utilisation = dict.fromkeys(circuit.cregs, 0)
    for instruction in circuit.data:
        if instruction.clbits:
            register = circuit.find_bit(instruction.clbits[0]).registers[0][0]
            utilisation[register] += 1
    return 0 not in utilisation.values()


def _accepted_by(backend, experiment, shots=32):
    """Submit to the device itself, which is the only thing that validates the topology.

    Running the experiment on an ideal simulator checks the arithmetic, and Aer will
    happily execute a two-qubit gate on any pair of wires. Only the backend checks that
    each gate sits on a locus the architecture actually has, which is what caught the
    layout being undone rather than honoured: the answers were right and the circuits
    were unrunnable.
    """
    ck.run_experiments(experiment, backend=backend, shots=shots)


def _run(marked, observables, backend, level, use_iqm):
    if not use_iqm and is_resonator_backend(backend):
        pytest.skip("qiskit's transpiler cannot route a resonator device's MOVEs")
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    transpiled = ck.transpile_subcircuits(
        cut_circuit, backend, optimization_level=level, use_iqm_transpiler=use_iqm
    )
    experiment = ck.get_experiment_circuits(transpiled, observables)
    results = ck.run_experiments(
        experiment, backend=_ideal(transpiled.subcircuits, backend), shots=SHOTS
    )
    return experiment, np.array(ck.estimate_expectation_values(results))


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _backends())
def test_transpiling_the_experiments_reaches_the_device(backend):
    """Experiment circuits transpiled by IQM's own transpiler still give the answer."""
    marked, plain, observables = _wire_cut()
    state = Statevector(plain)
    exact = np.array(
        [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]
    )

    experiment = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(marked.copy()), observables
    )
    transpiled = ck.transpile_experiments(experiment, backend, optimization_level=3)
    circuits = [
        circuit
        for groups in transpiled.experiments
        for obs_group in groups
        for circuit in obs_group.values()
    ]
    assert circuits and all("id" not in circuit.count_ops() for circuit in circuits)

    results = ck.run_experiments(
        transpiled, backend=_ideal(circuits, backend), shots=SHOTS
    )
    values = np.array(ck.estimate_expectation_values(results))
    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE, (
            f"on {backend.name}: expected {expected}, got {actual}"
        )


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("level", [0, 3])
@pytest.mark.parametrize("use_iqm", [True, False], ids=["iqm_transpiler", "qiskit"])
def test_a_cut_experiment_survives_a_real_topology(backend, case, level, use_iqm):
    """The answer has to come back right after being laid out on physical qubits."""
    marked, plain, observables = CASES[case]()
    state = Statevector(plain)
    exact = np.array(
        [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]
    )

    _, values = _run(marked, observables, backend, level, use_iqm)

    for expected, actual in zip(exact, values):
        assert abs(expected - actual) < TOLERANCE, (
            f"{case} on {backend.name} at O{level}: expected {expected}, got {actual}"
        )


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_no_circuit_carries_an_unwritten_register(case):
    """Real hardware refuses those, and QCut used to emit them for most groups."""
    marked, _plain, observables = CASES[case]()
    experiment, _ = _run(marked, observables, iqm.IQMFakeAdonis(), 3, True)

    offenders = [
        (group, obs, sub)
        for group, groups in enumerate(experiment.experiments)
        for obs, obs_group in enumerate(groups)
        for sub, circuit in obs_group.items()
        if not _every_register_is_written(circuit)
    ]
    assert offenders == []


@pytest.mark.sim
@pytest.mark.slow
def test_the_subcircuits_end_up_in_the_backend_basis():
    """Whichever transpiler ran, the result has to be something the device can run."""
    marked, _plain, observables = _gate_cut()
    # The qiskit path is not offered for a resonator device, so it is checked on a
    # backend that couples its qubits directly.
    for backend, use_iqm in (
        (iqm.IQMFakeDeneb(), True),
        (iqm.IQMFakeApollo(), True),
        (iqm.IQMFakeApollo(), False),
    ):
        cut_circuit = ck.get_locations_and_subcircuits(marked.copy())
        transpiled = ck.transpile_subcircuits(
            cut_circuit, backend, optimization_level=3, use_iqm_transpiler=use_iqm
        )
        for subcircuit in transpiled.subcircuits:
            for instruction in subcircuit.data:
                name = instruction.operation.name
                assert name in (
                    "r",
                    "cz",
                    "move",
                    "barrier",
                    "measure",
                ) or name.startswith(("cut", "obs_", "Meas_", "Init_")), (
                    f"unexpected {name} on {backend.name} "
                    f"with use_iqm_transpiler={use_iqm}"
                )


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    _direct_backends(),
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_device_accepts_the_circuits(backend, case):
    """Every gate has to sit on a locus the architecture has.

    Transpiling chooses a layout so the two-qubit gates land on coupled pairs. Renaming
    the wires afterwards to put the subcircuit's own qubits back in order would throw
    that away and the backend would refuse the job, so the placement is kept and the
    measurements go through the recorded layout instead.
    """
    marked, _plain, observables = CASES[case]()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)
    experiment = ck.get_experiment_circuits(transpiled, observables)

    _accepted_by(backend, experiment)


@pytest.mark.sim
@pytest.mark.slow
def test_a_resonator_device_gets_its_moves_inserted():
    """A star machine couples qubits only through its resonator.

    So a plain two-qubit gate is not runnable there whatever the layout, and the MOVEs
    have to go in while the subcircuits are transpiled. The pass that inserts them
    round-trips the circuit through IQM's own format, which loses the placeholders'
    labels, the classical registers and the layout, so what is pinned here is that all
    three come back and that the device then accepts the experiment.
    """
    backend = iqm.IQMFakeDeneb()
    marked, _plain, observables = _gate_cut()
    cut_circuit = ck.get_locations_and_subcircuits(marked.copy())
    names = set(_placeholder_gates(cut_circuit))
    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)

    assert any(sub.count_ops().get("move") for sub in transpiled.subcircuits)
    for before, after in zip(cut_circuit.subcircuits, transpiled.subcircuits):
        assert sorted(_markers_in(after, names)) == sorted(_markers_in(before, names))
        assert [(reg.name, reg.size) for reg in after.cregs] == [
            (reg.name, reg.size) for reg in before.cregs
        ]

    _accepted_by(backend, ck.get_experiment_circuits(transpiled, observables))


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_move_routed_circuit_is_read_the_same_without_its_layout(case):
    """Wire n has to be physical qubit n, not merely say so in the layout.

    A target that submits OpenQASM numbers the wires by position and reads no layout.
    """
    backend = iqm.IQMFakeDeneb()
    marked, _plain, observables = CASES[case]()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)
    experiment = ck.get_experiment_circuits(transpiled, observables)

    for group in experiment.experiments:
        for obs_group in group:
            for circuit in obs_group.values():
                stripped = circuit.copy()
                stripped._layout = None
                validate_circuit(stripped, backend)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _backends())
def test_an_x_or_y_observable_reaches_the_device(backend):
    """Its basis change has to arrive as the device's own gates.

    Every other case here measures in Z, which needs no basis change at all.
    """
    marked, plain, _z_only = _gate_cut()
    observables = ["IIXX", "YYII", "IIIZ", "IXYI"]
    state = Statevector(plain)
    exact = np.array(
        [float(np.real(state.expectation_value(Pauli(p)))) for p in observables]
    )

    cut_circuit = ck.get_locations_and_subcircuits(marked.copy())
    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)
    experiment = ck.get_experiment_circuits(transpiled, observables)

    _accepted_by(backend, experiment)

    results = ck.run_experiments(
        experiment, backend=_ideal(transpiled.subcircuits, backend), shots=SHOTS
    )
    values = np.array(ck.estimate_expectation_values(results))
    assert np.allclose(values, exact, atol=TOLERANCE)


@pytest.mark.parametrize(
    "transpile", [ck.transpile_subcircuits, ck.transpile_experiments]
)
def test_the_qiskit_path_is_refused_for_a_resonator_device(transpile):
    """It has no MOVE gate, and the coupling map claims the qubits couple directly."""
    marked, _plain, observables = _wire_cut()
    cut_circuit = ck.get_locations_and_subcircuits(marked.copy())
    argument = (
        cut_circuit
        if transpile is ck.transpile_subcircuits
        else ck.get_experiment_circuits(cut_circuit, observables)
    )

    with pytest.raises(QCutError, match="MOVE"):
        transpile(argument, iqm.IQMFakeDeneb(), use_iqm_transpiler=False)


@pytest.mark.sim
@pytest.mark.slow
def test_a_block_is_never_bundled_on_a_resonator_device():
    """Routing the MOVEs does not make a block placeable.

    A block's two-qubit gates go in after transpilation, so nothing routes them, and on
    a resonator machine there is no pair of qubits they could sit on. The device reports
    its qubits as fully coupled all the same, because the transpiler can reach any pair
    through the resonator, so the coupling map is not what the veto can go on.
    """
    backend = iqm.IQMFakeDeneb()
    marked, _plain, _observables = _locc_pair()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    transpiled = ck.transpile_subcircuits(cut_circuit, backend, optimization_level=3)

    fits = coupling_filter(
        transpiled.cut_locations, transpiled.subcircuits, transpiled.backend
    )
    assert fits is not None
    bundles = plan_bundles(
        transpiled.cut_locations,
        transpiled.subcircuits,
        transpiled.options,
        fits=fits,
    )
    assert all(bundle.size == 1 for bundle in bundles)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
@pytest.mark.parametrize("use_iqm", [True, False], ids=["iqm_transpiler", "qiskit"])
def test_a_block_spanning_two_wires_is_not_left_unrouted(backend, use_iqm):
    """A bundle's two-qubit gates have to sit on a locus the device has.

    They go in after transpilation, so the transpiler never saw them and had no reason
    to put the cuts' wires next to each other -- on a star machine it usually cannot.
    Planning has to notice and fall back to cutting the wires separately, which needs
    only one-qubit operations.
    """
    marked, _plain, observables = _locc_pair()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    assert _spans_two_wires(
        plan_bundles(
            cut_circuit.cut_locations, cut_circuit.subcircuits, cut_circuit.options
        )
    ), "the case has stopped exercising a multi-wire block"

    transpiled = ck.transpile_subcircuits(
        cut_circuit, backend, optimization_level=3, use_iqm_transpiler=use_iqm
    )
    experiment = ck.get_experiment_circuits(transpiled, observables)
    _accepted_by(backend, experiment)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
def test_transpiling_the_experiments_keeps_the_joint_decomposition(backend):
    """The fallback is only needed because the block goes in after transpilation.

    Build the experiment circuits first and the block is routed with everything else, so
    the cheaper decomposition survives and the device still takes the job.
    """
    marked, plain, observables = _locc_pair()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked.copy(), options=CutOptions(wire_cut_communication="always")
    )
    experiment = ck.get_experiment_circuits(cut_circuit, observables)
    separate = ck.get_experiment_circuits(
        ck.get_locations_and_subcircuits(
            marked.copy(), options=CutOptions(wire_cut_communication="never")
        ),
        observables,
    )
    assert experiment.num_circuits < separate.num_circuits

    transpiled = ck.transpile_experiments(experiment, backend, optimization_level=3)
    assert transpiled.num_circuits == experiment.num_circuits
    _accepted_by(backend, transpiled)

    results = ck.run_experiments(transpiled, backend=AerSimulator(), shots=SHOTS)
    values = np.array(ck.estimate_expectation_values(results))
    exact = np.array(
        [
            float(np.real(Statevector(plain).expectation_value(Pauli(pauli))))
            for pauli in observables
        ]
    )
    assert np.allclose(values, exact, atol=TOLERANCE)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
def test_a_block_too_wide_to_route_falls_back_to_narrow_blocks(backend):
    """Falling back is not the same as giving up on communicating.

    Two wires that cannot exchange their outcomes as one block still can as two blocks
    of one, at gamma 3 each against the 4 of the local decomposition. Planning has to
    retry the group in smaller pieces rather than drop straight to no bundling.

    IQM's transpiler is the path that needs it: it refuses a custom gate, so the block
    cannot be handed to it as one marker as wide as itself, and the wires it lands on
    are whatever the layout chose. The qiskit path keeps the whole block and is covered
    by :func:`test_a_wide_block_can_survive_subcircuit_transpilation`.
    """
    marked, _plain, observables = _locc_pair()

    def plan(communication):
        cut_circuit = ck.get_locations_and_subcircuits(
            marked.copy(), options=CutOptions(wire_cut_communication=communication)
        )
        transpiled = ck.transpile_subcircuits(
            cut_circuit, backend, optimization_level=3, use_iqm_transpiler=True
        )
        return transpiled, ck.get_experiment_circuits(transpiled, observables)

    transpiled, forced = plan("always")
    _, local = plan("never")

    bundles = plan_bundles(
        transpiled.cut_locations,
        transpiled.subcircuits,
        transpiled.options,
        fits=coupling_filter(
            transpiled.cut_locations, transpiled.subcircuits, transpiled.backend
        ),
    )
    assert [bundle.kind for bundle in bundles] == ["cc_wire", "cc_wire"]
    assert all(bundle.size == 1 for bundle in bundles)
    assert forced.num_circuits < local.num_circuits
    _accepted_by(backend, forced)


@pytest.mark.sim
@pytest.mark.slow
@pytest.mark.parametrize("backend", _direct_backends())
@pytest.mark.parametrize("level", [0, 3])
def test_a_wide_block_can_survive_subcircuit_transpilation(backend, level):
    """A block is routed properly when the transpiler is told its real width.

    Its placeholders are one qubit each, so left alone the transpiler lays them out as
    unrelated single-qubit gates and the block goes in afterwards on wires the device
    may not couple. Replacing them by one marker as wide as the block makes routing
    place it on a locus that exists, and nothing can be scheduled inside a single
    instruction, so the cuts it covers stay simultaneous.

    Falling back is always allowed -- the veto in ``coupling_filter`` sees to that -- so
    this asserts only that the answers are right and the device takes them, and that at
    least one configuration does keep the block.
    """
    marked, plain, observables = _locc_pair()

    def build(communication):
        cut_circuit = ck.get_locations_and_subcircuits(
            marked.copy(), options=CutOptions(wire_cut_communication=communication)
        )
        transpiled = ck.transpile_subcircuits(
            cut_circuit, backend, optimization_level=level, use_iqm_transpiler=False
        )
        return ck.get_experiment_circuits(transpiled, observables)

    experiment = build("auto")
    _accepted_by(backend, experiment)

    results = ck.run_experiments(experiment, backend=AerSimulator(), shots=SHOTS)
    values = np.array(ck.estimate_expectation_values(results))
    exact = np.array(
        [
            float(np.real(Statevector(plain).expectation_value(Pauli(pauli))))
            for pauli in observables
        ]
    )
    assert np.allclose(values, exact, atol=TOLERANCE)
    assert experiment.num_circuits <= build("never").num_circuits


@pytest.mark.parametrize("backend", _direct_backends())
@pytest.mark.parametrize("seed", [123, 126, 286])
def test_routing_may_put_two_cuts_on_one_wire(backend, seed):
    """The qubits left to measure are counted by qubit, not by the wire holding one.

    A cut ends what was on its wire, and routing is free to move another qubit there
    afterwards, so two placeholders can report the same wire. Counting wires then leaves
    a qubit still to be measured looking like it is already spoken for, and the measure
    is one bit short of its register. These seeds are three that did that.
    """
    marked, _plain, observables = _locc_pair()
    cut_circuit = ck.get_locations_and_subcircuits(
        marked, options=CutOptions(wire_cut_communication="never")
    )
    transpiled = ck.transpile_subcircuits(
        cut_circuit,
        backend,
        optimization_level=0,
        use_iqm_transpiler=False,
        transpile_options={"seed_transpiler": seed},
    )

    experiment = ck.get_experiment_circuits(transpiled, observables)

    for group in experiment.experiments:
        for subcircuits in group:
            for circuit in subcircuits.values():
                meas = next(r for r in circuit.cregs if r.name == "meas")
                written = {
                    index
                    for instruction in circuit.data
                    if instruction.operation.name == "measure"
                    for clbit in instruction.clbits
                    for register, index in circuit.find_bit(clbit).registers
                    if register.name == "meas"
                }
                assert written == set(range(meas.size))


@pytest.mark.sim
@pytest.mark.parametrize("option", sorted(_IQM_ENFORCED))
def test_an_option_that_would_break_a_placeholder_is_refused(option):
    """Asking for one of these used to be ignored, which is worse than being told no.

    Each of them rewrites a circuit in a way that only makes sense once nothing is left
    to be chosen, so the answer is to transpile the experiment circuits instead, and the
    error says so.
    """
    marked, _plain, _observables = _gate_cut()
    cut_circuit = ck.get_locations_and_subcircuits(marked)

    required, _why = _IQM_ENFORCED[option]
    with pytest.raises(QCutError, match="transpile_experiments"):
        ck.transpile_subcircuits(
            cut_circuit,
            iqm.IQMFakeAdonis(),
            transpile_options={option: not required},
        )


@pytest.mark.sim
@pytest.mark.parametrize("option", sorted(_IQM_ENFORCED))
def test_restating_an_enforced_option_is_allowed(option):
    """Passing the value it is already held at asks for nothing, so it is allowed."""
    required, _why = _IQM_ENFORCED[option]
    transpiled = ck.transpile_subcircuits(
        ck.get_locations_and_subcircuits(_gate_cut()[0]),
        iqm.IQMFakeAdonis(),
        transpile_options={option: required},
    )
    assert len(transpiled.subcircuits) == 2


@pytest.mark.sim
def test_the_qiskit_path_takes_its_own_options():
    """The refusal is about IQM's transpiler, so it must not reach the generic path."""
    transpiled = ck.transpile_subcircuits(
        ck.get_locations_and_subcircuits(_gate_cut()[0]),
        iqm.IQMFakeAdonis(),
        use_iqm_transpiler=False,
        transpile_options={"seed_transpiler": 3},
    )
    assert len(transpiled.subcircuits) == 2
