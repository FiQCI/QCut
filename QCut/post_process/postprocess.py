"""
A module for post-processing the results obtained from running the cut circuits and
calculating the estimated expectation values based on the results and the
provided observables.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from QCut.job.cutresult import RawResult, SubResult, TotalResult
from QCut.model.cutlocation import SingleQubitCutLocation
from QCut.utils.observables import _combine_pauli_ops, _get_observable_circuit_index


def _process_results(
    results: list,
    shots: int,
    qpd_bits: dict[tuple[int, int, int], tuple[int, int]] | None = None,
) -> list[list[TotalResult]]:
    """Transform results with post processing function {0,1} -> [-1, 1].

    Transform results so that we map 0 -> -1 and 1 -> 1. Gives processed results in form
    [TotalResult0, TotalResult1, ...], where TotalResult is
    [SubResult0, SubResult1, ...] and SubResult are [[[x0,x0,x0], [y0,y0], counts0],
    [[x1,x1,x1], [y1,y1], counts1], ...].

    Each outcome is weighted by how often it came up, so the weight carried through to
    the estimator is that outcome's probability.

    The qpd measurement bits are picked out by how many there are rather than by looking
    for their register in the reported counts. A backend is free to rename or merge
    classical registers -- IQM's transpiler does, when it routes through a resonator --
    and a key split on whitespace quietly returns the wrong field when it does. The
    widths come from the experiment, which knows them because it built the circuits.

    Bits that were dropped for going unwritten are put back as -1, which is what an
    unwritten bit reads as, so the sign is the one the full register would have given.

    Args:
        results (list): results from experiment circuits
        shots (int): number of shots the counts were taken at
        qpd_bits (dict): per circuit, how many qpd bits it wrote and how many were
            dropped. Without it the qpd register is taken to be the last whitespace
            separated field, as it was before the widths were recorded.

    Returns:
    -------
        list[TotalResult]:
            list of transformed results

    """
    processed_results = []

    for group_ind, circ_group in enumerate(results):
        for exp_ind, experiment_run in enumerate(circ_group):
            experiment_run_results = []
            for sub_ind, sub_result in experiment_run.items():
                circuit_results = []
                layout = (qpd_bits or {}).get((group_ind, exp_ind, sub_ind))
                for measurements, count in sub_result.counts().items():
                    if layout is None:
                        # No widths recorded: fall back to splitting the key. A circuit
                        # with nothing to write to its qpd register does not carry one,
                        # so there may be a single field; pad to two either way, since
                        # everything downstream reads the observable bits as the first
                        # and the qpd bits as the second.
                        fields = [f for f in measurements.split(" ") if f]
                        while len(fields) < 2:
                            fields.append("")
                        result_eigenvalues = [
                            np.array([-1 if x == "0" else 1 for x in field])
                            for field in fields
                        ]
                    else:
                        written, dropped = layout
                        bits = measurements.replace(" ", "")
                        split = len(bits) - written
                        qpd = [-1] * dropped + [
                            -1 if x == "0" else 1 for x in bits[split:]
                        ]
                        result_eigenvalues = [
                            np.array([-1 if x == "0" else 1 for x in bits[:split]]),
                            np.array(qpd),
                        ]
                    circuit_results.append(SubResult(result_eigenvalues, count / shots))
                experiment_run_results.append(circuit_results)
            if group_ind >= len(processed_results):
                processed_results.append([])
            processed_results[group_ind].append(TotalResult(experiment_run_results))

    return processed_results


#: Widest set of measured qubits to read by transform. ``2**24`` weights is about
#: 130 MB.
MAX_TRANSFORM_QUBITS = 24

#: Read every Z string at once while the spectrum is at most this many times the number
#: of observables asked for, and one at a time when they are sparser than that.
MAX_TRANSFORM_SPREAD = 8


def walsh_hadamard(values: np.ndarray) -> np.ndarray:
    r"""The Walsh-Hadamard transform of ``values``, whose length must be a power of two.

    Entry :math:`S` of the result is :math:`\sum_x (-1)^{|x \wedge S|} v_x`, the sum the
    definitions below take over every subset. Doing them together costs
    :math:`n 2^n` rather than :math:`4^n`.

    Args:
        values (np.ndarray): the vector to transform.

    Returns:
        np.ndarray: the transform, in the same indexing.
    """
    transformed = np.asarray(values, dtype=float)
    half = 1
    while half < len(transformed):
        pairs = transformed.reshape(-1, 2, half)
        transformed = np.concatenate(
            (pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]), axis=1
        ).reshape(-1)
        half *= 2
    return transformed


def _observables_by_setting(result_for_obs: list[dict]) -> dict[int, list[int]]:
    """Observable indices grouped by the measurement setting that covers them."""
    by_setting: dict[int, list[int]] = {}
    for ind, obs_data in enumerate(result_for_obs):
        by_setting.setdefault(obs_data["circuit_index"], []).append(ind)
    return by_setting


def _bit_layout(
    subcircuits: list, positions: list[int], map_qubits: Optional[dict[int, int]]
) -> list[tuple[list[int], list[int]]]:
    """Which of ``positions`` each subcircuit holds, and where in its own outcome.

    A position names a bit of the circuit-wide outcome, which is the subcircuits'
    outcomes concatenated, permuted and reversed. Tracing indices through that gives
    the same answer for every outcome, so it is worked out once.

    Args:
        subcircuits (list): one group's results, per subcircuit.
        positions (list[int]): the measured qubits, one bit of the outcome each.
        map_qubits (dict[int, int] | None): the permutation back to circuit order.

    Returns:
        list[tuple[list[int], list[int]]]: per subcircuit, which bits of the outcome it
        carries and the index of each in that subcircuit's own measurements.
    """
    widths = [len(sub[0].measurements[0]) for sub in subcircuits]
    starts = np.cumsum([0] + widths[:-1])
    ids = np.concatenate([np.arange(s, s + w) for s, w in zip(starts, widths)][::-1])
    if map_qubits is not None:
        ids = np.array(
            [ids[map_qubits[key]] for key in sorted(map_qubits.keys(), reverse=True)]
        )
    ids = ids[::-1]
    owner = np.concatenate([[s] * w for s, w in enumerate(widths)])

    layout = []
    for index, start in enumerate(starts):
        bits = [b for b, p in enumerate(positions) if owner[ids[p]] == index]
        layout.append((bits, [ids[positions[b]] - start for b in bits]))
    return layout


def _outcome_rows(sub: list, offsets: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """One subcircuit's weight for each outcome it actually produced.

    Args:
        sub (list): the subcircuit's results within one group.
        offsets (list[int]): where in the subcircuit's own measurements each of its bits
            sits, as :func:`_bit_layout` reports them.

    Returns:
        tuple[np.ndarray, np.ndarray]: the outcomes carrying a weight, ascending, and
        their weights. Bit ``t`` of an outcome is the bit ``offsets[t]`` names.
    """
    weights: dict[int, float] = {}
    for res in sub:
        outcome = 0
        for bit, offset in enumerate(offsets):
            if res.measurements[0][offset] < 0:
                outcome |= 1 << bit
        weights[outcome] = weights.get(outcome, 0.0) + res.count * np.prod(
            res.measurements[1]
        )
    if not weights:
        return np.empty(0, dtype=np.intp), np.empty(0)
    idx = np.fromiter(sorted(weights), dtype=np.intp, count=len(weights))
    return idx, np.array([weights[int(i)] for i in idx])


def _outcome_weights(sub: list, offsets: list[int]) -> np.ndarray:
    """The same weights written out, for the readers that index them densely.

    Args:
        sub (list): the subcircuit's results within one group.
        offsets (list[int]): where in the subcircuit's own measurements each of its bits
            sits, as :func:`_bit_layout` reports them.

    Returns:
        np.ndarray: ``2**len(offsets)`` weights, indexed so that bit ``t`` of the index
        is the bit ``offsets[t]`` names.
    """
    idx, values = _outcome_rows(sub, offsets)
    weights = np.zeros(1 << len(offsets))
    weights[idx] = values
    return weights


def _outcome_spectrum(
    results_processed: list,
    experiment,
    setting: int,
    positions: list[int],
    parity: int,
) -> np.ndarray:
    r"""For every Z string over ``positions``, the signed sum the estimator needs.

    Entry :math:`S` is :math:`\sum_g c_g \sum_x w_x (-1)^{|x \wedge S|}`, over the
    outcomes :math:`x` of group :math:`g` and the weight each carries.

    A subcircuit's outcome fixes its own bits of :math:`x` only, and the weight is a
    product over the subcircuits, so the sum factorises: each subcircuit is read on its
    own and the transforms multiplied. Walking the outcomes jointly instead would cost
    :math:`\prod_i m_i` for a group whose subcircuits reported :math:`m_i` outcomes
    each, against :math:`\sum_i m_i` here, and :math:`m_i` grows with the shot count.

    Args:
        results_processed (list): processed results, one entry per group.
        experiment (CutExperiment): the experiment the results came from.
        setting (int): index of the measurement setting to read.
        positions (list[int]): the measured qubits, one bit of the outcome each.
        parity (int): the wire cut sign shared by every term.

    Returns:
        np.ndarray: ``2**len(positions)`` sums, indexed by Z string.
    """
    width = 1 << len(positions)
    strings = np.arange(width)
    spectrum = np.zeros(width)
    spread: dict[tuple[int, ...], np.ndarray] = {}

    for experiment_run, coefficient in zip(results_processed, experiment.coefficients):
        subcircuits = experiment_run[setting].subcircuits[0]
        if any(len(sub) == 0 for sub in subcircuits):
            continue

        group = np.full(width, float(parity * coefficient))
        layout = _bit_layout(subcircuits, positions, experiment.map_qubit)
        for sub, (bits, offsets) in zip(subcircuits, layout):
            weights = _outcome_weights(sub, offsets)

            key = tuple(bits)
            if key not in spread:
                picked = np.zeros_like(strings)
                for bit, position in enumerate(key):
                    picked |= ((strings >> position) & 1) << bit
                spread[key] = picked
            group *= walsh_hadamard(weights)[spread[key]]
        spectrum += group

    return spectrum


def _expectation_values_by_transform(
    results_processed: list,
    experiment,
    setting: int,
    positions: list[int],
    observables: list[list[int]],
    parity: int,
) -> np.ndarray:
    """Every observable of one measurement setting, from a single pass over the results.

    Args:
        results_processed (list): processed results, one entry per group.
        experiment (CutExperiment): the experiment the results came from.
        setting (int): index of the measurement setting to read.
        positions (list[int]): the measured qubits covered by these observables.
        observables (list[list[int]]): each observable as the qubits it acts on.
        parity (int): the wire cut sign shared by every term.

    Returns:
        np.ndarray: one expectation value per entry of ``observables``.
    """
    spectrum = _outcome_spectrum(
        results_processed, experiment, setting, positions, parity
    )
    bit_of = {position: bit for bit, position in enumerate(positions)}
    masks = [sum(1 << bit_of[q] for q in obs) for obs in observables]
    signs = np.array([(-1) ** (len(obs) + 1) for obs in observables])
    return spectrum[masks] * signs


def _expectation_values_by_factors(
    results_processed: list,
    experiment,
    setting: int,
    positions: list[int],
    observables: list[list[int]],
    parity: int,
) -> np.ndarray:
    """The same, for observables spanning too many qubits to hold every Z string.

    Each subcircuit is still read once per group rather than walking the product of
    their outcomes, but the observables are taken one at a time, so nothing of size
    ``2**len(positions)`` is built. A subcircuit holding none of an observable's qubits
    still contributes its own weight, which is what the identity entry of the transform
    is in the other path.

    Args:
        results_processed (list): processed results, one entry per group.
        experiment (CutExperiment): the experiment the results came from.
        setting (int): index of the measurement setting to read.
        positions (list[int]): the measured qubits covered by these observables.
        observables (list[list[int]]): each observable as the qubits it acts on.
        parity (int): the wire cut sign shared by every term.

    Returns:
        np.ndarray: one expectation value per entry of ``observables``.
    """
    values = np.zeros(len(observables))
    bit_of = {position: bit for bit, position in enumerate(positions)}

    for experiment_run, coefficient in zip(results_processed, experiment.coefficients):
        subcircuits = experiment_run[setting].subcircuits[0]
        if any(len(sub) == 0 for sub in subcircuits):
            continue

        group = np.full(len(observables), float(parity * coefficient))
        layout = _bit_layout(subcircuits, positions, experiment.map_qubit)
        for sub, (bits, offsets) in zip(subcircuits, layout):
            eigenvalues = np.array([res.measurements[0] for res in sub])
            weights = np.array(
                [res.count * np.prod(res.measurements[1]) for res in sub]
            )
            offset_of = dict(zip(bits, offsets))
            for ind, obs in enumerate(observables):
                held = [offset_of[bit_of[q]] for q in obs if bit_of[q] in offset_of]
                group[ind] *= weights @ (
                    np.prod(eigenvalues[:, held], axis=1)
                    if held
                    else np.ones(len(weights))
                )
        values += group

    return values * np.array([(-1) ** (len(obs) + 1) for obs in observables])


def estimate_expectation_values(results: RawResult) -> np.ndarray:
    r"""Calculate the estimated expectation values.

    The estimate is the quasiprobability sum itself,

    .. math::

        \langle O \rangle = (-1)^{w+1} \sum_g c_g E_g

    over the subcircuit groups, where :math:`c_g` is the group's coefficient, :math:`w`
    counts the wire cuts and :math:`E_g` is the group's own estimate: over the outcomes
    of its subcircuits, the product of their probabilities times the observable's
    eigenvalue times the sign the mid-circuit measurements carry. The parity is the qpd
    register's sign convention: every one of its bits maps 0 to -1, so an unwritten bit
    contributes -1 and only the number allocated per subcircuit survives.

    Multi-qubit observables pick up a further :math:`(-1)^{m+1}` for their :math:`m`
    qubits.

    :math:`E_g` factorises over the subcircuits, so each is read once per group rather
    than walking the product of their outcomes, and the observables sharing a
    measurement setting are read together rather than one pass each. See
    :func:`_outcome_spectrum`.

    Args:
        results (RawResult): raw results from experiment circuits, carrying the
            experiment they came from.

    Returns:
        np.ndarray:
            one expectation value per observable, shaped like the observables the
            experiment was built with, so a single observable comes back as a
            zero-dimensional array

    """
    raw_results = results
    experiment = raw_results.experiment
    if experiment is None:
        raise ValueError(
            "These results carry no experiment, so there is nothing to interpret them "
            "with. Results from a QCutJob or QCut.run_experiments carry it already; a "
            "hand-built RawResult has to be given the experiment it came from."
        )
    results_processed = _process_results(
        raw_results.results, raw_results._shots, experiment.qpd_bits
    )

    wire_cuts = len(
        [i for i in experiment.cut_locations if isinstance(i, SingleQubitCutLocation)]
    )
    parity = np.power(-1, wire_cuts + 1)

    spec = experiment.observable_spec
    terms = experiment.observable_terms
    measurement_settings = _combine_pauli_ops(terms)

    result_for_obs = []

    for obs in terms.paulis:
        obs_circuit_info = _get_observable_circuit_index(obs, measurement_settings)
        result_for_obs.append(obs_circuit_info)

    expectation_values = np.zeros(len(terms))

    for obs_data in result_for_obs:
        if obs_data["circuit_index"] is None:
            raise ValueError("""Observable cannot be measured
                             with given measurement settings.""")

    for setting, indices in _observables_by_setting(result_for_obs).items():
        positions = sorted(
            {q for ind in indices for q in result_for_obs[ind]["obs_indices"]}
        )
        together = len(positions) <= MAX_TRANSFORM_QUBITS and (
            1 << len(positions)
        ) <= MAX_TRANSFORM_SPREAD * len(indices)
        read = (
            _expectation_values_by_transform
            if together
            else _expectation_values_by_factors
        )
        expectation_values[indices] = read(
            results_processed,
            experiment,
            setting,
            positions,
            [result_for_obs[ind]["obs_indices"] for ind in indices],
            parity,
        )

    return spec.combine(expectation_values)


def estimate_standard_errors(results: RawResult) -> np.ndarray:
    r"""An upper bound on the shot noise in :func:`estimate_expectation_values`.

    Each Pauli term is the sum :math:`\sum_g c_g E_g` over the groups, each group
    estimated from :math:`n` shots of outcomes in :math:`[-1, 1]`, so its standard
    error is at most

    .. math::

        \sqrt{\sum_g c_g^2 / n \; + \; \gamma^2 / D}

    with :math:`\gamma = \sum_g |c_g|`. The second term is there only when the
    decomposition was sampled, :math:`D` being the number of draws. This is the
    stratified-sampling variance with each group's variance taken at its largest, see
    arXiv:2602.11245. An observable :math:`\sum_j a_j P_j` gets
    :math:`\sum_j |a_j|` times that.

    It covers sampling error only, not device noise or bias.

    Args:
        results (RawResult): raw results from experiment circuits, carrying the
            experiment they came from.

    Returns:
        np.ndarray: one standard error per observable, shaped like the expectation
        values.
    """
    experiment = results.experiment
    if experiment is None:
        raise ValueError(
            "These results carry no experiment, so there is nothing to interpret them "
            "with. Results from a QCutJob or QCut.run_experiments carry it already; a "
            "hand-built RawResult has to be given the experiment it came from."
        )
    coefficients = np.asarray(experiment.coefficients, dtype=float)
    variance = np.sum(coefficients**2) / results._shots
    if experiment.sampled:
        variance += np.sum(np.abs(coefficients)) ** 2 / experiment.num_draws
    per_term = np.sqrt(variance)

    spec = experiment.observable_spec
    if spec.weights is None:
        return np.full(spec.shape, per_term)
    return (np.abs(spec.weights).sum(axis=1) * per_term).reshape(spec.shape)
