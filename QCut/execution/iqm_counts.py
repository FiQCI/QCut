"""Reading a job's counts when the full job is not available from the server.

A providers server (e.g VTT QX) can decline to hand back per-shot measurements for a
large job and offer only its histogram. ``job.result()`` asks for
the measurements regardless and fails while reading the refusal, so the counts are
fetched through the backend's client instead and keyed the way ``job.result()`` would
have keyed them.
"""

from __future__ import annotations

import logging
import re
from uuid import UUID

from QCut.errors.qcuterror import QCutError
from QCut.execution.qcutresult import CountsResult

logger: logging.Logger = logging.getLogger(__name__)

#: qiskit-iqm's name for one measurement: ``{creg}_{creg_len}_{creg_idx}_{clbit_idx}``.
_MEASUREMENT_KEY = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)$")


def withholds_measurements(error: Exception) -> bool:
    """Whether a job's results failed because the server offers only counts.

    The refusal arrives as pydantic failing to read it as measurements, and pydantic
    shortens the input it quotes in its message, cutting through the very words that
    say so: ``'Raw measurem...ement_counts instead.'``. ``errors()`` holds it whole.
    """
    texts = [str(error)]
    try:
        texts += [str(detail.get("input")) for detail in error.errors()]
    except Exception:  # noqa: BLE001 - not pydantic's, so the message is all there is
        pass
    return any("measurement_counts" in text for text in texts)


def qiskit_counts(
    measurement_keys: list[str], counts: dict[str, int]
) -> dict[str, int]:
    """IQM's histogram, keyed as ``IQMJob`` keys the counts it builds itself.

    Character ``i`` of an IQM bitstring is the bit measured under
    ``measurement_keys[i]``. ``IQMJob`` lays the bits out with one field per register
    measured into, the last-added register first and the highest bit first within each,
    unmeasured bits reading zero. A register nothing was measured into is left out, as
    it is there, so both paths give the same keys.
    """
    positions: list[tuple[int, int]] = []
    sizes: dict[int, int] = {}
    for key in measurement_keys:
        match = _MEASUREMENT_KEY.match(key)
        if match is None:
            raise QCutError(f"cannot read the IQM measurement key '{key}'")
        creg_len, creg_idx, clbit_idx = map(int, match.groups()[1:])
        sizes[creg_idx] = creg_len
        positions.append((creg_idx, clbit_idx))
    order = sorted(sizes, reverse=True)

    out: dict[str, int] = {}
    for bitstring, count in counts.items():
        if len(bitstring) != len(positions):
            raise QCutError(
                f"IQM bitstring '{bitstring}' does not match its {len(positions)} "
                "measurement keys"
            )
        registers = {creg: ["0"] * size for creg, size in sizes.items()}
        for bit, (creg, clbit) in zip(bitstring, positions):
            registers[creg][sizes[creg] - 1 - clbit] = bit
        key = " ".join("".join(registers[creg]) for creg in order)
        out[key] = out.get(key, 0) + count
    return out


def counts_result(job, error: Exception) -> CountsResult:
    """A job's counts from the counts endpoint, for a job withholding its measurements.

    The client is the job's own backend's, so a job handed out by a wrapper such as
    :class:`QCut.ParallelBackend` is read from the machine that ran it.
    """
    try:
        fetch = job.backend().client.get_job_measurement_counts
    except Exception:  # noqa: BLE001 - any shortfall means there is no client to ask
        raise QCutError(
            "the backend withheld this job's raw measurements and offers only counts, "
            "but the job has no IQM client to fetch them through"
        ) from error
    job_id = job.job_id()
    logger.info(f"Job {job_id} withholds its raw measurements, reading its counts")
    return CountsResult(
        tuple(
            qiskit_counts(circuit.measurement_keys, circuit.counts)
            for circuit in fetch(UUID(str(job_id)))
        )
    )
