from QCut.errors.qcuterror import QCutError


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
