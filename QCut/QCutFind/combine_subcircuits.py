"""
Utility function for combining subcircuits into larger subcircuits that fit within
specified qubit limits.
"""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister


def group_circuits(subcircuits, max_qubits):
    subcircuits.sort(key=lambda x: x.num_qubits, reverse=True)
    max_qubits.sort(reverse=True)

    op_subs = subcircuits.copy()

    groups = []

    for mq in max_qubits:
        current_group = []
        tot = 0

        # Use a while loop and iterate backwards to avoid index issues
        i = 0
        while i < len(op_subs):
            sub = op_subs[i]

            if tot + sub.num_qubits <= mq:
                current_group.append(sub)
                op_subs.remove(sub)
                tot += sub.num_qubits
                # Don't increment i since we removed an element
            else:
                i += 1  # Only increment if we didn't remove an element

        groups.append(current_group)

    return groups


def combine_subcircuits(subcircuits):
    # Count clbits per category
    def is_qpd_reg(name: str) -> bool:
        return name.startswith("qpd")

    total_q = sum(sc.num_qubits for sc in subcircuits)
    total_meas = sum(
        sum(1 for c in sc.clbits if c._register and c._register.name == "meas")
        for sc in subcircuits
    )
    total_qpd = sum(
        sum(1 for c in sc.clbits if c._register and is_qpd_reg(c._register.name))
        for sc in subcircuits
    )

    qr = QuantumRegister(total_q, "q")
    qpd = ClassicalRegister(total_qpd, "qpd_meas")
    meas = ClassicalRegister(total_meas, "meas")
    out = QuantumCircuit(qr, qpd, meas)

    qcur = 0
    qpd_cur = 0
    meas_cur = 0

    # Compose largest first (optional)
    for sc in sorted(subcircuits, key=lambda x: x.num_qubits, reverse=True):
        # Qubit mapping
        qmap = list(range(qcur, qcur + sc.num_qubits))
        qcur += sc.num_qubits

        # Clbit mapping in the same order as sc.clbits
        cmap = []
        used_qpd = 0
        used_meas = 0
        for c in sc.clbits:
            if is_qpd_reg(c._register.name if c._register else ""):
                cmap.append(qpd[qpd_cur + used_qpd])
                used_qpd += 1
            else:  # "meas", and any other register with it
                cmap.append(meas[meas_cur + used_meas])
                used_meas += 1

        out.compose(sc, qubits=qmap, clbits=cmap, inplace=True)
        qpd_cur += used_qpd
        meas_cur += used_meas

    return out


def construct_final_subcircuits(subcircuits, max_qubits):
    if len(subcircuits) <= len(max_qubits):
        # If the number of subcircuits matches the number of max_qubits, return
        # them as is
        return subcircuits
    return [
        group[0] if len(group) == 1 else combine_subcircuits(group)
        for group in group_circuits(subcircuits, max_qubits)
    ]
