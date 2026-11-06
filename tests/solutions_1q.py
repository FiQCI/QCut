from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp

from QCut import cut
from QCut.model.cutlocation import SingleQubitCutLocation as CutLocation

qc_1 = QuantumCircuit(3)
qc_1.x(0)
qc_1.cx(0,1)
qc_1.append(cut(), [1])
qc_1.cx(1,2)

qc_2 = QuantumCircuit(4)
qc_2.x(0)
qc_2.cx(0,1)
qc_2.cx(0,2)
qc_2.append(cut(), [1])
qc_2.append(cut(), [2])
qc_2.cx(1,3)
qc_2.cz(2,3)

qc_3 = QuantumCircuit(4)
qc_3.x(0)
qc_3.cx(0,1)
qc_3.append(cut(), [1])
qc_3.cx(1,2)
qc_3.append(cut(), [2])
qc_3.cz(2,3)

qc_4 = QuantumCircuit(5)
qc_4.h(0)
qc_4.cx(1,2)
qc_4.cx(2,3)
qc_4.append(cut(), [1])
qc_4.append(cut(), [3])
qc_4.cx(0,1)
qc_4.cx(3,4)
qc_4.cx(0,4)

qc_5 = QuantumCircuit(4)
qc_5.h(0)
qc_5.cx(1,2)
qc_5.append(cut(), [1])
qc_5.append(cut(), [2])
qc_5.cx(0,1)
qc_5.cx(2,3)
qc_5.cx(0,3)

qc_6 = QuantumCircuit(3)
qc_6.h(0)
qc_6.cx(0, 1)
qc_6.append(cut(), [1])
qc_6.cx(1, 2)
qc_6.append(cut(), [1])
qc_6.cx(0, 1)

test_circuits = [qc_1, qc_2, qc_3, qc_4, qc_5, qc_6]

cut_location_solutions = [[CutLocation(((QuantumRegister(3, "q"), 1), 2))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 3)), CutLocation(((QuantumRegister(4, "q"), 2), 3))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 2)), CutLocation(((QuantumRegister(4, "q"), 2), 3))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 3)), CutLocation(((QuantumRegister(4, "q"), 3), 3))],
                          [CutLocation(((QuantumRegister(4, "q"), 1), 2)), CutLocation(((QuantumRegister(4, "q"), 2), 2))],
                          [CutLocation(((QuantumRegister(3, "q"), 1), 2)), CutLocation(((QuantumRegister(3, "q"), 1), 3))],
                        ]



number_of_subcircuits = [2,2,3,2,2,2]
subcircuit_len = [
    [
        3,2
    ],
    [
        5, 4
    ],
    [
        3,3,2
    ],
    [
        6,4
    ],
    [
        6,3
    ],
    [
        5,3
    ]
]

test_observables = [
    ['IIZ', 'IZI', 'ZII', 'ZIZ', 'ZZI', 'ZZZ'],
    ['IIIZ', 'IIZI', 'IZII', 'ZIII', 'ZIIZ', 'ZZZI', 'ZZZZ'],
    ['IIIZ', 'IIZI', 'IZII', 'ZIII', 'ZIIZ', 'ZZZI', 'ZZZZ'],
    ['IIIIZ', 'IIIZI', 'IIZII', 'IIZIZ', 'ZZIIZ'],
    ['IIIZ', 'IIZI', 'IZII', 'ZIII', 'IZIZ', 'ZIZZ'],
    ['IIZ', 'IZI', 'ZII', 'ZIZ', 'IZZ'],
]

exp_val_solutions = [
    [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    [0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0],
]