from qiskit import QuantumCircuit
import numpy as np

def for_loop_circuit(qc: QuantumCircuit, n: int) -> QuantumCircuit:
    """
    Build a circuit with a for_loop that iterates at most n times.
    Each iteration applies RY, H, CX, and measurement, and breaks
    early when the measurement of qubit 0 yields 1.
    """
    with qc.for_loop(range(n)) as i:
        qc.ry(np.pi / n * i, 0)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, i)
        qc.break_loop().c_if(i, 1)
    return qc