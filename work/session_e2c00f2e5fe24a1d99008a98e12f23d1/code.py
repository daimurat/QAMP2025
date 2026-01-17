from qiskit import QuantumCircuit
from qiskit.circuit.library import YGate

def build_circuit() -> QuantumCircuit:
    """Create a QuantumCircuit with a multi-controlled Y gate on qubit 4 controlled by qubits 0-3."""
    qc = QuantumCircuit(5)
    mcy = YGate().control(num_ctrl_qubits=4)
    qc.append(mcy, [0, 1, 2, 3, 4])
    return qc

if __name__ == "__main__":
    circuit = build_circuit()
    print(circuit.draw())