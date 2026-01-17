from qiskit import QuantumCircuit
from qiskit.circuit.library import YGate

def mcy(qc: QuantumCircuit) -> QuantumCircuit:
    """ Add a multi-controlled-Y operation to qubit 4, controlled by qubits 0-3.
    """
    # Create a multi-controlled Y gate with 4 control qubits
    mcy_gate = YGate().control(num_ctrl_qubits=4)
    
    # Apply the gate to the circuit: controls on qubits 0-3, target on qubit 4
    qc.append(mcy_gate, [qc.qubits[0], qc.qubits[1], qc.qubits[2], qc.qubits[3], qc.qubits[4]])
    
    return qc