from qiskit import QuantumCircuit
import numpy as np

def create_cy_gate()->QuantumCircuit:
    """ Design a CY gate using only one CX gate and any other single qubit gates.
    """
    # Create a 2-qubit circuit
    qc = QuantumCircuit(2)
    
    # Apply S gate to control qubit (qubit 0)
    qc.s(0)
    
    # Apply CX gate with control on qubit 0 and target on qubit 1
    qc.cx(0, 1)
    
    # Apply Sdg gate to control qubit (qubit 0)
    qc.sdg(0)
    
    # Apply S gate to target qubit (qubit 1)
    qc.s(1)
    
    # Verify the unitary matrix
    from qiskit.quantum_info import Operator
    
    # Get the unitary matrix of the circuit
    op = Operator(qc)
    unitary = op.data
    
    # CY gate should have matrix diag([1,1,1,-1])
    expected = np.diag([1, 1, 1, -1])
    
    # Check if the unitary matches the expected CY gate
    if not np.allclose(unitary, expected):
        raise ValueError(f"Circuit does not implement CY gate. Expected diag([1,1,1,-1]), got unitary:\n{unitary}")
    
    return qc