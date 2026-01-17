from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Operator
import numpy as np

def equivalent_clifford_circuit(circuit: QuantumCircuit, n: int) -> list:
    """ Given a clifford circuit return a list of n random clifford circuits which are equivalent to the given circuit up to a relative and absolute tolerance of 0.4.
    """
    num_qubits = circuit.num_qubits
    if num_qubits == 0:
        return []
    
    target_op = Operator(circuit)
    equivalent_circuits = []
    attempts = 0
    max_attempts = n * 100  # reasonable upper bound to avoid infinite loops
    
    while len(equivalent_circuits) < n and attempts < max_attempts:
        attempts += 1
        # Generate a random Clifford operator and convert to a circuit
        cliff = random_clifford(num_qubits)
        random_circ = cliff.to_circuit()
        
        # Convert the random circuit to an operator
        random_op = Operator(random_circ)
        
        # Compare operators with tolerance 0.4 for both relative and absolute tolerance
        if np.allclose(target_op.data, random_op.data, rtol=0.4, atol=0.4):
            equivalent_circuits.append(random_circ)
    
    return equivalent_circuits