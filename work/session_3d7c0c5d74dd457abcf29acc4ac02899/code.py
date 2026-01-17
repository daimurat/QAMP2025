from qiskit import QuantumCircuit
from qiskit.quantum_info import random_clifford, Operator
import numpy as np

def equivalent_clifford_circuit(circuit: QuantumCircuit, n: int) -> list:
    """ Given a clifford circuit return a list of n random clifford circuits which are equivalent to the given circuit up to a relative and absolute tolerance of 0.4.
    """
    if n <= 0:
        return []
    
    # Convert input circuit to Clifford object
    from qiskit.quantum_info import Clifford
    target_clifford = Clifford(circuit)
    target_operator = target_clifford.to_operator()
    
    num_qubits = circuit.num_qubits
    equivalent_circuits = []
    
    # Maximum iterations to avoid infinite loop
    max_attempts = 1000 * n
    attempts = 0
    
    while len(equivalent_circuits) < n and attempts < max_attempts:
        attempts += 1
        
        # Generate a random Clifford of same number of qubits
        rand_clifford = random_clifford(num_qubits)
        rand_operator = rand_clifford.to_operator()
        
        # Compare operators with specified tolerance
        if np.allclose(target_operator.data, rand_operator.data, rtol=0.4, atol=0.4):
            # Convert the random Clifford back to a circuit
            equiv_circuit = rand_clifford.to_circuit()
            equivalent_circuits.append(equiv_circuit)
    
    return equivalent_circuits