from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler
from qiskit.primitives.containers.primitive_result import PrimitiveResult

def bv_algorithm(s: str) -> [list, PrimitiveResult]:
    # Create AerSimulator backend but don't pass it to Sampler
    backend = AerSimulator()
    
    # Initialize Sampler without backend parameter
    sampler = Sampler()
    
    # Create BV algorithm circuit
    n = len(s)
    qc = QuantumCircuit(n + 1, n)
    
    # Apply Hadamard to all qubits
    for i in range(n + 1):
        qc.h(i)
    
    # Apply X to the last qubit
    qc.x(n)
    
    # Apply Hadamard to the last qubit
    qc.h(n)
    
    # Apply U_f (controlled-X based on s)
    for i in range(n):
        if s[i] == '1':
            qc.cx(i, n)
    
    # Apply Hadamard to all qubits except last
    for i in range(n):
        qc.h(i)
    
    # Measure the first n qubits
    for i in range(n):
        qc.measure(i, i)
    
    # Run sampler without backend parameter
    job = sampler.run(qc, shots=1024)
    result = job.result()
    
    # Extract bit strings from quasi_dists
    if hasattr(result, 'quasi_dists') and len(result.quasi_dists) > 0:
        quasi_dist = result.quasi_dists[0]
        bitstrings = list(quasi_dist.binary_probabilities().keys())
    else:
        bitstrings = []
    
    return bitstrings, result