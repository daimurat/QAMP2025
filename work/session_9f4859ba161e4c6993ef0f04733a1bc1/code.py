from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit.library import OrGate

def or_gate(a: int, b: int) -> dict:
    """ Given two 3-bit integers a and b, design a quantum circuit that acts as a classical OR gate. Simulate the circuit using Qiskit Sampler with the Aer simulator as backend and return the counts of the result.
    """
    
    # Convert integers to 3-bit binary strings
    a_bits = format(a, '03b')
    b_bits = format(b, '03b')
    
    # Create quantum registers: 3 for a, 3 for b, and 3 for result bits
    a_reg = QuantumRegister(3, 'a')
    b_reg = QuantumRegister(3, 'b')
    result_reg = QuantumRegister(3, 'result')
    c_reg = ClassicalRegister(3, 'c')
    
    # Create circuit
    qc = QuantumCircuit(a_reg, b_reg, result_reg, c_reg)
    
    # Initialize input bits based on the integers a and b
    for i in range(3):
        if a_bits[i] == '1':
            qc.x(a_reg[i])
        if b_bits[i] == '1':
            qc.x(b_reg[i])
    
    # Apply OR gate to each bit pair
    for i in range(3):
        # Use OrGate for each bit pair (a[i] OR b[i] -> result[i])
        or_gate_single = OrGate(num_variable_qubits=2)
        qc.append(or_gate_single, [a_reg[i], b_reg[i], result_reg[i]])
    
    # Measure the result bits
    for i in range(3):
        qc.measure(result_reg[i], c_reg[i])
    
    # Setup backend and sampler
    backend = AerSimulator()
    sampler = Sampler(mode=backend)
    
    # Run the sampler
    result = sampler.run(circuits=[qc], shots=1024).result()
    
    # Extract counts from quasi_dists
    counts = {}
    if hasattr(result, 'quasi_dists') and len(result.quasi_dists) > 0:
        quasi_dist = result.quasi_dists[0]
        # Convert quasi-probabilities to counts by multiplying by shots and rounding
        shots = 1024
        for bitstring, prob in quasi_dist.items():
            counts[bitstring] = int(prob * shots)
    
    return counts