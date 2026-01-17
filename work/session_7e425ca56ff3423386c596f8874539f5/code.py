from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler

def bb84_circuit_generate_key(sender_basis: list[int], receiver_basis: list[int], circuit: QuantumCircuit) -> str:
    # Measure all qubits
    circuit.measure_all()

    # Simulate the circuit
    backend = AerSimulator()
    sampler = Sampler(options={'backend': backend})
    job = sampler.run(circuit, shots=1)
    result = job.result()
    counts = result.quasi_dists[0].binary_probabilities()
    outcome = max(counts, key=counts.get)

    # Filter bits where sender_basis[i] == receiver_basis[i]
    filtered_bits = [outcome[i] for i in range(len(sender_basis)) if sender_basis[i] == receiver_basis[i]]
    return ''.join(filtered_bits)