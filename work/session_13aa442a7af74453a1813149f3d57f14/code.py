from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler

def random_coin_flip(samples: int) -> dict:
    """
    Simulate a random coin flip using a quantum single-qubit circuit.
    
    Args:
        samples (int): Number of times to run the circuit and collect outcomes.
    
    Returns:
        dict: Dictionary with keys 'Heads' and 'Tails' mapping to integer counts.
    """
    # Create a single-qubit quantum circuit
    qc = QuantumCircuit(1, 1)
    
    # Apply a Hadamard gate to put the qubit in equal superposition
    qc.h(0)
    
    # Measure the qubit in the computational (Z) basis
    qc.measure(0, 0)
    
    # Use Qiskit's Sampler to execute the circuit
    sampler = Sampler(backend_options={"method": "statevector"})
    
    # Collect samples
    job = sampler.run([qc], shots=samples)
    result = job.result()
    
    # Get the quasi-probability distribution, convert to binary counts
    counts = result.quasi_dists[0].nearest_probability_distribution().binary_probabilities(samples)
    
    # Convert counts dictionary keys from integers to 'Heads'/'Tails'
    outcomes = {'Heads': counts.get(0, 0), 'Tails': counts.get(1, 0)}
    
    return outcomes