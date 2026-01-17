from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2

def random_coin_flip(samples: int) -> dict:
    """ Design a Quantum Circuit that simulates random coin flips for the given samples using Qiskit Sampler with the Aer simulator as backend and outputs the count of heads and tails in a dictionary. The heads should be stored in the dict as 'Heads' and tails as 'Tails'. For example
    random_coin_flip(10) == {'Heads' : 5, 'Tails' : 5}
    random_coin_flip(20) == {'Heads' : 9, 'Tails': 11}.
    """
    # Create a single-qubit circuit with H gate and measurement
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Create superposition state
    qc.measure(0, 0)  # Measure and store result in classical bit
    
    # Initialize AerSimulator as backend
    backend = AerSimulator()
    
    # Create sampler with backend
    sampler = BackendSamplerV2(backend=backend)
    
    # Run the sampler with specified number of shots
    job = sampler.run([qc], shots=samples)
    result = job.result()
    
    # Get counts from result
    counts = result[0].join_data().get_counts()
    
    # Process results: 0 -> Heads, 1 -> Tails
    heads = counts.get('0', 0)
    tails = counts.get('1', 0)
    
    return {'Heads': heads, 'Tails': tails}