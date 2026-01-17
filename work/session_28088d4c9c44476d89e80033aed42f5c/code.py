from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler

def random_coin_flip(samples: int) -> dict:
    """ Design a Quantum Circuit that simulates random coin flips for the given samples using Qiskit Sampler with the Aer simulator as backend and outputs the count of heads and tails in a dictionary. The heads should be stored in the dict as 'Heads' and tails as 'Tails'. For example
    random_coin_flip(10) == {'Heads' : 5, 'Tails : 5}
    random_coin_flip(20) == {'Heads' : 9, 'Tails : 11}.
    """
    # Create a single-qubit quantum circuit
    qc = QuantumCircuit(1, 1)
    
    # Apply Hadamard gate to create superposition (fair coin flip)
    qc.h(0)
    
    # Measure the qubit
    qc.measure(0, 0)
    
    # Use AerSimulator as backend
    backend = AerSimulator()
    
    # Create sampler with backend
    sampler = Sampler(mode=backend)
    
    # Run the circuit with specified number of shots
    job = sampler.run([(qc, [], samples)])
    result = job.result()
    
    # Get counts from the result
    # For SamplerV2 with measure_all, the register is named 'meas'
    # But since we used measure(0, 0), we need to use the data from the result
    counts = result[0].data.get_counts()
    
    # Process results: 0 -> Heads, 1 -> Tails
    heads_count = counts.get(0, 0)
    tails_count = counts.get(1, 0)
    
    return {'Heads': heads_count, 'Tails': tails_count}