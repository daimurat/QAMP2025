from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler

def random_coin_flip(samples=1):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    
    backend = AerSimulator()
    sampler = Sampler(backend=backend)
    result = sampler.run([qc], shots=samples).result()
    counts_raw = result.quasi_dists[0].binary_probabilities()
    
    counts = {'Heads': 0, 'Tails': 0}
    for bit, c in counts_raw.items():
        if bit == '0':
            counts['Heads'] += int(c * samples)
        elif bit == '1':
            counts['Tails'] += int(c * samples)
    
    total = sum(counts.values())
    if total < samples:
        remainder = samples - total
        if '1' in counts_raw:
            counts['Tails'] += remainder
        else:
            counts['Heads'] += remainder
    
    return counts