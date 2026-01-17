from qiskit import QuantumCircuit
from qiskit_ibm_runtime import Sampler
from qiskit_aer import AerSimulator

sampler = Sampler(backend=AerSimulator())

def random_coin_flip(samples: int) -> dict[str,int]:
    """
    Quantum coin flip returning 'Heads'/'Tails' counts summing to samples.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    job = sampler.run(qc, shots=samples)
    dist = job.result().quasi_dists[0].binary_probabilities()
    
    heads = round(dist.get("1", 0.0) * samples)
    tails = samples - heads
    
    return {"Heads": heads, "Tails": tails}