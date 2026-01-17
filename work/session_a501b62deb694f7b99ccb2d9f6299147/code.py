from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator


def random_coin_flip(samples: int):
    """Quantum random coin flip using a single qubit."""
    # Create circuit: 1 qubit, 1 classical bit
    qc = QuantumCircuit(1, 1)

    # Apply Hadamard to create equal superposition
    qc.h(0)

    # Measure the qubit
    qc.measure(0, 0)

    # Use AerSimulator and Sampler
    backend = AerSimulator()
    sampler = Sampler(options={'backend': backend})

    job = sampler.run(qc, shots=samples)
    result = job.result()

    counts = result.quasi_dists[0].binary_probabilities()

    # Convert counts to 'Heads'/'Tails' dictionary
    outcome = {
        'Heads': int(counts.get('0', 0) * samples),
        'Tails': int(counts.get('1', 0) * samples)
    }

    # Correct for rounding errors by adjusting the largest count
    total = sum(outcome.values())
    if total != samples:
        diff = samples - total
        if outcome['Heads'] > outcome['Tails']:
            outcome['Heads'] += diff
        else:
            outcome['Tails'] += diff

    return outcome