from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import AerSimulator

def random_coin_flip(shots: int) -> dict:
    """
    Simulate a quantum coin flip by preparing a single qubit in superposition
    and measuring it multiple times.

    Parameters
    ----------
    shots : int
        Number of repeated measurements (must be a positive integer).

    Returns
    -------
    dict
        Dictionary with keys 'Heads' and 'Tails' containing the aggregated
        counts of the |0⟩ and |1⟩ outcomes respectively.
    """
    if not isinstance(shots, int) or shots <= 0:
        raise ValueError("'shots' must be a positive integer.")

    # Build circuit: Hadamard followed by measurement
    qc = QuantumCircuit(1, 1)
    qc.h(0)          # superposition
    qc.measure(0, 0) # collapse to |0> or |1>

    # Run on local Aer simulator
    backend = AerSimulator()
    job = execute(qc, backend, shots=shots, memory=False)
    counts = job.result().get_counts()

    # Map |0> → 'Heads', |1> → 'Tails'
    heads = counts.get('0', 0)
    tails = counts.get('1', 0)
    return {'Heads': heads, 'Tails': tails}