from qiskit import QuantumCircuit, transpile
from qiskit.providers.basic_provider import BasicProvider
from qiskit.transpiler import PassManager
import numpy as np

def random_coin_flip(samples: int) -> dict:
    """
    Generate a sequence of fair coin flips using a quantum superposition.

    Each flip is obtained by placing a qubit in the state (|0⟩+|1⟩)/√2, measuring,
    and mapping 0→Heads, 1→Tails.  The experiment is repeated `samples` times.

    Parameters
    ----------
    samples : int
        Number of coin flips to perform.

    Returns
    -------
    dict
        A dictionary with keys 'Heads' and 'Tails' containing the observed counts.

    Raises
    ------
    ValueError
        If `samples` is not a non-negative integer.
    """
    if not isinstance(samples, int) or samples < 0:
        raise ValueError("`samples` must be a non-negative integer.")

    backend = BasicProvider().get_backend('basic_simulator')
    counts_dict = {"0": 0, "1": 0}

    for _ in range(samples):
        qc = QuantumCircuit(1, 1)
        qc.h(0)            # Superposition  (|0⟩+|1⟩)/√2
        qc.measure(0, 0)   # Measure
        job = backend.run(transpile(qc, backend), shots=1)
        counts = job.result().get_counts()
        for key in counts:
            counts_dict[key] += counts[key]

    return {
        "Heads": counts_dict["0"],
        "Tails": counts_dict["1"]
    }