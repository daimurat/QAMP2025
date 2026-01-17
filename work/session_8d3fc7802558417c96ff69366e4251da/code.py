from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit import QuantumCircuit


def create_product_formula_circuit(
    pauli_strings: list,
    times: list,
    order: int,
    reps: int,
) -> QuantumCircuit:
    """
    Create a quantum circuit using Lie‑Trotter product‑formula synthesis.

    Parameters
    ----------
    pauli_strings : list
        List of Pauli strings (e.g. "XIZ").
    times : list
        List of evolution times (coefficients) corresponding to each Pauli string.
    order : int
        Order of the Lie‑Trotter product formula (must be a positive integer).
    reps : int
        Number of repetitions (time steps) of the product formula
        (must be a positive integer).

    Returns
    -------
    QuantumCircuit
        The synthesized circuit implementing the evolution.

    Raises
    ------
    ValueError
        If any input validation fails.
    """
    # ---- Input validation -------------------------------------------------
    if not isinstance(pauli_strings, list) or not isinstance(times, list):
        raise ValueError("Both 'pauli_strings' and 'times' must be lists.")

    if len(pauli_strings) == 0:
        raise ValueError("'pauli_strings' list cannot be empty.")

    if len(pauli_strings) != len(times):
        raise ValueError("'pauli_strings' and 'times' must have the same length.")

    for idx, p in enumerate(pauli_strings):
        if not isinstance(p, str):
            raise ValueError(f"Pauli string at index {idx} is not a string.")

    for idx, t in enumerate(times):
        if not isinstance(t, (int, float)):
            raise ValueError(f"Time at index {idx} is not a numeric type.")
        if t < 0:
            raise ValueError(f"Time at index {idx} is negative; must be non‑negative.")

    if not isinstance(order, int) or order <= 0:
        raise ValueError("'order' must be a positive integer.")

    if not isinstance(reps, int) or reps <= 0:
        raise ValueError("'reps' must be a positive integer.")

    # ---- Build the operator ------------------------------------------------
    # Each term is (pauli_string, coefficient) where coefficient = time
    term_list = [(pauli_strings[i], times[i]) for i in range(len(pauli_strings))]
    sparse_op = SparsePauliOp.from_list(term_list)

    # ---- Create the evolution gate (overall time set to 1.0) ---------------
    evolution_gate = PauliEvolutionGate(sparse_op, time=1.0)

    # ---- Synthesize using Lie‑Trotter --------------------------------------
    lt = LieTrotter(order=order, reps=reps)
    circuit = lt.synthesize(evolution_gate)

    return circuit