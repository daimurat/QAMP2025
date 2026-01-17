from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp


def create_product_formula_circuit(
    pauli_strings: list, times: list, order: int, reps: int
) -> QuantumCircuit:
    """
    Create a quantum circuit using a product‑formula (Lie‑Trotter or Suzuki‑Trotter)
    for a list of Pauli strings and their associated evolution times.

    Parameters
    ----------
    pauli_strings : list
        List of Pauli strings, e.g. ["XIZY", "ZZII"].
    times : list
        List of evolution times (float) corresponding to each Pauli string.
    order : int
        Order of the product formula. ``order == 1`` uses Lie‑Trotter,
        ``order > 1`` uses Suzuki‑Trotter.
    reps : int
        Number of repetitions (time steps) of the formula.

    Returns
    -------
    QuantumCircuit
        Circuit implementing the specified product‑formula evolution.
    """
    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------
    if len(pauli_strings) != len(times):
        raise ValueError(
            f"Length mismatch: {len(pauli_strings)} Pauli strings but {len(times)} times provided."
        )
    if order < 1:
        raise ValueError("Order must be a positive integer (>= 1).")
    if reps < 1:
        raise ValueError("Reps must be a positive integer (>= 1).")

    # -----------------------------------------------------------------
    # Convert Pauli strings to SparsePauliOp objects (list of operators)
    # -----------------------------------------------------------------
    # SparsePauliOp can be constructed directly from an iterable of strings.
    operators = [SparsePauliOp(s) for s in pauli_strings]

    # -----------------------------------------------------------------
    # Choose the appropriate synthesiser (LieTrotter or SuzukiTrotter)
    # -----------------------------------------------------------------
    if order == 1:
        synthesiser = LieTrotter(operators, times, reps=reps)
    else:
        synthesiser = SuzukiTrotter(operators, times, order=order, reps=reps)

    # -----------------------------------------------------------------
    # Build a dummy PauliEvolutionGate – only its qubit count is needed.
    # The actual evolution is dictated by the synthesiser's operators/times.
    # -----------------------------------------------------------------
    # Combine the Pauli terms into a single operator (coefficients = 1).
    combined_operator = SparsePauliOp(pauli_strings)
    evolution_gate = PauliEvolutionGate(combined_operator, time=1.0)

    # -----------------------------------------------------------------
    # Synthesize the circuit
    # -----------------------------------------------------------------
    circuit = synthesiser.synthesize(evolution_gate)

    return circuit