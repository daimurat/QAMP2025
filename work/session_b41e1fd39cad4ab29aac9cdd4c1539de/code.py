from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.synthesis import LieTrotter
import numpy as np


def create_product_formula_circuit(
    pauli_strings: list, times: list, order: int, reps: int
) -> QuantumCircuit:
    """
    Create a quantum circuit using LieTrotter for a list of Pauli strings and times.

    Parameters
    ----------
    pauli_strings : list
        List of Pauli strings (e.g. ["XIZ", "YYI"]).
    times : list
        List of evolution times (or coefficients) corresponding to each Pauli string.
    order : int
        Order of the Suzuki‑Trotter expansion (must be a positive integer).
    reps : int
        Number of repetitions of the product formula (must be a positive integer).

    Returns
    -------
    QuantumCircuit
        Circuit implementing the product‑formula evolution.

    Raises
    ------
    ValueError
        If the input lists have mismatched lengths or if ``order``/``reps`` are not
        positive integers.
    """
    # ---- Input validation -------------------------------------------------
    if len(pauli_strings) != len(times):
        raise ValueError(
            "The number of Pauli strings must match the number of times provided."
        )
    if not isinstance(order, int) or order <= 0:
        raise ValueError("`order` must be a positive integer.")
    if not isinstance(reps, int) or reps <= 0:
        raise ValueError("`reps` must be a positive integer.")

    # ---- Convert strings to Pauli objects (validation) --------------------
    # This also checks that all strings have the same length (same number of qubits)
    paulis = [Pauli(label) for label in pauli_strings]

    # ---- Build the SparsePauliOp with the given coefficients ---------------
    coeffs = np.array(times, dtype=complex)
    # SparsePauliOp can be constructed directly from a list of labels and coeffs
    hamiltonian = SparsePauliOp(pauli_strings, coeffs=coeffs)

    # ---- Create a PauliEvolutionGate (overall time = 1, coefficients already encode the times)
    evolution_gate = PauliEvolutionGate(hamiltonian, 1.0)

    # ---- Initialise the Lie‑Trotter product formula -----------------------
    # LieTrotter inherits from SuzukiTrotter, which accepts `order` and `reps`.
    trotter = LieTrotter(order=order, reps=reps)

    # ---- Synthesize the circuit -------------------------------------------
    circuit = trotter.synthesize(evolution_gate)

    return circuit