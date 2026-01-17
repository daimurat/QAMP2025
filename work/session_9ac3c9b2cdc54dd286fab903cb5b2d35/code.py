from qiskit import QuantumCircuit

def create_quantum_circuit(n_qubits):
    """ Generate a Quantum Circuit for the given int 'n_qubits' and return it.
    """
    # Validate the input
    if not isinstance(n_qubits, int):
        raise ValueError(f"`n_qubits` must be an integer, got {type(n_qubits).__name__}.")
    if n_qubits <= 0:
        raise ValueError(f"`n_qubits` must be a positive integer, got {n_qubits}.")

    # Create and return the circuit with the requested number of qubits
    return QuantumCircuit(n_qubits)


# ----------------------------------------------------------------------
# Example usage (will run when this script is executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a 3‑qubit circuit and display its basic info
    qc = create_quantum_circuit(3)
    print(qc)
    # Visual representation (text drawing)
    print(qc.draw())