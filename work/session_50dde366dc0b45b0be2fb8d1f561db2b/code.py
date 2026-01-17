from qiskit import QuantumCircuit
from numpy import pi

def chsh_circuit(alice: int, bob: int) -> QuantumCircuit:
    """
    Construct a CHSH game circuit.

    Parameters
    ----------
    alice : int
        Alice's input bit (0 or 1). If 1, an additional Hadamard gate is applied
        to Alice's qubit before measurement.
    bob : int
        Bob's input bit (0 or 1). Determines the rotation applied to Bob's qubit:
        * 0 → RY(π/4)
        * 1 → RY(-π/4)

    Returns
    -------
    QuantumCircuit
        A circuit with 2 quantum bits and 2 classical bits, named
        ``"chsh_circuit"``, that prepares a Bell state, applies the
        appropriate basis‑change gates, and measures qubit 0 into classical
        bit 0 and qubit 1 into classical bit 1.
    """
    # Validate inputs (optional but helpful)
    if alice not in (0, 1):
        raise ValueError("alice must be 0 or 1")
    if bob not in (0, 1):
        raise ValueError("bob must be 0 or 1")

    # Create a 2‑qubit, 2‑classical‑bit circuit
    circuit = QuantumCircuit(2, 2)
    circuit.name = "chsh_circuit"

    # Bell state preparation
    circuit.h(0)
    circuit.cx(0, 1)

    # Alice's measurement basis change
    if alice == 1:
        circuit.h(0)

    # Bob's measurement basis change
    if bob == 0:
        circuit.ry(pi / 4, 1)
    else:  # bob == 1
        circuit.ry(-pi / 4, 1)

    # Measurements (preserve order: qubit 0 → c0, qubit 1 → c1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)

    return circuit