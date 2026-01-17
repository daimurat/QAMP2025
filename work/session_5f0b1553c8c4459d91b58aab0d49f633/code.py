from qiskit import QuantumCircuit
from qiskit.circuit.library import Barrier


def split_circuit_at_barriers(circuit: QuantumCircuit) -> list[QuantumCircuit]:
    """
    Split a ``QuantumCircuit`` into a list of sub‑circuits at each barrier.

    The barriers themselves are omitted from the returned sub‑circuits.
    Each sub‑circuit retains the original quantum and classical registers,
    allowing it to be executed independently.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to be split.

    Returns
    -------
    list[QuantumCircuit]
        A list of sub‑circuits containing the operations that were between
        consecutive barriers. Empty segments (e.g. caused by consecutive
        barriers) are omitted.
    """
    # Helper to create an empty circuit with the same registers, metadata, etc.
    def _empty_like(orig: QuantumCircuit) -> QuantumCircuit:
        return orig.copy_empty_like()

    subcircuits: list[QuantumCircuit] = []
    current = _empty_like(circuit)

    for instr in circuit.data:
        # ``instr`` is a ``CircuitInstruction``; its operation is accessed via .operation
        if isinstance(instr.operation, Barrier):
            # End the current segment (if it has any instructions) and start a new one
            if len(current.data) > 0:
                subcircuits.append(current)
                current = _empty_like(circuit)
            # If the current segment is already empty, just continue (skip the barrier)
            continue

        # Append the instruction to the current sub‑circuit
        # ``instr.qubits`` and ``instr.clbits`` are tuples of the target bits
        current.append(instr.operation, instr.qubits, instr.clbits)

    # Append the final segment if it contains any instructions
    if len(current.data) > 0:
        subcircuits.append(current)

    return subcircuits