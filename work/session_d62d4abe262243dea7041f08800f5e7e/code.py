from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.classical import expr

def gate_if_clbits(
    circuit: QuantumCircuit, gate: Gate, qubits: list[int], condition_clbits: list[int]
) -> None:
    """ Apply `gate` to qubits with indices `qubits`, conditioned on all `condition_clbits` being 1.
    """
    # Build a condition that is True only if all specified clbits are 1
    cond = expr.bit_and(*[circuit.clbits[i] for i in condition_clbits])
    
    with circuit.if_test((cond, True)):
        circuit.append(gate, qubits)