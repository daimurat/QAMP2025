from qiskit.quantum_info import Operator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp

def create_product_formula_circuit(pauli_strings: list, times: list, order: int, reps: int) -> QuantumCircuit:
    """ Create a quantum circuit using LieTrotter for a list of Pauli strings and times. Each Pauli string is associated with a corresponding time in the 'times' list. The function should return the resulting QuantumCircuit.
    """
    # Create SparsePauliOp from pauli strings and times
    if len(pauli_strings) != len(times):
        raise ValueError("pauli_strings and times must have the same length")
    
    # Combine pauli strings with their corresponding times as coefficients
    pauli_list = [(pauli_str, time) for pauli_str, time in zip(pauli_strings, times)]
    sparse_pauli_op = SparsePauliOp.from_list(pauli_list)
    
    # Determine number of qubits from the pauli strings
    num_qubits = len(pauli_strings[0]) if pauli_strings else 0
    
    # Create LieTrotter synthesis with specified order and reps
    lie_trotter = LieTrotter(order=order, reps=reps)
    
    # Create PauliEvolutionGate with total time evolution
    # Since times are already incorporated as coefficients, use time=1.0
    evolution_gate = PauliEvolutionGate(sparse_pauli_op, time=1.0, synthesis=lie_trotter)
    
    # Create quantum circuit with appropriate number of qubits
    circuit = QuantumCircuit(num_qubits)
    
    # Apply the evolution gate to all qubits
    circuit.append(evolution_gate, range(num_qubits))
    
    return circuit