from qiskit.dagcircuit import DAGCircuit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag

def bell_dag() -> DAGCircuit:
    """ Construct a DAG circuit for a 3-qubit Quantum Circuit with the bell state applied on qubit 0 and 1. Finally return the DAG Circuit object.
    """
    # Create a QuantumCircuit with 3 qubits and 3 classical bits
    q = QuantumRegister(3, 'q')
    c = ClassicalRegister(3, 'c')
    circ = QuantumCircuit(q, c)
    
    # Apply Hadamard gate to qubit 0
    circ.h(q[0])
    
    # Apply CX gate from qubit 0 to qubit 1 to create Bell state
    circ.cx(q[0], q[1])
    
    # Convert the circuit to DAG using circuit_to_dag
    dag = circuit_to_dag(circ)
    
    # Return the DAGCircuit object
    return dag