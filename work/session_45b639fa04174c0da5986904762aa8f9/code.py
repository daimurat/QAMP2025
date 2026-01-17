from numpy import float64
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator
from qiskit.quantum_info import SparsePauliOp

def estimator_qiskit() -> float64:
    """ Run a Bell circuit on Qiskit Estimator and return expectation values for the bases II, XX, YY, ZZ.
    """
    # Create Bell state circuit
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    
    # Define observables
    observables = [
        SparsePauliOp("II"),  # Identity on both qubits
        SparsePauliOp("XX"),  # Pauli-X on both qubits
        SparsePauliOp("YY"),  # Pauli-Y on both qubits
        SparsePauliOp("ZZ")   # Pauli-Z on both qubits
    ]
    
    # Use local estimator for simulation (no cloud backend needed)
    from qiskit.primitives import StatevectorEstimator
    estimator = StatevectorEstimator()
    
    # Calculate expectation values
    job = estimator.run([(bell_circuit, observables)])
    result = job.result()
    
    # Extract expectation values
    expectation_values = result[0].data.evs
    
    # Return all four expectation values as tuple of float64
    return tuple(float64(ev) for ev in expectation_values)