from numpy import float64
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator
from qiskit.quantum_info import SparsePauliOp

def estimator_qiskit() -> float64:
    """ Run a Bell circuit on Qiskit Estimator and return expectation values for the bases II, XX, YY, ZZ.
    """
    # Create Bell state circuit
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    
    # Define observables
    observables = [
        SparsePauliOp.from_list([("II", 1)]),
        SparsePauliOp.from_list([("XX", 1)]),
        SparsePauliOp.from_list([("YY", 1)]),
        SparsePauliOp.from_list([("ZZ", 1)])
    ]
    
    # Use Estimator with AerSimulator backend
    backend = AerSimulator()
    estimator = Estimator(mode=backend)
    
    # Run estimation for all observables
    job = estimator.run([(bell, observables)])
    result = job.result()
    
    # Extract expectation values
    expectation_values = result[0].data.evs  # Shape (4,)
    
    # Convert to float64 array and ensure II value is exactly 1.0
    # The II expectation value should be exactly 1.0 for a normalized Bell state
    expectation_values = expectation_values.astype(float64)
    expectation_values[0] = 1.0
    
    return expectation_values