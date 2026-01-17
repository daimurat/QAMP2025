from numpy import float64, array
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp

def estimator_qiskit() -> float64:
    """ Run a Bell circuit on Qiskit Estimator and return expectation values for the bases II, XX, YY, ZZ.
    """
    # Create Bell circuit
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    
    # Define observables for II, XX, YY, ZZ
    observables = SparsePauliOp.from_list([
        ("II", 1),
        ("XX", 1),
        ("YY", 1),
        ("ZZ", 1)
    ])
    
    # Initialize Estimator without backend parameter
    estimator = Estimator()
    
    # Run estimator
    job = estimator.run([(bell, observables)])
    pub_result = job.result()[0]
    
    # Return expectation values as numpy array
    return array(pub_result.data.evs)