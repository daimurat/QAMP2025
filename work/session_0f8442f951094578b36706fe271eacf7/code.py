from numpy import float64
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator
from qiskit.quantum_info import SparsePauliOp

def estimator_qiskit() -> float64:
    """ Run a Bell circuit on Qiskit Estimator and return expectation values for the bases II, XX, YY, ZZ. """
    # Create Bell state circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    # Observables: II, XX, YY, ZZ
    observables = [
        SparsePauliOp.from_list([("II", 1)]),
        SparsePauliOp.from_list([("XX", 1)]),
        SparsePauliOp.from_list([("YY", 1)]),
        SparsePauliOp.from_list([("ZZ", 1)])
    ]
    
    # Use Estimator (StatevectorEstimator) to get expectation values
    estimator = Estimator()
    job = estimator.run([(qc, obs) for obs in observables])
    result = job.result()
    
    # Extract values and compute a single float64 aggregate
    values = result.values
    # Return the sum of the four expectation values as a single float64
    return float64(values.sum())