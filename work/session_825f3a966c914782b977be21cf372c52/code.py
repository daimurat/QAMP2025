from numpy import float64
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator
from qiskit.quantum_info import SparsePauliOp

def estimator_qiskit() -> float64:
    """Run a Bell circuit on Qiskit Estimator and return expectation values for the bases II, XX, YY, ZZ."""
    # Create a Bell circuit on 2 qubits
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)

    # Define the four observables
    observables = SparsePauliOp.from_list([("II", 1), ("XX", 1), ("YY", 1), ("ZZ", 1)])

    # For local simulation, use AerSimulator and BackendEstimatorV2
    backend = AerSimulator()
    from qiskit.primitives import BackendEstimatorV2
    estimator = BackendEstimatorV2(backend=backend)

    # Run the estimator
    job = estimator.run([(bell, observables)])
    result = job.result()

    # Extract the four expectation values
    evs = result[0].data.evs

    # Return as float64 array
    return evs.astype(float64)