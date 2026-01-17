from numpy import float64
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp

def estimator_qiskit() -> float64:
    """ Run a Bell circuit on Qiskit Estimator and return expectation values for the bases II, XX, YY, ZZ.
    """
    # Create a Bell state circuit with 2 qubits
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    
    # Prepare the observables
    ii_obs = SparsePauliOp.from_list([("II", 1)])
    xx_obs = SparsePauliOp.from_list([("XX", 1)])
    yy_obs = SparsePauliOp.from_list([("YY", 1)])
    zz_obs = SparsePauliOp.from_list([("ZZ", 1)])
    
    # Use StatevectorEstimator for local simulation (no backend/cloud needed)
    from qiskit.primitives import StatevectorEstimator as Estimator
    
    estimator = Estimator()
    
    # Compute expectation values for each observable
    ii_job = estimator.run([(bell_circuit, ii_obs, [])])
    xx_job = estimator.run([(bell_circuit, xx_obs, [])])
    yy_job = estimator.run([(bell_circuit, yy_obs, [])])
    zz_job = estimator.run([(bell_circuit, zz_obs, [])])
    
    ii_result = ii_job.result()[0].data.evs[0]
    xx_result = xx_job.result()[0].data.evs[0]
    yy_result = yy_job.result()[0].data.evs[0]
    zz_result = zz_job.result()[0].data.evs[0]
    
    return float64(zz_result)