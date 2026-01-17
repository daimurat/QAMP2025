from qiskit import transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, EstimatorOptions
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

def run_circuit_with_dd_trex() -> 'PrimitiveJob':
    """Run EfficientSU2 on FakeCairoV2 with dynamic decoupling and T-REx enabled."""
    
    # Setup
    backend = FakeCairoV2()
    
    # Circuit setup
    circuit = EfficientSU2(num_qubits=5, reps=2, entanglement='pairwise')
    observable = SparsePauliOp(['ZIII'], coeffs=[-1])
    
    # Transpile
    transpiled_circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=1,
        seed_transpiler=789
    )
    
    # Configure options
    options = EstimatorOptions(
        optimization_level=1,
        resilience_level=1,
        dynamical_decoupling=True,
        trex=True
    )
    
    # Run
    estimator = Estimator(backend=backend, options=options)
    job = estimator.run(circuits=[transpiled_circuit], observables=[observable])
    
    return job


# Demonstration
if __name__ == "__main__":
    job = run_circuit_with_dd_trex()
    print("Job submitted:", job.job_id())