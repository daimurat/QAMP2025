from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeCairoV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Estimator, EstimatorOptions

def run_circuit_with_dd_trex():
    circuit = EfficientSU2(num_qubits=5, reps=2, entanglement='pairwise')
    observable = SparsePauliOp.from_list([("Z", -1)])
    backend = FakeCairoV2()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=789)
    transpiled_circuit = pm.run(circuit)
    
    options = EstimatorOptions()
    options.dynamical_decoupling.enable = True
    options.twirling.enable_trex = True
    
    estimator = Estimator(backend=backend, options=options)
    job = estimator.run(transpiled_circuit, observable)
    return job