from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import Sampler
from qiskit_aer import AerSimulator

def run_bell_state_simulator():
    # Create a Bell state (phi plus) circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()  # This labels the classical register as 'meas'

    # Transpile with optimization_level=1
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)

    # Run Sampler with AerSimulator backend
    sampler = Sampler(backend=backend)
    job = sampler.run([isa_qc])
    result = job.result()

    # Access counts via result[0].data.meas.get_counts()
    counts = result[0].data.meas.get_counts()
    return counts