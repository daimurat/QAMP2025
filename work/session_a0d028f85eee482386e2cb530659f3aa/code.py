from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_ibm_runtime import Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import BackendSamplerV2

def noisy_bell():
    """ Transpile a bell circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and get the execution counts.
    """
    # Create a Bell state circuit with 2 qubits
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure_all()
    
    # Create AerSimulator backend instance
    backend = AerSimulator()
    
    # Use generate_preset_pass_manager with optimization_level=1
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    
    # Transpile the circuit using the pass manager
    isa_qc = pm.run(bell)
    
    # Initialize Sampler with the AerSimulator backend using BackendSamplerV2
    sampler = BackendSamplerV2(backend=backend)
    
    # Run the transpiled circuit using Sampler
    job = sampler.run([isa_qc])
    result = job.result()
    
    # Return the execution counts from the result
    counts = result[0].join_data().get_counts()
    return counts