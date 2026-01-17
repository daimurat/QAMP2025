from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_ibm_runtime import Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def noisy_bell():
    """ Transpile a bell circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and get the execution counts.
    """
    # Create a 2-qubit Bell state circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # Create AerSimulator backend instance
    backend = AerSimulator()
    
    # Use generate_preset_pass_manager with optimization_level=1
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)
    
    # Use Sampler from qiskit_ibm_runtime without backend in constructor
    sampler = Sampler()
    
    # Pass backend to sampler.run method and get execution counts
    result = sampler.run([isa_qc], backend=backend).result()
    counts = result[0].join_data().get_counts()
    
    return counts