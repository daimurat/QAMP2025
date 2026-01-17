from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_ibm_runtime import Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def noisy_bell():
    """ Transpile a bell circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and get the execution counts.
    """
    # Create a Bell state circuit with 2 qubits
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # Use generate_preset_pass_manager with optimization_level=1
    pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1)
    
    # Transpile the circuit using the pass manager
    isa_qc = pm.run(qc)
    
    # Initialize Sampler with AerSimulator as backend
    sampler = Sampler(backend=AerSimulator())
    
    # Run the sampler and extract execution counts
    result = sampler.run([isa_qc]).result()
    counts = result[0].join_data().get_counts()
    
    # Return the counts result
    return counts