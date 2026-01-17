from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def run_bell_state_simulator():
    """ Define a phi plus bell state using Qiskit, transpile the circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and return the counts dictionary.
    """
    # Create a phi plus bell state circuit (|00⟩ + |11⟩)/√2
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # Create AerSimulator backend
    backend = AerSimulator()
    
    # Generate pass manager with optimization level 1
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    
    # Transpile the circuit
    isa_qc = pm.run(qc)
    
    # Create sampler with AerSimulator backend
    sampler = Sampler(backend=backend)
    
    # Run the transpiled circuit and get results
    result = sampler.run([isa_qc]).result()
    
    # Extract counts dictionary
    counts = result[0].data.meas.get_counts()
    
    return counts