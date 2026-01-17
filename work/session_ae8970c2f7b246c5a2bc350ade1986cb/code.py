from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def run_bell_state_simulator():
    """ Define a phi plus bell state using Qiskit, transpile the circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and return the counts dictionary.
    """
    # Create phi plus bell state circuit (|00⟩ + |11⟩)/√2
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_circuit.measure_all()
    
    # Initialize AerSimulator as backend
    backend = AerSimulator()
    
    # Generate pass manager with optimization level 1
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend
    )
    
    # Transpile the circuit
    transpiled_circuit = pass_manager.run(bell_circuit)
    
    # Use Sampler with AerSimulator backend
    sampler = Sampler(backend=backend)
    
    # Run the circuit and get results
    job = sampler.run([transpiled_circuit], shots=1024)
    result = job.result()
    
    # Return the counts dictionary
    counts = result[0].data.get_counts()
    return counts