from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def run_bell_state_simulator():
    """ Define a phi plus bell state using Qiskit, transpile the circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and return the counts dictionary.
    """
    # Create a phi plus bell state circuit (|00> + |11>)/√2
    circuit = QuantumCircuit(2)
    circuit.h(0)  # Apply Hadamard to first qubit
    circuit.cx(0, 1)  # Apply CNOT to create entanglement
    circuit.measure_all()  # Measure all qubits
    
    # Initialize AerSimulator as backend
    backend = AerSimulator()
    
    # Generate preset pass manager with optimization level 1
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend
    )
    
    # Transpile the circuit using the pass manager
    transpiled_circuit = pass_manager.run(circuit)
    
    # Initialize Sampler with AerSimulator as backend
    sampler = BackendSamplerV2(backend=backend)
    
    # Run the transpiled circuit using Sampler (default 1024 shots)
    job = sampler.run([transpiled_circuit])
    result = job.result()
    
    # Get counts from the measurement results
    counts = result[0].data.meas.get_counts()
    
    return counts