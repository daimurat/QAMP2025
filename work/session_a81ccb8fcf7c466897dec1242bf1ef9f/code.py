from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_ibm_runtime import Sampler
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def noisy_bell():
    """ Transpile a bell circuit using pass manager with optimization level as 1, run it using Qiskit Sampler with the Aer simulator as backend and get the execution counts.
    """
    # Create a Bell state circuit
    bell_circuit = QuantumCircuit(2)
    bell_circuit.h(0)
    bell_circuit.cx(0, 1)
    bell_circuit.measure_all()
    
    # Configure AerSimulator with noise model from FakeBelemV2
    backend = AerSimulator.from_backend(FakeBelemV2())
    
    # Generate pass manager with optimization level 1
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    
    # Transpile the circuit
    isa_qc = pm.run(bell_circuit)
    
    # Initialize Sampler with the AerSimulator backend
    sampler = Sampler(backend=backend)
    
    # Run the transpiled circuit
    result = sampler.run([isa_qc]).result()
    
    # Extract counts from result
    counts = result[0].join_data().get_counts()
    
    return counts