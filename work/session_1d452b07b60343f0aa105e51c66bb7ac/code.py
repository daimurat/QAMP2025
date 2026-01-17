from qiskit_ibm_runtime.fake_provider import FakeCairoV2
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def transpile_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """ For the given Quantum Circuit, return the transpiled circuit for the Fake Cairo V2 backend using pass manager with optimization level as 1.
    """
    # Create a FakeCairoV2 backend instance
    backend = FakeCairoV2()
    
    # Generate a preset pass manager with optimization level 1 and the backend
    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    
    # Apply the pass manager to the input circuit
    transpiled_circuit = pass_manager.run(circuit)
    
    # Return the transpiled circuit
    return transpiled_circuit