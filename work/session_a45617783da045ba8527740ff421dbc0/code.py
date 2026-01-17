from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

def ai_transpiling(num_qubits):
    """ Generate an EfficientSU2 circuit with the given number of qubits, 1 reps and make entanglement circular. 
    Then use the Qiskit Transpiler service with the AI flag turned on, use the ibm_brisbane backend and an optimization level of 3 and transpile the generated circuit.
    """
    # Create EfficientSU2 circuit with specified parameters
    circuit = EfficientSU2(num_qubits=num_qubits, reps=1, entanglement='circular')
    
    # Get the backend
    service = QiskitRuntimeService()
    backend = service.backend('ibm_brisbane')
    
    # Generate preset pass manager with optimization level 3
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    
    # Transpile the circuit
    transpiled_circuit = pm.run(circuit)
    
    return transpiled_circuit