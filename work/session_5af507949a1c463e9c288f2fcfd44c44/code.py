from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def bell_each_shot() -> list[str]:
    """ Run a phi plus Bell circuit using Qiskit Sampler with the Aer simulator as backend for 10 shots and return measurement results for each shots. To do so, transpile the circuit using a pass manager with optimization level as 1.
    """
    # Create Bell (phi+) circuit
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    
    # Initialize backend and transpile with optimization_level=1
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)
    
    # Initialize Sampler with AerSimulator as backend and run with 10 shots
    sampler = Sampler(backend=backend)
    result = sampler.run([isa_qc], shots=10).result()
    
    # Extract per-shot measurement bitstrings
    # The result contains classical register data - get bitstrings for each shot
    bitstrings = result[0].data.creg.get_bitstrings()
    
    # Return as list of 2-bit strings
    return [bitstring for bitstring in bitstrings]