from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def random_coin_flip(samples: int) -> dict:
    """
    Simulate quantum coin flips using a single-qubit Hadamard gate.
    
    Args:
        samples (int): Number of shots/samples to execute
        
    Returns:
        dict: Dictionary with 'Heads' and 'Tails' counts summing to samples
    """
    # Create single-qubit circuit with Hadamard and measurement
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Hadamard gate creates superposition
    qc.measure(0, 0)  # Measure qubit 0 into classical bit 0
    
    # Set up AerSimulator backend
    aer_sim = AerSimulator()
    
    # Transpile circuit for the backend
    pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
    isa_qc = pm.run(qc)
    
    # Create sampler with AerSimulator backend
    sampler = Sampler(backend=aer_sim)
    
    # Execute the sampler
    job = sampler.run([isa_qc], shots=samples)
    result = job.result()
    
    # Extract counts from result
    bitvals = result[0].data.c0  # Classical register name is auto-generated as c0
    counts = bitvals.get_counts()
    
    # Convert counts to Heads/Tails mapping
    heads = counts.get(0, 0)  # 0 maps to Heads
    tails = counts.get(1, 0)  # 1 maps to Tails
    
    return {'Heads': heads, 'Tails': tails}