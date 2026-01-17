from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler

def random_number_generator_unsigned_8bit(n: int)->[int]:
    """ Write a function that generates n number of random 8-bit unsigned integers using a Quantum Circuit and outputs a list of integers.
    """
    # Create a quantum circuit with 8 qubits and 8 classical bits
    qc = QuantumCircuit(8, 8)
    
    # Apply Hadamard gates to all qubits to create superposition
    qc.h(range(8))
    
    # Measure all qubits
    qc.measure(range(8), range(8))
    
    # Use AerSimulator as backend (local simulation mode)
    backend = AerSimulator()
    
    # Create sampler with backend (correct usage for local testing mode)
    sampler = Sampler(mode=backend)
    
    # Run the circuit n times with 1 shot each to get n random numbers
    random_numbers = []
    for _ in range(n):
        job = sampler.run([qc], shots=1)
        result = job.result()
        # Extract bit values from the result (using join_data as recommended)
        bitvals = result[0].join_data()
        # Convert bitstring to integer - need to get the measurement outcome
        counts = bitvals.get_counts()
        # Get the single measured bitstring
        bitstring = list(counts.keys())[0]
        # Convert binary string to integer
        number = int(bitstring, 2)
        random_numbers.append(number)
    
    return random_numbers