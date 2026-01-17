from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def conditional_two_qubit_circuit():
    """ Create a quantum circuit with one qubit and two classical bits. The qubit's operation depends on its measurement outcome: if it measures to 1 (|1> state), it flips the qubit's state back to |0> using an X gate. The qubit's initial state is randomized using a Hadamard gate. When building the quantum circuit make sure the classical registers is named 'c'.
    """
    # Create quantum register with 1 qubit
    qreg = QuantumRegister(1)
    # Create classical register with 2 bits named 'c'
    creg = ClassicalRegister(2, 'c')
    
    # Create quantum circuit
    qc = QuantumCircuit(qreg, creg)
    
    # Apply Hadamard gate to randomize qubit state
    qc.h(qreg[0])
    
    # Measure the qubit and store result in classical register
    qc.measure(qreg[0], creg[0])
    
    # Conditionally apply X gate if measurement outcome is 1
    with qc.if_test((creg[0], 1)):
        qc.x(qreg[0])
    
    return qc