from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def send_bits(bitstring: str)->QuantumCircuit:
    """ Provide a quantum circuit that enables the transmission of two classical bits from the sender to the receiver through a single qubit of quantum communication, given that the sender and receiver have access to entangled qubits.
    """
    # Validate input
    if bitstring not in ["00", "01", "10", "11"]:
        raise ValueError("bitstring must be one of '00', '01', '10', '11'")
    
    # Create quantum circuit with 2 qubits and 2 classical bits
    qreg_q = QuantumRegister(2, 'q')
    creg_c = ClassicalRegister(2, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)
    
    # Create entangled Bell pair between qubit 0 (sender) and qubit 1 (receiver)
    circuit.h(qreg_q[0])  # Hadamard on sender qubit
    circuit.cx(qreg_q[0], qreg_q[1])  # CNOT to create entanglement
    
    # Encode the 2-bit string onto the sender qubit
    if bitstring == "00":
        pass  # No additional gates needed
    elif bitstring == "01":
        circuit.z(qreg_q[0])  # Z gate for 01
    elif bitstring == "10":
        circuit.x(qreg_q[0])  # X gate for 10
    elif bitstring == "11":
        circuit.z(qreg_q[0])  # Z then X for 11
        circuit.x(qreg_q[0])
    
    # Measure both qubits to recover the transmitted bits
    circuit.measure(qreg_q[0], creg_c[0])
    circuit.measure(qreg_q[1], creg_c[1])
    
    return circuit