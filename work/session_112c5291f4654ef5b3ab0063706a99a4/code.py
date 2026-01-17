from qiskit import QuantumCircuit

def create_ghz(drawing=False):
    """ Generate a QuantumCircuit for a 3 qubit GHZ State and measure it. If `drawing` is True, return both the circuit object and the Matplotlib drawing of the circuit, otherwise return just the circuit object.
    """
    # Create a QuantumCircuit with 3 qubits and 3 classical bits for measurement
    circuit = QuantumCircuit(3, 3)
    
    # Apply Hadamard gate to the first qubit
    circuit.h(0)
    
    # Apply CX gates to create entanglement (GHZ state)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    
    # Measure all qubits
    circuit.measure([0, 1, 2], [0, 1, 2])
    
    # Return circuit and matplotlib drawing if requested, otherwise just the circuit
    if drawing:
        fig = circuit.draw(output='mpl')
        return circuit, fig
    else:
        return circuit