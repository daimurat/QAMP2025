from qiskit import QuantumCircuit

def create_ghz(drawing=False):
    """ Generate a QuantumCircuit for a 3 qubit GHZ State and measure it. If `drawing` is True, return both the circuit object and the Matplotlib drawing of the circuit, otherwise return just the circuit object.
    """
    circuit = QuantumCircuit(3, 3)
    
    # Create GHZ state
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    
    # Measure all qubits
    circuit.measure_all()
    
    if drawing:
        fig = circuit.draw(output='mpl')
        return circuit, fig
    
    return circuit