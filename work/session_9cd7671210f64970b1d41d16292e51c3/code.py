from qiskit.circuit import QuantumCircuit, Parameter

def circuit():
    """ Return an ansatz to create a quantum dataset of pure states distributed equally across the bloch sphere. Use minimum number of gates in the ansatz.
    """
    # Create a single qubit circuit
    qc = QuantumCircuit(1)
    
    # Create parameters for the two angles needed for Bloch sphere coverage
    theta = Parameter('theta')
    phi = Parameter('phi')
    
    # Apply Ry(theta) to set the polar angle (latitude on Bloch sphere)
    qc.ry(theta, 0)
    
    # Apply Rz(phi) to set the azimuthal angle (longitude on Bloch sphere)
    qc.rz(phi, 0)
    
    return qc