from qiskit.circuit import QuantumCircuit, Parameter

def circuit() -> QuantumCircuit:
    """ Create a parameterized quantum circuit using minimum resources whose statevector output cover the equatorial plane of the surface of the bloch sphere.
    """
    qc = QuantumCircuit(1)
    theta = Parameter('theta')
    qc.ry(theta, 0)
    return qc