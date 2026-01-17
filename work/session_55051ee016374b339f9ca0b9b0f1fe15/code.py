from qiskit.quantum_info import Statevector
from math import sqrt

def create_bell_statevector() -> Statevector:
    """ Return a phi+ Bell statevector.
    """
    # Create the phi+ Bell state (|00> + |11>)/sqrt(2)
    bell_data = [1/sqrt(2), 0, 0, 1/sqrt(2)]
    return Statevector(bell_data)