from qiskit.quantum_info import Statevector
from math import sqrt

def create_bell_statevector() -> Statevector:
    """ Return a phi+ Bell statevector.
    """
    # The phi+ Bell state is (|00> + |11>) / sqrt(2)
    state = Statevector([
        1/sqrt(2),  # amplitude for |00>
        0,          # amplitude for |01>
        0,          # amplitude for |10>
        1/sqrt(2)   # amplitude for |11>
    ])
    return state