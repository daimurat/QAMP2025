import numpy as np
from qiskit.quantum_info import Statevector

def create_phi_plus_bell_state() -> Statevector:
    """
    Create a Statevector representing the phi+ Bell state (|00⟩ + |11⟩)/√2.
    
    Returns:
        Statevector: The normalized phi+ Bell state.
    """
    # Create statevector array with |00⟩ and |11⟩ components
    sv_data = np.zeros(4, dtype=complex)
    sv_data[0] = 1 / np.sqrt(2)  # |00⟩ component
    sv_data[3] = 1 / np.sqrt(2)  # |11⟩ component
    
    return Statevector(sv_data)

# Create and verify the phi+ Bell state
phi_plus = create_phi_plus_bell_state()
print("Phi+ Bell state:", phi_plus)
print("Amplitudes:", phi_plus.data)
print("Is valid:", phi_plus.is_valid())