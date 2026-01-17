from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit.primitives import Sampler
from qiskit.providers.aer import AerSimulator

def sift_key(sender_basis, receiver_basis, qc=None):
    """
    Generate a sifted key by measuring all qubits in sender_basis, then keeping
    only the bits where sender_basis[i] == receiver_basis[i].
    
    Parameters
    ----------
    sender_basis : list[int]
        0 (Z-basis) or 1 (X-basis) for each qubit.
    receiver_basis : list[int]
        0 (Z-basis) or 1 (X-basis) for each qubit.
    qc : QuantumCircuit, optional
        Pre-built quantum circuit with n qubits. If None, a simple uniform
        superposition (H on all qubits) is used for demonstration.
    
    Returns
    -------
    str
        Binary string of the sifted key.
    """
    n = len(sender_basis)
    if len(receiver_basis) != n:
        raise ValueError("sender_basis and receiver_basis must have the same length.")
    
    # If no circuit provided, create a simple test circuit
    if qc is None:
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.h(i)
    
    # Add measurements in the sender's chosen basis
    creg = ClassicalRegister(n, name='c')
    qc.add_register(creg)
    
    for i in range(n):
        if sender_basis[i] == 1:          # X-basis measurement
            qc.h(i)
        # Z-basis measurement needs no rotation
        qc.measure(i, creg[i])
    
    # Run on AerSimulator via Sampler
    sampler = Sampler(options={'backend': AerSimulator()})
    job = sampler.run(qc, shots=1024)
    counts = job.result().quasi_dists[0].binary_probabilities()
    
    # Convert to counts dict (int keys)
    counts = {int(k, 2): int(v * 1024) for k, v in counts.items()}
    most_prob = max(counts, key=counts.get)
    bitstring = format(most_prob, f'0{n}b')
    
    # Sift: keep bits where bases match
    sifted_bits = ''.join(bitstring[i]
                          for i in range(n)
                          if sender_basis[i] == receiver_basis[i])
    return sifted_bits


# Example usage
if __name__ == "__main__":
    sender   = [0, 1, 0, 1, 0]
    receiver = [0, 1, 1, 1, 0]
    print("Sifted key:", sift_key(sender, receiver))