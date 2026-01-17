from qiskit import QuantumCircuit

def bv_function(s: str) -> QuantumCircuit:
    """ Write a function to design a Bernstein-Vazirani oracle from a bitstring and return it.
    """
    n = len(s)
    # Create a circuit with n+1 qubits (n input qubits + 1 ancilla)
    circuit = QuantumCircuit(n + 1, name="BV_oracle")
    
    # Apply X gate to the last qubit (ancilla) to prepare |1>
    circuit.x(n)
    
    # Apply Hadamard gates to all qubits
    circuit.h(range(n + 1))
    
    # Implement the oracle: for each bit in the bitstring, apply CX if bit is '1'
    for i in range(n):
        if s[i] == '1':
            # Apply CX with control on qubit i and target on ancilla (qubit n)
            circuit.cx(i, n)
    
    # Apply Hadamard gates to all qubits again
    circuit.h(range(n + 1))
    
    return circuit