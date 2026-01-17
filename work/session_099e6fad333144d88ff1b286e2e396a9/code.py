import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeKyoto
from qiskit.circuit.library import GraphState
from qiskit.circuit import QuantumCircuit


def get_graph_state() -> QuantumCircuit:
    """
    Return the circuit for the graph state of the coupling map of the FakeKyoto backend.

    The function:
    1. Instantiates the FakeKyoto backend.
    2. Retrieves its directed coupling map and converts it to a symmetric
       adjacency matrix (undirected graph) using only NumPy.
    3. Builds a GraphState circuit from that adjacency matrix.

    Returns:
        QuantumCircuit: A circuit that prepares the graph state corresponding
        to the backend's connectivity.
    """
    # 1. Load the fake backend
    backend = FakeKyoto()

    # 2. Get the directed edges from the coupling map
    coupling_map = backend.coupling_map          # CouplingMap object
    edges = coupling_map.get_edges()             # list of (src, dst) tuples

    # 3. Build a symmetric adjacency matrix
    n_qubits = backend.num_qubits
    adj = np.zeros((n_qubits, n_qubits), dtype=int)

    for src, dst in edges:
        adj[src, dst] = 1
        adj[dst, src] = 1   # make the matrix undirected (symmetric)

    # 4. Create the GraphState circuit
    # GraphState accepts either a list of lists or a NumPy array.
    circuit = GraphState(adj)

    return circuit