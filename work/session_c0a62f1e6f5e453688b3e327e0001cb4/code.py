from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2
from qiskit_ibm_runtime import IBMBackend


def two_qubit_conections(
    backend
) -> list:
    """Return the two qubit connections for any input backend.
    
    The backend parameter can be either FakeBackendV2 or IBMBackend.
    """
    return list(backend.coupling_map.get_edges())