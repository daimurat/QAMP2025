from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider.backends import FakeCairoV2
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Batch, SamplerV2

def run_bell_states() -> tuple[dict[str, PrimitiveJob], str | None]:
    backend = FakeCairoV2()
    pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=123, backend=backend)
    circuits = {}

    # |Φ+⟩
    phi_plus = QuantumCircuit(2, 2)
    phi_plus.h(0)
    phi_plus.cx(0, 1)
    phi_plus.measure([0, 1], [0, 1])
    circuits['phi_plus'] = phi_plus

    # |Φ−⟩
    phi_minus = QuantumCircuit(2, 2)
    phi_minus.h(0)
    phi_minus.cx(0, 1)
    phi_minus.z(0)
    phi_minus.measure([0, 1], [0, 1])
    circuits['phi_minus'] = phi_minus

    # |Ψ+⟩
    psi_plus = QuantumCircuit(2, 2)
    psi_plus.h(0)
    psi_plus.cx(0, 1)
    psi_plus.x(0)
    psi_plus.measure([0, 1], [0, 1])
    circuits['psi_plus'] = psi_plus

    # |Ψ−⟩
    psi_minus = QuantumCircuit(2, 2)
    psi_minus.h(0)
    psi_minus.cx(0, 1)
    psi_minus.x(0)
    psi_minus.z(0)
    psi_minus.measure([0, 1], [0, 1])
    circuits['psi_minus'] = psi_minus

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)
        jobs = {}
        for name, qc in circuits.items():
            transpiled = pm.run(qc)
            jobs[name] = sampler.run([transpiled])

    return jobs, None