from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def run_bell_state_simulator() -> dict:
    """
    Create a |Φ⁺⟩ Bell state circuit, transpile it with optimization level 1,
    run it on the AerSimulator using the qiskit_ibm_runtime Sampler primitive,
    and return the resulting counts dictionary.

    Returns:
        dict: Mapping of bit‑string outcomes (e.g., "00", "11") to integer counts.
    """
    # 1. Build the Bell state circuit and add measurements
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    # 2. Backend (Aer simulator)
    backend = AerSimulator()

    # 3. Generate a preset pass manager with optimization level 1 and transpile
    pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_qc = pass_manager.run(qc)

    # 4. Instantiate the Sampler primitive with the AerSimulator backend
    sampler = Sampler(backend=backend)

    # 5. Run the sampler (use 1024 shots)
    shots = 1024
    job = sampler.run([transpiled_qc], shots=shots)
    result = job.result()

    # 6. Convert the quasi‑distribution to integer counts
    # SamplerResult stores a list of quasi‑distributions (one per circuit)
    quasi_dist = result.quasi_dists[0]  # dict of {bitstring: probability}
    counts = {bitstr: int(round(prob * shots)) for bitstr, prob in quasi_dist.items()}

    # 7. Adjust for any rounding discrepancy so that total counts == shots
    total_counts = sum(counts.values())
    diff = shots - total_counts
    if diff != 0:
        # Find the outcome with the largest probability to absorb the difference
        max_key = max(quasi_dist, key=quasi_dist.get)
        counts[max_key] += diff

    return counts