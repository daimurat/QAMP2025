from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBelemV2
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendSamplerV2


def noisy_bell() -> dict:
    """Transpile a Bell circuit using a preset pass manager (optimization level 1),
    run it with the Qiskit Sampler primitive on an AerSimulator that includes the
    noise model from ``FakeBelemV2``, and return the execution counts.

    Returns:
        dict: A dictionary of measurement counts, e.g. ``{'00': 512, '11': 512}``.
    """
    # ------------------------------------------------------------------
    # 1. Build the Bell state circuit (with measurement)
    # ------------------------------------------------------------------
    bell = QuantumCircuit(2, 2)
    bell.h(0)
    bell.cx(0, 1)
    bell.measure([0, 1], [0, 1])

    # ------------------------------------------------------------------
    # 2. Create a fake backend to obtain its noise model and target
    # ------------------------------------------------------------------
    fake_backend = FakeBelemV2()
    noise_model = fake_backend.noise_model

    # ------------------------------------------------------------------
    # 3. Generate a preset pass manager (optimization level 1) using the
    #    fake backend's target (so the pass manager knows the device
    #    constraints) and transpile the circuit.
    # ------------------------------------------------------------------
    pm = generate_preset_pass_manager(optimization_level=1, backend=fake_backend)
    transpiled_circ = pm.run(bell)

    # ------------------------------------------------------------------
    # 4. Configure an AerSimulator with the fake backend's noise model
    # ------------------------------------------------------------------
    aer_backend = AerSimulator(noise_model=noise_model)

    # ------------------------------------------------------------------
    # 5. Initialise the Sampler primitive (BackendSamplerV2) with the
    #    noisy AerSimulator.
    # ------------------------------------------------------------------
    sampler = BackendSamplerV2(backend=aer_backend)

    # ------------------------------------------------------------------
    # 6. Run the sampler and retrieve counts
    # ------------------------------------------------------------------
    job = sampler.run([transpiled_circ], shots=1024)
    result = job.result()
    # ``result[0].data`` is a BitArray; ``get_counts()`` returns a dict.
    counts = result[0].data.get_counts()

    return counts