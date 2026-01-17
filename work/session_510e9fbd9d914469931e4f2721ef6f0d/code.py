from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def random_coin_flip(samples: int) -> dict:
    """
    Flips a quantum coin (a single qubit in superposition) a specified number of times,
    returning aggregated counts labelled as 'Heads' (qubit measurement 0) and 'Tails' (1).

    Parameters
    ----------
    samples : int
        Number of times to flip the coin (i.e., shots for the sampler).

    Returns
    -------
    dict
        Dictionary with keys 'Heads' and 'Tails' whose values are the counts
        of measuring 0 and 1 respectively.
    """
    # 1. Build the quantum circuit: H on qubit 0, then measure to creg 'm'
    qc = QuantumCircuit(1, 1, name="coin_flip")  # 1 qubit, 1 classical bit
    qc.h(0)                                      # put qubit in superposition
    qc.measure(0, 0)                             # measure qubit 0 into cbit 0

    # 2. Prepare the backend and transpile for it
    backend = AerSimulator()
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)

    # 3. Instantiate the sampler (V2) bound to AerSimulator backend
    sampler = Sampler(backend=backend)

    # 4. Run sampler with requested number of shots
    job = sampler.run([(isa_qc, None, samples)])  # pub tuple: (circuit, params, shots)
    result = job.result()
    bit_array = result[0].data.get("0")           # classical register named '0' by default

    # 5. Count bit values across all shots
    heads = 0
    tails = 0
    for shot_bits in bit_array:  # iterate over shots
        val = int(shot_bits)       # 0 or 1 for single-qubit measurement
        if val == 0:
            heads += 1
        else:
            tails += 1

    return {"Heads": heads, "Tails": tails}