from qiskit import QuantumCircuit, Aer, execute
import numpy as np

def run_zeno_elitzur_vaidman():
    """
    Runs the Elitzur–Vaidman bomb tester with Zeno suppression.
    Returns:
        live_success_pct: float  # success % when the bomb is live
        dud_success_pct: float # success % when the bomb is a dud
        detonation_pct: float   # fraction of shots discarded due to absorption=1
    """
    backend = Aer.get_backend('qasm_simulator')
    shots = 10000
    cycles = 25
    theta = np.pi / 2 / cycles

    # ---------- 1. LIVE BOMB ----------
    circuits_live = []
    for _ in range(shots):
        qc = QuantumCircuit(3, 2)  # qubits: photon, bomb, absorption
        qc.x(0)  # start with the photon in |1>
        detonated = False
        for _ in range(cycles):
            qc.ry(theta, 0)
            qc.cx(0, 1)           # bomb-photon interaction
            qc.cx(0, 2)           # absorption qubit
            qc.measure(2, 0)     # measure absorption
            qc.reset(2)          # reset for next cycle
            # if measurement outcome is 1, mark detonated
            # (we'll track this classically via post-processing)
        qc.measure(0, 1)           # final photon measurement
        circuits_live.append(qc)

    job_live = execute(circuits_live, backend, shots=1, memory=True)
    mem_live = job_live.result().get_memory()
    # mem_live[i] is a 2-bit string: [absorption_outcomes][final_photon]
    # absorption_outcomes: we treat any '1' in the first bit as detonation
    detonated_live = 0
    success_live = 0
    for r in mem_live:
        abs_bits = r[0]
        final_photon = int(r[1])
        if '1' in abs_bits:
            detonated_live += 1
        else:
            if final_photon == 0:
                success_live += 1

    # ---------- 2. DUD BOMB ----------
    circuits_dud = []
    for _ in range(shots):
        qc = QuantumCircuit(3, 2)
        qc.x(0)
        for _ in range(cycles):
            qc.ry(theta, 0)
            # no CX on bomb qubit, but still CX on absorption
            qc.cx(0, 2)
            qc.measure(2, 0)
            qc.reset(2)
        qc.measure(0, 1)
        circuits_dud.append(qc)

    job_dud = execute(circuits_dud, backend, shots=1, memory=True)
    mem_dud = job_dud.result().get_memory()
    detonated_dud = 0
    success_dud = 0
    for r in mem_dud:
        abs_bits = r[0]
        final_photon = int(r[1])
        if '1' in abs_bits:
            detonated_dud += 1
        else:
            if final_photon == 0:
                success_dud += 1

    # ---------- 3. COMPUTE PERCENTAGES ----------
    valid_live = shots - detonated_live
    valid_dud  = shots - detonated_dud

    live_success_pct = (success_live / valid_live * 100) if valid_live else 0.0
    dud_success_pct  = (success_dud  / valid_dud  * 100) if valid_dud  else 0.0
    detonation_pct   = ((detonated_live + detonated_dud) / (2 * shots) * 100)

    return live_success_pct, dud_success_pct, detonation_pct

if __name__ == "__main__":
    lv, dv, det = run_zeno_elitzur_vaidman()
    print(f"Live-bomb success: {lv:.2f}%")
    print(f"Dud-bomb success:  {dv:.2f}%")
    print(f"Detonation rate:   {det:.2f}%")