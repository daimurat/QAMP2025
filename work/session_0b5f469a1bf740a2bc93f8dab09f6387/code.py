from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import Batch, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeCairoV2

def submit_bell_state_jobs():
    # Create Bell states
    phi_plus = QuantumCircuit(2)
    phi_plus.h(0)
    phi_plus.cx(0, 1)
    
    phi_minus = QuantumCircuit(2)
    phi_minus.h(0)
    phi_minus.cx(0, 1)
    phi_minus.z(0)
    phi_minus.z(1)
    
    psi_plus = QuantumCircuit(2)
    psi_plus.h(0)
    psi_plus.cx(0, 1)
    psi_plus.x(1)
    
    psi_minus = QuantumCircuit(2)
    psi_minus.h(0)
    psi_minus.cx(0, 1)
    psi_minus.x(1)
    psi_minus.z(0)
    psi_minus.z(1)
    
    # Instantiate backend
    backend = FakeCairoV2()
    
    # Create pass manager with optimization level 3 and seed 123
    pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=123, backend=backend)
    
    # Transpile all circuits
    phi_plus_t = pm.run(phi_plus)
    phi_minus_t = pm.run(phi_minus)
    psi_plus_t = pm.run(psi_plus)
    psi_minus_t = pm.run(psi_minus)
    
    # Use Batch context and submit jobs
    with Batch(backend=backend) as batch:
        sampler = SamplerV2(backend=backend)
        job_phi_plus = sampler.run([phi_plus_t])
        job_phi_minus = sampler.run([phi_minus_t])
        job_psi_plus = sampler.run([psi_plus_t])
        job_psi_minus = sampler.run([psi_minus_t])
        
        jobs = {
            'phi_plus': job_phi_plus,
            'phi_minus': job_phi_minus,
            'psi_plus': job_psi_plus,
            'psi_minus': job_psi_minus
        }
        
        batch_id = batch.session_id
    
    return jobs, batch_id