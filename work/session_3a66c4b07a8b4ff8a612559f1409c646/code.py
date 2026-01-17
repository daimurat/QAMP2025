# Required imports
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, ClassicalRegister, Clbit, CircuitError

def gate_if_clbits(
    circuit: QuantumCircuit, gate: Gate, qubits: list[int], condition_clbits: list[int]
) -> None:
    """Apply ``gate`` to ``qubits`` conditioned on all ``condition_clbits`` being ``1``.

    The function works with a ``QuantumCircuit`` that already contains the desired
    quantum and classical registers. ``condition_clbits`` are interpreted as indices
    into ``circuit.clbits`` (the global list of classical bits of the circuit).

    If ``condition_clbits`` is empty the gate is added unconditionally.
    All bits in ``condition_clbits`` must belong to the *same* classical register;
    otherwise a ``ValueError`` is raised.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to which the (conditional) gate will be added.
    gate : Gate
        The gate (or instruction) to be applied.
    qubits : list[int]
        Indices of the qubits (into ``circuit.qubits``) the gate acts on.
    condition_clbits : list[int]
        Indices of the classical bits (into ``circuit.clbits``) that must all be ``1``.

    Raises
    ------
    ValueError
        If the number of target qubits does not match ``gate.num_qubits`` or if the
        condition bits span more than one classical register.
    """
    # --------------------------------------------------------------------- #
    # 1. Validate the number of target qubits matches the gate's arity.
    # --------------------------------------------------------------------- #
    if len(qubits) != gate.num_qubits:
        raise ValueError(
            f"The gate expects {gate.num_qubits} qubits, but {len(qubits)} were provided."
        )

    # --------------------------------------------------------------------- #
    # 2. If no condition is requested, simply append the gate.
    # --------------------------------------------------------------------- #
    if not condition_clbits:
        circuit.append(gate, [circuit.qubits[i] for i in qubits])
        return

    # --------------------------------------------------------------------- #
    # 3. Resolve the classical bits from the supplied indices.
    # --------------------------------------------------------------------- #
    try:
        clbits = [circuit.clbits[i] for i in condition_clbits]
    except IndexError as exc:
        raise ValueError("One or more condition_clbits indices are out of range.") from exc

    # --------------------------------------------------------------------- #
    # 4. Ensure all condition bits belong to the same ClassicalRegister.
    # --------------------------------------------------------------------- #
    registers = {bit.register for bit in clbits}
    if len(registers) != 1:
        raise ValueError(
            "All condition bits must belong to the same ClassicalRegister. "
            f"Found registers: {[reg.name for reg in registers]}"
        )
    register: ClassicalRegister = registers.pop()

    # --------------------------------------------------------------------- #
    # 5. Compute the integer value that corresponds to all selected bits = 1.
    #    In Qiskit the least‑significant bit is index 0 of the register.
    # --------------------------------------------------------------------- #
    condition_value = 0
    for bit in clbits:
        condition_value |= 1 << bit.index  # set the bit at its register index

    # --------------------------------------------------------------------- #
    # 6. Create a copy of the gate, attach the condition, and append it.
    # --------------------------------------------------------------------- #
    gated_copy = gate.copy()
    # ``c_if`` is deprecated but still functional; it sets a classical equality
    # condition on the instruction.
    gated_copy.c_if(register, condition_value)

    circuit.append(gated_copy, [circuit.qubits[i] for i in qubits])

    # The function intentionally returns ``None``.