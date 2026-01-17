from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import circuit_to_dag
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZGate, HGate, XGate

class HXHPass(TransformationPass):
    def run(self, dag):
        for node in dag.op_nodes():
            if isinstance(node.op, ZGate) and node.op.num_ctrl_qubits == 0:
                # Build replacement circuit: H-X-H
                replacement = QuantumCircuit(1)
                replacement.h(0)
                replacement.x(0)
                replacement.h(0)
                replacement_dag = circuit_to_dag(replacement)
                dag.substitute_node_with_dag(node, replacement_dag)
        return dag

def create_hxh_pass():
    return HXHPass()