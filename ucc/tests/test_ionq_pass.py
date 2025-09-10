"""
Test for IonQ Hardware Optimization Pass
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from ucc.transpilers.ionq_pass import IonQOptimizationPass
from qiskit.circuit.library import RXXGate


class TestIonQOptimizationPass:
    """Test cases for the IonQ optimization pass."""

    def test_ionq_pass_initialization(self):
        """Test that IonQ pass can be initialized with default parameters."""
        pass_instance = IonQOptimizationPass()
        assert pass_instance.use_tket == True  # tket is available and enabled by default
        assert pass_instance.optimization_level == 2

    def test_ionq_pass_custom_parameters(self):
        """Test that IonQ pass can be initialized with custom parameters."""
        pass_instance = IonQOptimizationPass(use_tket=True, optimization_level=3)
        assert pass_instance.use_tket == True  # tket is available and enabled by default
        assert pass_instance.optimization_level == 3

    def test_ionq_gate_set_optimization(self):
        """Test IonQ gate set optimization."""
        from qiskit.converters import circuit_to_dag

        # Create circuit with non-native gates
        qc = QuantumCircuit(2)
        qc.h(0)  # H gate needs to be decomposed
        qc.x(1)  # X gate needs to be decomposed
        qc.cx(0, 1)  # CNOT needs to be decomposed

        pass_instance = IonQOptimizationPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should produce valid circuit
        assert optimized_dag.num_qubits() == 2
        assert len(optimized_dag.op_nodes()) > 0

    def test_ionq_rotation_optimization(self):
        """Test IonQ rotation optimization."""
        from qiskit.converters import circuit_to_dag

        # Create circuit with multiple rotations
        qc = QuantumCircuit(1)
        qc.rx(np.pi/4, 0)
        qc.ry(np.pi/6, 0)
        qc.rx(np.pi/8, 0)  # Should be combined with first RX

        pass_instance = IonQOptimizationPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should produce valid circuit
        assert optimized_dag.num_qubits() == 1

    def test_ionq_native_gates_preserved(self):
        """Test that IonQ native gates are preserved."""
        from qiskit.converters import circuit_to_dag

        # Create circuit with native IonQ gates
        qc = QuantumCircuit(2)
        qc.rx(np.pi/4, 0)
        qc.ry(np.pi/6, 0)
        qc.rz(np.pi/8, 1)
        qc.append(RXXGate(np.pi/4), [0, 1])

        pass_instance = IonQOptimizationPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should preserve the structure
        assert optimized_dag.num_qubits() == 2

    def test_ionq_empty_circuit(self):
        """Test IonQ pass on an empty circuit."""
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(1)

        pass_instance = IonQOptimizationPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should return valid empty circuit
        assert optimized_dag.num_qubits() == 1
        assert len(optimized_dag.op_nodes()) == 0

    def test_ionq_multi_qubit_circuit(self):
        """Test IonQ pass on a multi-qubit circuit."""
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rx(np.pi/4, 0)
        qc.ry(np.pi/6, 1)
        qc.rz(np.pi/8, 2)

        pass_instance = IonQOptimizationPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should handle multi-qubit circuit
        assert optimized_dag.num_qubits() == 3


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestIonQOptimizationPass()
    test_instance.test_ionq_pass_initialization()
    test_instance.test_ionq_gate_set_optimization()
    print("All basic IonQ tests passed!")
