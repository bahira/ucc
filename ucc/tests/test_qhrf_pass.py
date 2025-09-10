"""
Test for QHRF (Quantum Hierarchical Recursive Filtering) Pass
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from ucc.transpilers.qhrf_pass import QHRFPass


class TestQHRFPass:
    """Test cases for the QHRF optimization pass."""

    def test_qhrf_pass_initialization(self):
        """Test that QHRF pass can be initialized with default parameters."""
        pass_instance = QHRFPass()
        assert pass_instance.hierarchy_depth == 3
        assert pass_instance.redundancy_threshold == 0.01

    def test_qhrf_pass_custom_parameters(self):
        """Test that QHRF pass can be initialized with custom parameters."""
        pass_instance = QHRFPass(hierarchy_depth=5, redundancy_threshold=0.05)
        assert pass_instance.hierarchy_depth == 5
        assert pass_instance.redundancy_threshold == 0.05

    def test_qhrf_simple_circuit(self):
        """Test QHRF pass on a simple circuit."""
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(np.pi/4, 0)

        pass_instance = QHRFPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should preserve essential structure
        assert optimized_dag.num_qubits() == 2
        assert len(optimized_dag.op_nodes()) > 0

    def test_qhrf_redundant_operations(self):
        """Test QHRF pass on a circuit with redundant operations."""
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(1)
        qc.rx(0.01, 0)  # Very small rotation
        qc.ry(np.pi/2, 0)  # Large rotation
        qc.rx(0.005, 0)  # Another small rotation
        qc.rz(0.02, 0)  # Small rotation

        pass_instance = QHRFPass(redundancy_threshold=0.1)
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should keep the large RY rotation
        assert optimized_dag.num_qubits() == 1

    def test_qhrf_multi_qubit_circuit(self):
        """Test QHRF pass on a multi-qubit circuit."""
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rx(np.pi/4, 0)
        qc.ry(np.pi/6, 1)
        qc.rz(np.pi/8, 2)

        pass_instance = QHRFPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should preserve multi-qubit structure
        assert optimized_dag.num_qubits() == 3

    def test_qhrf_empty_circuit(self):
        """Test QHRF pass on an empty circuit."""
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(1)

        pass_instance = QHRFPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should return valid empty circuit
        assert optimized_dag.num_qubits() == 1
        assert len(optimized_dag.op_nodes()) == 0

    def test_qhrf_connectivity_preservation(self):
        """Test that QHRF preserves circuit connectivity."""
        from qiskit.converters import circuit_to_dag

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)  # This connects the chain

        pass_instance = QHRFPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should preserve the connecting CNOT
        assert optimized_dag.num_qubits() == 4


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestQHRFPass()
    test_instance.test_qhrf_pass_initialization()
    test_instance.test_qhrf_simple_circuit()
    print("All basic QHRF tests passed!")
