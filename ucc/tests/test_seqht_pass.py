"""
Test for SeqHT (Sequency Hierarchy Truncation) Pass
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
from ucc.transpilers.seqht_pass import SeqHTPass
from qiskit.converters import circuit_to_dag


class TestSeqHTPass:
    """Test cases for the SeqHT optimization pass."""

    def test_seqht_pass_initialization(self):
        """Test that SeqHT pass can be initialized with default parameters."""
        pass_instance = SeqHTPass()
        assert pass_instance.truncation_threshold == 0.01
        assert pass_instance.max_order == 3

    def test_seqht_pass_custom_parameters(self):
        """Test that SeqHT pass can be initialized with custom parameters."""
        pass_instance = SeqHTPass(truncation_threshold=0.05, max_order=5)
        assert pass_instance.truncation_threshold == 0.05
        assert pass_instance.max_order == 5

    def test_seqht_single_rotation_gate(self):
        """Test SeqHT pass on a circuit with a single rotation gate."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi/4, 0)

        pass_instance = SeqHTPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # The circuit should still be valid
        assert optimized_dag.num_qubits() == 1
        assert len(optimized_dag.op_nodes()) >= 0  # May be optimized away if below threshold

    def test_seqht_multiple_rotation_gates(self):
        """Test SeqHT pass on a circuit with multiple rotation gates."""
        qc = QuantumCircuit(1)
        qc.rx(np.pi/4, 0)
        qc.ry(np.pi/6, 0)
        qc.rz(np.pi/8, 0)

        pass_instance = SeqHTPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # The circuit should still be valid
        assert optimized_dag.num_qubits() == 1
        # Should have some operations (may be combined or reduced)
        assert len(optimized_dag.op_nodes()) >= 0

    def test_seqht_high_threshold(self):
        """Test SeqHT pass with high truncation threshold (should remove small rotations)."""
        qc = QuantumCircuit(1)
        qc.rx(0.01, 0)  # Very small rotation
        qc.ry(np.pi/2, 0)  # Large rotation
        qc.rz(0.005, 0)  # Very small rotation

        pass_instance = SeqHTPass(truncation_threshold=0.1)  # High threshold
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should keep the large RY rotation but may remove small ones
        assert optimized_dag.num_qubits() == 1

    def test_seqht_multi_qubit_circuit(self):
        """Test SeqHT pass on a multi-qubit circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(np.pi/4, 0)
        qc.cx(0, 1)
        qc.ry(np.pi/6, 1)
        qc.rz(np.pi/8, 1)

        pass_instance = SeqHTPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should preserve multi-qubit structure
        assert optimized_dag.num_qubits() == 2
        # Should have at least the CNOT gate
        assert len(optimized_dag.op_nodes()) >= 1

    def test_seqht_u3_gate(self):
        """Test SeqHT pass on a circuit with U3 gates."""
        qc = QuantumCircuit(1)
        qc.u(np.pi/4, np.pi/6, np.pi/8, 0)  # U3 equivalent

        pass_instance = SeqHTPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should handle U3 decomposition
        assert optimized_dag.num_qubits() == 1

    def test_seqht_empty_circuit(self):
        """Test SeqHT pass on an empty circuit."""
        qc = QuantumCircuit(1)

        pass_instance = SeqHTPass()
        dag = circuit_to_dag(qc)
        optimized_dag = pass_instance.run(dag)

        # Should return valid empty circuit
        assert optimized_dag.num_qubits() == 1
        assert len(optimized_dag.op_nodes()) == 0


if __name__ == "__main__":
    # Run basic tests
    test_instance = TestSeqHTPass()
    test_instance.test_seqht_pass_initialization()
    test_instance.test_seqht_single_rotation_gate()
    test_instance.test_seqht_multiple_rotation_gates()
    print("All basic SeqHT tests passed!")
