"""
Integration test for SeqHT (Sequency Hierarchy Truncation) Pass

This test demonstrates the SeqHT optimization pass working with the full UCC
compilation pipeline, including sequency domain analysis and hierarchical truncation.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from ucc import compile
from ucc.transpilers.seqht_pass import SeqHTPass


def test_seqht_basic_integration():
    """Test basic SeqHT pass integration with UCC compilation."""

    # Create a test circuit with multiple rotation gates
    qc = QuantumCircuit(2, name="seqht_test_circuit")
    qc.h(0)
    qc.rx(np.pi/4, 0)
    qc.ry(np.pi/6, 0)
    qc.rz(np.pi/8, 0)
    qc.cx(0, 1)
    qc.rx(np.pi/3, 1)
    qc.ry(np.pi/5, 1)

    print("=" * 60)
    print("SEQHT INTEGRATION TEST - Basic Circuit")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Compile with SeqHT pass
    compiled_circuit = compile(qc, custom_passes=[SeqHTPass(truncation_threshold=0.01)])

    print("\nCompiled with SeqHT (threshold=0.01):")
    print(compiled_circuit)
    print(f"Compiled depth: {compiled_circuit.depth()}")
    print(f"Compiled gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def test_seqht_threshold_comparison():
    """Test SeqHT pass with different truncation thresholds."""

    # Create circuit with various rotation angles
    qc = QuantumCircuit(3, name="threshold_test_circuit")
    qc.h(0)
    qc.rx(np.pi/100, 0)  # Very small angle
    qc.ry(np.pi/50, 0)   # Small angle
    qc.rz(np.pi/10, 0)   # Medium angle
    qc.cx(0, 1)
    qc.rx(np.pi/20, 1)   # Small-medium angle
    qc.ry(np.pi/4, 1)    # Large angle
    qc.cx(1, 2)
    qc.rz(np.pi/200, 2)  # Very small angle

    print("\n" + "=" * 60)
    print("SEQHT INTEGRATION TEST - Threshold Comparison")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Test with different thresholds
    thresholds = [0.001, 0.01, 0.1, 0.5]

    for threshold in thresholds:
        print(f"\n--- Threshold {threshold} ---")
        compiled_circuit = compile(qc, custom_passes=[SeqHTPass(truncation_threshold=threshold)])
        print(f"Threshold {threshold} depth: {compiled_circuit.depth()}")
        print(f"Threshold {threshold} gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def test_seqht_qft_circuit():
    """Test SeqHT pass with QFT circuit."""

    # Create QFT circuit which has many rotation gates
    qft_circuit = QFT(3, name="qft_seqht_test")

    print("\n" + "=" * 60)
    print("SEQHT INTEGRATION TEST - QFT Circuit")
    print("=" * 60)
    print("Original QFT circuit:")
    print(qft_circuit)
    print(f"Original depth: {qft_circuit.depth()}")
    print(f"Original gate count: {qft_circuit.count_ops()}")

    # Compile with SeqHT optimization
    compiled_qft = compile(qft_circuit, custom_passes=[SeqHTPass(truncation_threshold=0.01)])

    print("\nQFT compiled with SeqHT (threshold=0.01):")
    print(compiled_qft)
    print(f"Compiled depth: {compiled_qft.depth()}")
    print(f"Compiled gate count: {compiled_qft.count_ops()}")

    return compiled_qft


def test_seqht_multi_qubit_rotations():
    """Test SeqHT pass with circuits having many rotation gates on multiple qubits."""

    # Create circuit with rotations on multiple qubits
    qc = QuantumCircuit(4, name="multi_qubit_rotations")

    # Apply rotations to all qubits
    for i in range(4):
        qc.rx(np.pi/(i+2), i)
        qc.ry(np.pi/(i+3), i)
        qc.rz(np.pi/(i+4), i)

    # Add some entangling gates
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # More rotations
    for i in range(4):
        qc.rx(np.pi/(i+5), i)
        qc.ry(np.pi/(i+6), i)

    print("\n" + "=" * 60)
    print("SEQHT INTEGRATION TEST - Multi-Qubit Rotations")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Compile with SeqHT
    compiled_circuit = compile(qc, custom_passes=[SeqHTPass(truncation_threshold=0.05)])

    print("\nCompiled with SeqHT (threshold=0.05):")
    print(compiled_circuit)
    print(f"Compiled depth: {compiled_circuit.depth()}")
    print(f"Compiled gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def main():
    """Run all SeqHT integration tests."""
    print("Starting SeqHT Optimization Integration Tests...")

    try:
        # Run basic integration test
        test_seqht_basic_integration()

        # Run threshold comparison test
        test_seqht_threshold_comparison()

        # Run QFT test
        test_seqht_qft_circuit()

        # Run multi-qubit rotations test
        test_seqht_multi_qubit_rotations()

        print("\n" + "=" * 60)
        print("✅ ALL SEQHT INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR during SeqHT integration testing: {e}")
        raise


if __name__ == "__main__":
    main()
