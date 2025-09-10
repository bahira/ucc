"""
Integration test for QHRF (Quantum Hierarchical Recursive Filtering) Pass

This test demonstrates the QHRF optimization pass working with the full UCC
compilation pipeline, including hierarchical filtering and redundancy removal.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from ucc import compile
from ucc.transpilers.qhrf_pass import QHRFPass


def test_qhrf_basic_integration():
    """Test basic QHRF pass integration with UCC compilation."""

    # Create a test circuit with redundant operations
    qc = QuantumCircuit(3, name="qhrf_test_circuit")
    qc.h(0)
    qc.rx(0.01, 0)  # Small rotation that might be filtered
    qc.ry(np.pi/2, 0)  # Large rotation
    qc.cx(0, 1)
    qc.rz(0.005, 1)  # Small rotation that might be filtered
    qc.cx(1, 2)
    qc.rx(np.pi/4, 2)
    qc.ry(0.02, 2)  # Small rotation that might be filtered

    print("=" * 60)
    print("QHRF INTEGRATION TEST - Basic Circuit")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Compile with QHRF pass
    compiled_circuit = compile(qc, custom_passes=[QHRFPass(redundancy_threshold=0.1)])

    print("\nCompiled with QHRF (threshold=0.1):")
    print(compiled_circuit)
    print(f"Compiled depth: {compiled_circuit.depth()}")
    print(f"Compiled gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def test_qhrf_threshold_comparison():
    """Test QHRF pass with different redundancy thresholds."""

    # Create circuit with various small rotations
    qc = QuantumCircuit(4, name="threshold_test_circuit")
    qc.h(0)
    qc.rx(0.001, 0)  # Very small rotation
    qc.ry(0.01, 0)   # Small rotation
    qc.rz(0.1, 0)    # Medium rotation
    qc.cx(0, 1)
    qc.rx(0.005, 1)  # Small rotation
    qc.ry(0.02, 1)   # Small rotation
    qc.cx(1, 2)
    qc.rz(0.003, 2)  # Very small rotation
    qc.cx(2, 3)

    print("\n" + "=" * 60)
    print("QHRF INTEGRATION TEST - Threshold Comparison")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Test with different thresholds
    thresholds = [0.001, 0.01, 0.1, 0.5]

    for threshold in thresholds:
        print(f"\n--- Threshold {threshold} ---")
        compiled_circuit = compile(qc, custom_passes=[QHRFPass(redundancy_threshold=threshold)])
        print(f"Threshold {threshold} depth: {compiled_circuit.depth()}")
        print(f"Threshold {threshold} gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def test_qhrf_complex_circuit():
    """Test QHRF pass with a more complex circuit."""

    # Create a complex circuit with multiple layers
    qc = QuantumCircuit(3, name="complex_qhrf_test")

    # Layer 1
    qc.h(0)
    qc.h(1)
    qc.h(2)

    # Layer 2 - small rotations that might be filtered
    qc.rx(0.01, 0)
    qc.ry(0.005, 1)
    qc.rz(0.02, 2)

    # Layer 3 - entangling gates
    qc.cx(0, 1)
    qc.cx(1, 2)

    # Layer 4 - more small rotations
    qc.rx(0.003, 0)
    qc.ry(0.008, 1)
    qc.rz(0.015, 2)

    # Layer 5 - final rotations
    qc.rx(np.pi/4, 0)
    qc.ry(np.pi/3, 1)
    qc.rz(np.pi/6, 2)

    print("\n" + "=" * 60)
    print("QHRF INTEGRATION TEST - Complex Circuit")
    print("=" * 60)
    print("Original complex circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Compile with QHRF
    compiled_circuit = compile(qc, custom_passes=[QHRFPass(redundancy_threshold=0.01)])

    print("\nCompiled with QHRF (threshold=0.01):")
    print(compiled_circuit)
    print(f"Compiled depth: {compiled_circuit.depth()}")
    print(f"Compiled gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def main():
    """Run all QHRF integration tests."""
    print("Starting QHRF Optimization Integration Tests...")

    try:
        # Run basic integration test
        test_qhrf_basic_integration()

        # Run threshold comparison test
        test_qhrf_threshold_comparison()

        # Run complex circuit test
        test_qhrf_complex_circuit()

        print("\n" + "=" * 60)
        print("✅ ALL QHRF INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR during QHRF integration testing: {e}")
        raise


if __name__ == "__main__":
    main()
