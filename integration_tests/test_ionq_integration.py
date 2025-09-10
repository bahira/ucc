"""
Integration test for IonQ Hardware Optimization Pass

This test demonstrates the IonQ optimization pass working with the full UCC
compilation pipeline, including gate decomposition and hardware-specific optimizations.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from ucc import compile
from ucc.transpilers.ionq_pass import IonQOptimizationPass


def test_ionq_basic_integration():
    """Test basic IonQ pass integration with UCC compilation."""

    # Create a test circuit with gates that need IonQ optimization
    qc = QuantumCircuit(3, name="ionq_test_circuit")
    qc.h(0)  # Will be decomposed to RX, RY
    qc.x(1)  # Will be decomposed to RX
    qc.y(2)  # Will be decomposed to RY
    qc.cx(0, 1)  # Will be decomposed using RXX
    qc.z(2)  # Will be decomposed to RZ
    qc.rx(np.pi/4, 0)
    qc.ry(np.pi/6, 1)
    qc.rz(np.pi/8, 2)

    print("=" * 60)
    print("IONQ INTEGRATION TEST - Basic Circuit")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Compile with IonQ pass
    compiled_circuit = compile(qc, custom_passes=[IonQOptimizationPass()])

    print("\nCompiled with IonQ optimization:")
    print(compiled_circuit)
    print(f"Compiled depth: {compiled_circuit.depth()}")
    print(f"Compiled gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def test_ionq_optimization_levels():
    """Test IonQ pass with different optimization levels."""

    # Create a more complex test circuit
    qc = QuantumCircuit(4, name="complex_ionq_test")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.rx(np.pi/3, 0)
    qc.ry(np.pi/4, 1)
    qc.rz(np.pi/5, 2)
    qc.u(np.pi/6, np.pi/7, np.pi/8, 3)  # U3 gate

    print("\n" + "=" * 60)
    print("IONQ INTEGRATION TEST - Optimization Levels")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original depth: {qc.depth()}")
    print(f"Original gate count: {qc.count_ops()}")

    # Test with different optimization levels
    levels = [1, 2, 3]

    for level in levels:
        print(f"\n--- Optimization Level {level} ---")
        compiled_circuit = compile(qc, custom_passes=[IonQOptimizationPass(optimization_level=level)])
        print(f"Level {level} depth: {compiled_circuit.depth()}")
        print(f"Level {level} gate count: {compiled_circuit.count_ops()}")

    return compiled_circuit


def test_ionq_qft_circuit():
    """Test IonQ pass with QFT circuit."""

    # Create QFT circuit
    qft_circuit = QFT(3, name="qft_test")

    print("\n" + "=" * 60)
    print("IONQ INTEGRATION TEST - QFT Circuit")
    print("=" * 60)
    print("Original QFT circuit:")
    print(qft_circuit)
    print(f"Original depth: {qft_circuit.depth()}")
    print(f"Original gate count: {qft_circuit.count_ops()}")

    # Compile with IonQ optimization
    compiled_qft = compile(qft_circuit, custom_passes=[IonQOptimizationPass()])

    print("\nQFT compiled with IonQ optimization:")
    print(compiled_qft)
    print(f"Compiled depth: {compiled_qft.depth()}")
    print(f"Compiled gate count: {compiled_qft.count_ops()}")

    return compiled_qft


def main():
    """Run all IonQ integration tests."""
    print("Starting IonQ Hardware Optimization Integration Tests...")

    try:
        # Run basic integration test
        test_ionq_basic_integration()

        # Run optimization levels test
        test_ionq_optimization_levels()

        # Run QFT test
        test_ionq_qft_circuit()

        print("\n" + "=" * 60)
        print("✅ ALL IONQ INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR during IonQ integration testing: {e}")
        raise


if __name__ == "__main__":
    main()
