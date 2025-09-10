"""
U3 Gate Handling Test

This test verifies that UCC can properly handle circuits with u3 gates
in the target gate set, addressing issue #456.
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from ucc import compile


def test_basic_circuit_with_u3():
    """Test basic circuit compilation with u3 in basis gates."""

    # Create a simple test circuit
    qc = QuantumCircuit(3, name="u3_basic_test")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    print("=" * 60)
    print("U3 GATE HANDLING TEST - Basic Circuit")
    print("=" * 60)
    print("Original circuit:")
    print(qc)
    print(f"Original gate count: {qc.count_ops()}")

    # Test with u3 in basis gates
    try:
        compiled = compile(qc, target_gateset=['u3', 'cx'])
        print("\n✅ Compiled with u3 basis gates successfully!")
        print("Compiled circuit:")
        print(compiled)
        print(f"Compiled gate count: {compiled.count_ops()}")
        return compiled
    except Exception as e:
        print(f"❌ Error with u3 basis gates: {e}")
        raise


def test_qft_with_u3():
    """Test QFT circuit compilation with u3 in basis gates."""

    # Create QFT circuit
    qft_circuit = QFT(3, name="qft_u3_test")

    print("\n" + "=" * 60)
    print("U3 GATE HANDLING TEST - QFT Circuit")
    print("=" * 60)
    print("Original QFT circuit:")
    print(qft_circuit)
    print(f"Original depth: {qft_circuit.depth()}")
    print(f"Original gate count: {qft_circuit.count_ops()}")

    # Test with u3 basis gates
    try:
        compiled_qft = compile(qft_circuit, target_gateset=['u3', 'cx'])
        print("\n✅ QFT compiled with u3 basis gates successfully!")
        print("Compiled QFT circuit:")
        print(compiled_qft)
        print(f"Compiled depth: {compiled_qft.depth()}")
        print(f"Compiled gate count: {compiled_qft.count_ops()}")
        return compiled_qft
    except Exception as e:
        print(f"❌ Error with QFT and u3 basis gates: {e}")
        raise


def test_u3_with_rotations():
    """Test circuit with explicit rotation gates compiled to u3 basis."""

    # Create circuit with various rotation gates
    qc = QuantumCircuit(2, name="rotation_u3_test")
    qc.rx(3.14159/4, 0)  # π/4
    qc.ry(3.14159/3, 1)  # π/3
    qc.rz(3.14159/6, 0)  # π/6
    qc.cx(0, 1)
    qc.u(3.14159/2, 0, 3.14159/4, 1)  # U3 gate

    print("\n" + "=" * 60)
    print("U3 GATE HANDLING TEST - Rotation Gates")
    print("=" * 60)
    print("Original circuit with rotations:")
    print(qc)
    print(f"Original gate count: {qc.count_ops()}")

    # Test compilation with u3 basis
    try:
        compiled = compile(qc, target_gateset=['u3', 'cx'])
        print("\n✅ Rotations compiled with u3 basis successfully!")
        print("Compiled circuit:")
        print(compiled)
        print(f"Compiled gate count: {compiled.count_ops()}")
        return compiled
    except Exception as e:
        print(f"❌ Error compiling rotations with u3 basis: {e}")
        raise


def main():
    """Run all U3 gate handling tests."""
    print("Starting U3 Gate Handling Tests...")

    try:
        # Test basic circuit
        test_basic_circuit_with_u3()

        # Test QFT circuit
        test_qft_with_u3()

        # Test rotations
        test_u3_with_rotations()

        print("\n" + "=" * 60)
        print("✅ ALL U3 GATE HANDLING TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ ERROR during U3 gate handling testing: {e}")
        raise


if __name__ == "__main__":
    main()
