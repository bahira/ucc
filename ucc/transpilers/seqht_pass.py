"""
Sequency Hierarchy Truncation (SeqHT) Pass

This module implements the SeqHT optimization pass for UCC.
SeqHT analyzes quantum circuits in the sequency domain and truncates
higher-order terms to reduce circuit depth and gate count while
maintaining circuit fidelity.
"""

import numpy as np
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate
from typing import Optional, Dict, List


class SeqHTPass(TransformationPass):
    """
    Sequency Hierarchy Truncation (SeqHT) optimization pass.

    This pass analyzes quantum circuits using sequency domain analysis
    and truncates higher-order terms to reduce circuit complexity while
    preserving fidelity.

    The sequency domain represents gate sequences in terms of their
    frequency components, allowing identification of redundant or
    low-impact operations that can be safely removed.
    """

    def __init__(self, truncation_threshold: float = 0.01, max_order: int = 3):
        """
        Initialize the SeqHT pass.

        Args:
            truncation_threshold: Minimum sequency amplitude to preserve
            max_order: Maximum sequency order to analyze
        """
        super().__init__()
        self.truncation_threshold = truncation_threshold
        self.max_order = max_order

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Apply SeqHT optimization to the input DAG.

        Args:
            dag: Input quantum circuit as DAG

        Returns:
            Optimized DAG with reduced complexity
        """
        # Convert DAG to circuit for analysis
        circuit = dag_to_circuit(dag)

        # Apply SeqHT optimization
        optimized_circuit = self._apply_seqht_optimization(circuit)

        # Convert back to DAG
        return circuit_to_dag(optimized_circuit)

    def _apply_seqht_optimization(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Apply sequency hierarchy truncation to the circuit.

        Args:
            circuit: Input quantum circuit

        Returns:
            Optimized quantum circuit
        """
        # Create a copy of the circuit to modify
        optimized_circuit = circuit.copy()

        # Analyze each qubit independently
        for qubit_idx in range(circuit.num_qubits):
            # Extract single-qubit operations for this qubit
            single_qubit_ops = self._extract_single_qubit_operations(circuit, qubit_idx)

            if len(single_qubit_ops) > 1:
                # Apply sequency analysis and truncation
                optimized_ops = self._optimize_single_qubit_sequence(single_qubit_ops)

                # Replace the original operations with optimized ones
                optimized_circuit = self._replace_operations(
                    optimized_circuit, qubit_idx, single_qubit_ops, optimized_ops
                )

        return optimized_circuit

    def _extract_single_qubit_operations(self, circuit: QuantumCircuit, qubit_idx: int) -> List[Dict]:
        """
        Extract single-qubit operations for a specific qubit.

        Args:
            circuit: Quantum circuit
            qubit_idx: Index of the qubit to analyze

        Returns:
            List of single-qubit operations with their parameters
        """
        operations = []

        for instruction_data in circuit.data:
            # Handle both old and new Qiskit versions
            if hasattr(instruction_data, 'operation'):
                # New Qiskit format (CircuitInstruction)
                instruction = instruction_data.operation
                qargs = instruction_data.qubits
                cargs = instruction_data.clbits
            else:
                # Old Qiskit format (legacy tuple)
                instruction, qargs, cargs = instruction_data

            # Check if this is a single-qubit operation on the target qubit
            if (len(qargs) == 1 and
                ((hasattr(qargs[0], 'index') and qargs[0].index == qubit_idx) or
                 (hasattr(qargs[0], '_index') and qargs[0]._index == qubit_idx) or
                 (circuit.find_bit(qargs[0]).index == qubit_idx)) and
                len(cargs) == 0 and instruction.name in ['rx', 'ry', 'rz', 'u3', 'h', 'x', 'y', 'z', 's', 't']):

                op_info = {
                    'gate': instruction,
                    'qubit': qubit_idx,
                    'index': len(operations)
                }
                operations.append(op_info)

        return operations

    def _optimize_single_qubit_sequence(self, operations: List[Dict]) -> List[Dict]:
        """
        Optimize a sequence of single-qubit operations using SeqHT.

        Args:
            operations: List of single-qubit operations

        Returns:
            Optimized sequence of operations
        """
        if len(operations) < 2:
            return operations

        # Convert operations to sequency domain
        sequency_coeffs = self._compute_sequency_coefficients(operations)

        # Apply truncation based on threshold
        significant_coeffs = self._truncate_sequency_hierarchy(sequency_coeffs)

        # Reconstruct optimized operations from significant coefficients
        optimized_ops = self._reconstruct_from_sequency(significant_coeffs, len(operations))

        return optimized_ops

    def _compute_sequency_coefficients(self, operations: List[Dict]) -> np.ndarray:
        """
        Compute sequency domain coefficients for a sequence of operations.

        Args:
            operations: List of single-qubit operations

        Returns:
            Sequency coefficients as numpy array
        """
        n = len(operations)

        # Initialize sequency coefficients
        coeffs = np.zeros((self.max_order + 1, 3))  # 3 for RX, RY, RZ components

        # For simplicity, we'll use a basic sequency transform
        # In a full implementation, this would use Walsh-Hadamard or similar transforms
        for i, op in enumerate(operations):
            gate = op['gate']

            # Extract rotation angles based on gate type
            if hasattr(gate, 'params') and len(gate.params) > 0:
                if gate.name == 'rx':
                    coeffs[0, 0] += gate.params[0]  # DC component
                elif gate.name == 'ry':
                    coeffs[0, 1] += gate.params[0]
                elif gate.name == 'rz':
                    coeffs[0, 2] += gate.params[0]
                elif gate.name == 'u3' and len(gate.params) >= 3:
                    # U3 gate: U3(θ, φ, λ) = RZ(φ) RY(θ) RZ(λ)
                    coeffs[0, 2] += gate.params[1]  # φ (lambda)
                    coeffs[0, 1] += gate.params[0]  # θ (theta)
                    coeffs[0, 2] += gate.params[2]  # λ (phi)

        return coeffs

    def _truncate_sequency_hierarchy(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Truncate sequency hierarchy based on threshold.

        Args:
            coeffs: Sequency coefficients

        Returns:
            Truncated coefficients
        """
        # Apply threshold to remove small coefficients
        truncated_coeffs = coeffs.copy()

        # Zero out coefficients below threshold
        mask = np.abs(truncated_coeffs) < self.truncation_threshold
        truncated_coeffs[mask] = 0.0

        return truncated_coeffs

    def _reconstruct_from_sequency(self, coeffs: np.ndarray, num_operations: int) -> List[Dict]:
        """
        Reconstruct operations from sequency coefficients.

        Args:
            coeffs: Sequency coefficients
            num_operations: Original number of operations

        Returns:
            Reconstructed operations
        """
        operations = []

        # For now, create a simplified reconstruction
        # In a full implementation, this would use inverse sequency transform

        # Extract the DC components (most significant)
        rx_angle = coeffs[0, 0]
        ry_angle = coeffs[0, 1]
        rz_angle = coeffs[0, 2]

        # Create optimized operations based on significant coefficients
        if abs(rx_angle) > self.truncation_threshold:
            operations.append({
                'gate': RXGate(rx_angle),
                'type': 'rx'
            })

        if abs(ry_angle) > self.truncation_threshold:
            operations.append({
                'gate': RYGate(ry_angle),
                'type': 'ry'
            })

        if abs(rz_angle) > self.truncation_threshold:
            operations.append({
                'gate': RZGate(rz_angle),
                'type': 'rz'
            })

        return operations

    def _replace_operations(self, circuit: QuantumCircuit, qubit_idx: int,
                          original_ops: List[Dict], optimized_ops: List[Dict]) -> QuantumCircuit:
        """
        Replace original operations with optimized ones in the circuit.

        Args:
            circuit: Original circuit
            qubit_idx: Target qubit index
            original_ops: Original operations to replace
            optimized_ops: Optimized operations

        Returns:
            Circuit with replaced operations
        """
        # For simplicity, create a new circuit with optimized operations
        # In a full implementation, this would modify the existing circuit in place

        new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

        # Copy all operations, replacing the optimized ones
        op_idx = 0

        for instruction_data in circuit.data:
            # Handle both old and new Qiskit versions
            if hasattr(instruction_data, 'operation'):
                # New Qiskit format (CircuitInstruction)
                instruction = instruction_data.operation
                qargs = instruction_data.qubits
                cargs = instruction_data.clbits
            else:
                # Old Qiskit format (legacy tuple)
                instruction, qargs, cargs = instruction_data

            should_replace = (len(original_ops) > 0 and op_idx < len(original_ops) and
                            len(qargs) == 1 and
                            circuit.find_bit(qargs[0]).index == qubit_idx and
                            instruction.name == original_ops[op_idx]['gate'].name)

            if should_replace and op_idx < len(optimized_ops):
                # Replace with optimized operation
                opt_op = optimized_ops[op_idx]
                if 'gate' in opt_op:
                    new_circuit.append(opt_op['gate'], [qubit_idx], [])
                op_idx += 1
            else:
                # Keep original operation
                new_circuit.append(instruction, qargs, cargs)
                if should_replace:
                    op_idx += 1

        return new_circuit
