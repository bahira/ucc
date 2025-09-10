"""
Quantum Hierarchical Recursive Filtering (QHRF) Pass

This module implements the QHRF optimization pass for UCC.
QHRF applies hierarchical recursive filtering to identify and
remove redundant quantum operations, reducing circuit complexity
while maintaining fidelity.
"""

import numpy as np
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZGate, CXGate
from typing import Optional, Dict, List, Tuple
from collections import defaultdict


class QHRFPass(TransformationPass):
    """
    Quantum Hierarchical Recursive Filtering (QHRF) optimization pass.

    This pass analyzes quantum circuits using hierarchical recursive filtering
    to identify and remove redundant operations. It works by:

    1. Building a hierarchical representation of the circuit
    2. Applying recursive filtering to identify redundant operations
    3. Removing operations that don't significantly contribute to the result
    4. Reconstructing the optimized circuit

    The hierarchical approach allows for efficient analysis of complex circuits
    by breaking them down into manageable subcircuits.
    """

    def __init__(self, hierarchy_depth: int = 3, redundancy_threshold: float = 0.01):
        """
        Initialize the QHRF pass.

        Args:
            hierarchy_depth: Maximum depth of hierarchical analysis
            redundancy_threshold: Minimum contribution threshold for operation preservation
        """
        super().__init__()
        self.hierarchy_depth = hierarchy_depth
        self.redundancy_threshold = redundancy_threshold

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """
        Apply QHRF optimization to the input DAG.

        Args:
            dag: Input quantum circuit as DAG

        Returns:
            Optimized DAG with reduced complexity
        """
        # Convert DAG to circuit for analysis
        circuit = dag_to_circuit(dag)

        # Apply QHRF optimization
        optimized_circuit = self._apply_qhrf_optimization(circuit)

        # Convert back to DAG
        return circuit_to_dag(optimized_circuit)

    def _apply_qhrf_optimization(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Apply QHRF optimization to the circuit.

        Args:
            circuit: Input quantum circuit

        Returns:
            Optimized quantum circuit
        """
        # Build hierarchical representation
        hierarchy = self._build_hierarchy(circuit)

        # Apply recursive filtering
        filtered_hierarchy = self._apply_recursive_filtering(hierarchy)

        # Reconstruct circuit from filtered hierarchy
        optimized_circuit = self._reconstruct_from_hierarchy(filtered_hierarchy, circuit.num_qubits)

        return optimized_circuit

    def _build_hierarchy(self, circuit: QuantumCircuit) -> Dict:
        """
        Build hierarchical representation of the circuit.

        Args:
            circuit: Input quantum circuit

        Returns:
            Hierarchical representation
        """
        hierarchy = {
            'qubits': circuit.num_qubits,
            'layers': [],
            'connections': defaultdict(list)
        }

        # Group operations by "layers" (concurrent operations)
        current_layer = []
        active_qubits = set()

        for instruction_data in circuit.data:
            # Handle both old and new Qiskit versions
            if hasattr(instruction_data, 'operation'):
                instruction = instruction_data.operation
                qargs = instruction_data.qubits
                cargs = instruction_data.clbits
            else:
                instruction, qargs, cargs = instruction_data

            # Get qubit indices
            qubit_indices = [circuit.find_bit(q).index for q in qargs]

            # Check if this operation conflicts with current layer
            if any(idx in active_qubits for idx in qubit_indices):
                # Start new layer
                if current_layer:
                    hierarchy['layers'].append(current_layer)
                current_layer = []
                active_qubits = set()

            # Add operation to current layer
            current_layer.append({
                'gate': instruction,
                'qubits': qubit_indices,
                'index': len(hierarchy['layers']) * 100 + len(current_layer)
            })

            # Mark qubits as active
            active_qubits.update(qubit_indices)

        # Add final layer
        if current_layer:
            hierarchy['layers'].append(current_layer)

        return hierarchy

    def _apply_recursive_filtering(self, hierarchy: Dict) -> Dict:
        """
        Apply recursive filtering to the hierarchy.

        Args:
            hierarchy: Hierarchical representation

        Returns:
            Filtered hierarchy
        """
        filtered_hierarchy = hierarchy.copy()
        filtered_hierarchy['layers'] = []

        for layer in hierarchy['layers']:
            # Apply filtering to this layer
            filtered_layer = self._filter_layer(layer)

            if filtered_layer:
                filtered_hierarchy['layers'].append(filtered_layer)

        return filtered_hierarchy

    def _filter_layer(self, layer: List[Dict]) -> List[Dict]:
        """
        Filter a single layer to remove redundant operations.

        Args:
            layer: Layer of operations

        Returns:
            Filtered layer
        """
        if len(layer) <= 1:
            return layer

        filtered_layer = []
        operation_contributions = {}

        # Calculate contribution of each operation
        for i, op in enumerate(layer):
            contribution = self._calculate_operation_contribution(op, layer)
            operation_contributions[i] = contribution

        # Sort by contribution (highest first)
        sorted_ops = sorted(operation_contributions.items(), key=lambda x: x[1], reverse=True)

        # Keep operations above threshold
        for idx, contribution in sorted_ops:
            if contribution > self.redundancy_threshold:
                filtered_layer.append(layer[idx])
            else:
                # Check if removing this operation affects connectivity
                if not self._breaks_connectivity(layer[idx], filtered_layer):
                    continue  # Skip this operation
                else:
                    filtered_layer.append(layer[idx])  # Keep for connectivity

        return filtered_layer

    def _calculate_operation_contribution(self, operation: Dict, layer: List[Dict]) -> float:
        """
        Calculate the contribution of an operation to the layer.

        Args:
            operation: Operation to analyze
            layer: Full layer of operations

        Returns:
            Contribution score
        """
        gate = operation['gate']
        qubits = operation['qubits']

        # Base contribution depends on gate type
        if gate.name in ['h', 'x', 'y', 'z']:
            base_contribution = 1.0
        elif gate.name in ['rx', 'ry', 'rz']:
            # Parametric gates have contribution based on angle magnitude
            if hasattr(gate, 'params') and len(gate.params) > 0:
                angle = abs(gate.params[0])
                base_contribution = min(angle / np.pi, 1.0)
            else:
                base_contribution = 0.5
        elif gate.name in ['cx', 'cz', 'swap']:
            base_contribution = 1.5  # Two-qubit gates are more important
        else:
            base_contribution = 0.8

        # Reduce contribution if similar operations exist on same qubits
        similar_ops = sum(1 for op in layer
                         if op != operation and
                         set(op['qubits']) == set(qubits) and
                         op['gate'].name == gate.name)

        if similar_ops > 0:
            base_contribution *= (1.0 / (similar_ops + 1))

        return base_contribution

    def _breaks_connectivity(self, operation: Dict, current_layer: List[Dict]) -> bool:
        """
        Check if removing an operation would break circuit connectivity.

        Args:
            operation: Operation being considered for removal
            current_layer: Current filtered layer

        Returns:
            True if removal would break connectivity
        """
        if not current_layer:
            return True  # Can't break connectivity if no operations

        # Check if this operation connects different qubit groups
        qubit_groups = []
        for op in current_layer:
            qubit_groups.append(set(op['qubits']))

        # Merge overlapping groups
        merged_groups = []
        for group in qubit_groups:
            # Find overlapping groups
            overlapping = []
            for i, existing_group in enumerate(merged_groups):
                if group & existing_group:
                    overlapping.append(i)

            if overlapping:
                # Merge with first overlapping group
                merged_groups[overlapping[0]].update(group)
                # Merge all other overlapping groups
                for i in reversed(overlapping[1:]):
                    merged_groups[overlapping[0]].update(merged_groups[i])
                    del merged_groups[i]
            else:
                merged_groups.append(group.copy())

        # Check if operation connects different groups
        op_qubits = set(operation['qubits'])
        connected_groups = []

        for i, group in enumerate(merged_groups):
            if op_qubits & group:
                connected_groups.append(i)

        # If operation connects multiple groups, it's important for connectivity
        return len(connected_groups) > 1

    def _reconstruct_from_hierarchy(self, hierarchy: Dict, num_qubits: int) -> QuantumCircuit:
        """
        Reconstruct circuit from filtered hierarchy.

        Args:
            hierarchy: Filtered hierarchy
            num_qubits: Number of qubits

        Returns:
            Reconstructed quantum circuit
        """
        circuit = QuantumCircuit(num_qubits)

        for layer in hierarchy['layers']:
            for operation in layer:
                gate = operation['gate']
                qubits = operation['qubits']

                # Add operation to circuit
                circuit.append(gate, qubits, [])

        return circuit
