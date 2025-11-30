"""
DAG (Directed Acyclic Graph) Representation for Quantum Circuits.
The internal object model that tracks qubit connectivity, gate ordering, and classical registers.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class GateType(Enum):
    """Types of quantum gates."""
    SINGLE_QUBIT = "single"
    TWO_QUBIT = "two"
    MULTI_QUBIT = "multi"
    MEASUREMENT = "measurement"
    BARRIER = "barrier"
    RESET = "reset"


@dataclass
class GateNode:
    """Represents a gate operation in the DAG."""
    id: int
    name: str
    gate_type: GateType
    qubits: list[int]
    params: list[float] = field(default_factory=list)
    classical_bits: list[int] = field(default_factory=list)
    condition: dict[str, Any] | None = None
    
    def __hash__(self):
        return hash(self.id)
    
    @property
    def is_parameterized(self) -> bool:
        return len(self.params) > 0
    
    @property
    def qubit_count(self) -> int:
        return len(self.qubits)


@dataclass
class QubitWire:
    """Represents a qubit wire in the circuit."""
    index: int
    gates: list[int] = field(default_factory=list)  # Gate IDs in order
    
    @property
    def depth(self) -> int:
        return len(self.gates)


@dataclass
class ClassicalRegister:
    """Represents a classical register for measurement results."""
    name: str
    size: int
    values: list[int] = field(default_factory=list)


class CircuitDAG:
    """
    Directed Acyclic Graph representation of a quantum circuit.
    Provides efficient traversal and analysis of circuit structure.
    """
    
    def __init__(self, num_qubits: int, num_classical_bits: int = 0):
        self.num_qubits = num_qubits
        self.num_classical_bits = num_classical_bits
        
        # Qubit wires
        self.qubits: list[QubitWire] = [QubitWire(i) for i in range(num_qubits)]
        
        # Gate storage
        self.gates: dict[int, GateNode] = {}
        self._next_gate_id = 0
        
        # Classical registers
        self.classical_registers: list[ClassicalRegister] = []
        if num_classical_bits > 0:
            self.classical_registers.append(
                ClassicalRegister("c", num_classical_bits)
            )
        
        # Adjacency list for DAG edges (gate_id -> list of successor gate_ids)
        self.edges: dict[int, list[int]] = {}
        self.reverse_edges: dict[int, list[int]] = {}  # For backwards traversal
    
    def add_gate(
        self,
        name: str,
        qubits: list[int],
        params: list[float] | None = None,
        classical_bits: list[int] | None = None,
        condition: dict[str, Any] | None = None
    ) -> int:
        """Add a gate to the DAG and return its ID."""
        # Determine gate type
        if name.lower() in ("measure", "m"):
            gate_type = GateType.MEASUREMENT
        elif name.lower() == "barrier":
            gate_type = GateType.BARRIER
        elif name.lower() == "reset":
            gate_type = GateType.RESET
        elif len(qubits) == 1:
            gate_type = GateType.SINGLE_QUBIT
        elif len(qubits) == 2:
            gate_type = GateType.TWO_QUBIT
        else:
            gate_type = GateType.MULTI_QUBIT
        
        gate_id = self._next_gate_id
        self._next_gate_id += 1
        
        gate = GateNode(
            id=gate_id,
            name=name,
            gate_type=gate_type,
            qubits=qubits,
            params=params or [],
            classical_bits=classical_bits or [],
            condition=condition
        )
        
        self.gates[gate_id] = gate
        self.edges[gate_id] = []
        self.reverse_edges[gate_id] = []
        
        # Add to qubit wires and create edges
        for qubit in qubits:
            if self.qubits[qubit].gates:
                prev_gate_id = self.qubits[qubit].gates[-1]
                if gate_id not in self.edges[prev_gate_id]:
                    self.edges[prev_gate_id].append(gate_id)
                    self.reverse_edges[gate_id].append(prev_gate_id)
            self.qubits[qubit].gates.append(gate_id)
        
        return gate_id
    
    @property
    def nodes(self) -> dict[int, GateNode]:
        """Return all gate nodes in the circuit."""
        return self.gates
    
    def get(self, gate_id: int, default=None) -> GateNode | None:
        """Get a gate node by ID with default fallback."""
        return self.gates.get(gate_id, default)
    
    @property
    def depth(self) -> int:
        """Calculate circuit depth using topological layers."""
        if not self.gates:
            return 0
        
        # Use BFS to find longest path
        depths: dict[int, int] = {}
        
        # Find entry gates (no predecessors)
        entry_gates = [gid for gid, preds in self.reverse_edges.items() if not preds]
        
        for gid in entry_gates:
            depths[gid] = 1
        
        # Process in topological order
        to_process = list(entry_gates)
        while to_process:
            current = to_process.pop(0)
            current_depth = depths[current]
            
            for successor in self.edges.get(current, []):
                new_depth = current_depth + 1
                if successor not in depths or depths[successor] < new_depth:
                    depths[successor] = new_depth
                    if successor not in to_process:
                        to_process.append(successor)
        
        return max(depths.values()) if depths else 0
    
    @property
    def gate_count(self) -> int:
        """Total number of gates."""
        return len([g for g in self.gates.values() 
                   if g.gate_type not in (GateType.BARRIER, GateType.MEASUREMENT)])
    
    @property
    def two_qubit_gate_count(self) -> int:
        """Number of two-qubit gates."""
        return len([g for g in self.gates.values() if g.gate_type == GateType.TWO_QUBIT])
    
    def get_gates_on_qubit(self, qubit: int) -> list[GateNode]:
        """Get all gates acting on a specific qubit."""
        return [self.gates[gid] for gid in self.qubits[qubit].gates]
    
    def get_gate_sequence(self) -> list[GateNode]:
        """Get gates in topological order."""
        visited = set()
        result = []

        def dfs(gate_id: int):
            if gate_id in visited:
                return
            visited.add(gate_id)
            for pred in self.reverse_edges.get(gate_id, []):
                dfs(pred)
            result.append(self.gates[gate_id])

        for gate_id in self.gates:
            dfs(gate_id)

        return result

    def add_measurement(self, qubit: int, clbit: int) -> int:
        """Add a measurement operation to the circuit.
        
        Args:
            qubit: The qubit to measure
            clbit: The classical bit to store the result
            
        Returns:
            Gate ID of the measurement operation
        """
        return self.add_gate("measure", [qubit], classical_bits=[clbit])

    def to_qasm(self) -> str:
        """Convert DAG to OpenQASM 2.0 string.
        
        Returns:
            OpenQASM 2.0 representation of the circuit
        """
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{self.num_qubits}];",
        ]
        
        if self.num_classical_bits > 0:
            lines.append(f"creg c[{self.num_classical_bits}];")
        
        for node in self.get_gate_sequence():
            gate_name = node.name.lower()
            qubits = node.qubits
            params = node.params
            classical_bits = node.classical_bits
            
            if gate_name in ("measure", "m"):
                # Handle measurement
                for q, c in zip(qubits, classical_bits if classical_bits else range(len(qubits))):
                    lines.append(f"measure q[{q}] -> c[{c}];")
            elif gate_name == "barrier":
                qubit_str = ", ".join(f"q[{q}]" for q in qubits)
                lines.append(f"barrier {qubit_str};")
            elif gate_name == "reset":
                for q in qubits:
                    lines.append(f"reset q[{q}];")
            elif params:
                # Parameterized gate
                param_str = ", ".join(str(p) for p in params)
                qubit_str = ", ".join(f"q[{q}]" for q in qubits)
                lines.append(f"{gate_name}({param_str}) {qubit_str};")
            else:
                # Non-parameterized gate
                qubit_str = ", ".join(f"q[{q}]" for q in qubits)
                lines.append(f"{gate_name} {qubit_str};")
        
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize DAG to dictionary."""
        return {
            "num_qubits": self.num_qubits,
            "num_classical_bits": self.num_classical_bits,
            "depth": self.depth,
            "gate_count": self.gate_count,
            "two_qubit_gate_count": self.two_qubit_gate_count,
            "gates": [
                {
                    "id": g.id,
                    "name": g.name,
                    "type": g.gate_type.value,
                    "qubits": g.qubits,
                    "params": g.params,
                    "classical_bits": g.classical_bits
                }
                for g in self.get_gate_sequence()
            ]
        }

    def compose(self, other: "CircuitDAG", qubit_mapping: dict[int, int] | None = None) -> "CircuitDAG":
        """
        Compose this circuit with another circuit (append other to self).
        
        Args:
            other: The circuit to append
            qubit_mapping: Optional mapping from other's qubits to self's qubits.
                          If None, identity mapping is used (qubit i -> qubit i).
        
        Returns:
            A new CircuitDAG with both circuits composed.
        
        Example:
            # Create Bell pair preparation
            bell = CircuitDAG(2)
            bell.add_gate("h", [0])
            bell.add_gate("cx", [0, 1])
            
            # Create measurement circuit
            measure = CircuitDAG(2, 2)
            measure.add_measurement(0, 0)
            measure.add_measurement(1, 1)
            
            # Compose them
            full = bell.compose(measure)
        """
        # Determine qubit mapping
        if qubit_mapping is None:
            qubit_mapping = {i: i for i in range(other.num_qubits)}
        
        # Validate mapping
        max_target_qubit = max(qubit_mapping.values()) if qubit_mapping else 0
        new_num_qubits = max(self.num_qubits, max_target_qubit + 1)
        new_num_classical = max(self.num_classical_bits, other.num_classical_bits)
        
        # Create new circuit with combined size
        result = CircuitDAG(new_num_qubits, new_num_classical)
        
        # Copy gates from self
        for gate in self.get_gate_sequence():
            result.add_gate(
                name=gate.name,
                qubits=gate.qubits.copy(),
                params=gate.params.copy() if gate.params else None,
                classical_bits=gate.classical_bits.copy() if gate.classical_bits else None,
                condition=gate.condition.copy() if gate.condition else None
            )
        
        # Copy gates from other with qubit mapping
        for gate in other.get_gate_sequence():
            mapped_qubits = [qubit_mapping.get(q, q) for q in gate.qubits]
            result.add_gate(
                name=gate.name,
                qubits=mapped_qubits,
                params=gate.params.copy() if gate.params else None,
                classical_bits=gate.classical_bits.copy() if gate.classical_bits else None,
                condition=gate.condition.copy() if gate.condition else None
            )
        
        return result

    def inverse(self) -> "CircuitDAG":
        """
        Generate the inverse (adjoint/dagger) of this circuit.
        
        The inverse circuit:
        1. Reverses the order of all gates
        2. Applies the conjugate transpose to each gate
        
        For common gates:
        - H† = H (Hermitian)
        - X† = X, Y† = Y, Z† = Z (Hermitian)
        - S† = Sdg, T† = Tdg
        - Rx(θ)† = Rx(-θ), Ry(θ)† = Ry(-θ), Rz(θ)† = Rz(-θ)
        - CX† = CX, CZ† = CZ (Hermitian)
        - SWAP† = SWAP (Hermitian)
        
        Returns:
            A new CircuitDAG representing the inverse circuit.
        
        Note:
            Measurements and resets are NOT included in the inverse
            as they are not unitary operations.
        """
        # Gate inversion rules
        hermitian_gates = {
            "h", "x", "y", "z", "cx", "cnot", "cz", "swap", "ccx", "toffoli",
            "i", "id", "identity"
        }
        
        dagger_mapping = {
            "s": "sdg",
            "sdg": "s",
            "t": "tdg",
            "tdg": "t",
            "sx": "sxdg",
            "sxdg": "sx",
        }
        
        negated_param_gates = {"rx", "ry", "rz", "p", "phase", "u1", "crx", "cry", "crz", "cp"}
        
        # Create result circuit (no classical bits for unitary inverse)
        result = CircuitDAG(self.num_qubits, 0)
        
        # Get gates in reverse order (excluding measurements and resets)
        gates = self.get_gate_sequence()
        unitary_gates = [
            g for g in gates 
            if g.gate_type not in (GateType.MEASUREMENT, GateType.RESET, GateType.BARRIER)
        ]
        
        for gate in reversed(unitary_gates):
            gate_name = gate.name.lower()
            new_params = gate.params.copy() if gate.params else []
            
            # Determine inverse gate name
            if gate_name in hermitian_gates:
                new_name = gate.name  # Same gate
            elif gate_name in dagger_mapping:
                new_name = dagger_mapping[gate_name]
            elif gate_name in negated_param_gates:
                new_name = gate.name
                new_params = [-p for p in gate.params]  # Negate parameters
            elif gate_name == "u3":
                # U3(θ, φ, λ)† = U3(-θ, -λ, -φ)
                new_name = "u3"
                if len(gate.params) == 3:
                    new_params = [-gate.params[0], -gate.params[2], -gate.params[1]]
            elif gate_name == "u2":
                # U2(φ, λ)† = U2(-λ-π, -φ+π)
                new_name = "u2"
                if len(gate.params) == 2:
                    import math
                    new_params = [-gate.params[1] - math.pi, -gate.params[0] + math.pi]
            else:
                # For unknown gates, just reverse (assume Hermitian)
                new_name = gate.name
            
            result.add_gate(
                name=new_name,
                qubits=gate.qubits.copy(),
                params=new_params if new_params else None
            )
        
        return result

    def tensor(self, other: "CircuitDAG") -> "CircuitDAG":
        """
        Tensor product of this circuit with another (parallel composition).
        
        The other circuit's qubits are placed after this circuit's qubits.
        
        Args:
            other: The circuit to tensor with
        
        Returns:
            A new CircuitDAG with both circuits in parallel.
        
        Example:
            # Two separate single-qubit circuits
            circuit1 = CircuitDAG(1)
            circuit1.add_gate("h", [0])
            
            circuit2 = CircuitDAG(1)
            circuit2.add_gate("x", [0])
            
            # Tensor them: H⊗X on 2 qubits
            combined = circuit1.tensor(circuit2)
            # Result: H on qubit 0, X on qubit 1
        """
        new_num_qubits = self.num_qubits + other.num_qubits
        new_num_classical = self.num_classical_bits + other.num_classical_bits
        
        result = CircuitDAG(new_num_qubits, new_num_classical)
        
        # Copy gates from self (qubits unchanged)
        for gate in self.get_gate_sequence():
            result.add_gate(
                name=gate.name,
                qubits=gate.qubits.copy(),
                params=gate.params.copy() if gate.params else None,
                classical_bits=gate.classical_bits.copy() if gate.classical_bits else None,
                condition=gate.condition.copy() if gate.condition else None
            )
        
        # Copy gates from other (qubits shifted by self.num_qubits)
        qubit_offset = self.num_qubits
        clbit_offset = self.num_classical_bits
        
        for gate in other.get_gate_sequence():
            shifted_qubits = [q + qubit_offset for q in gate.qubits]
            shifted_clbits = [c + clbit_offset for c in gate.classical_bits] if gate.classical_bits else None
            
            result.add_gate(
                name=gate.name,
                qubits=shifted_qubits,
                params=gate.params.copy() if gate.params else None,
                classical_bits=shifted_clbits,
                condition=gate.condition.copy() if gate.condition else None
            )
        
        return result

    def repeat(self, n: int) -> "CircuitDAG":
        """
        Repeat this circuit n times (sequential composition with itself).
        
        Args:
            n: Number of repetitions
        
        Returns:
            A new CircuitDAG with the circuit repeated n times.
        """
        if n <= 0:
            return CircuitDAG(self.num_qubits, self.num_classical_bits)
        
        result = CircuitDAG(self.num_qubits, self.num_classical_bits)
        
        for _ in range(n):
            for gate in self.get_gate_sequence():
                if gate.gate_type not in (GateType.MEASUREMENT,):  # Don't repeat measurements
                    result.add_gate(
                        name=gate.name,
                        qubits=gate.qubits.copy(),
                        params=gate.params.copy() if gate.params else None,
                        classical_bits=gate.classical_bits.copy() if gate.classical_bits else None,
                        condition=gate.condition.copy() if gate.condition else None
                    )
        
        return result

    def copy(self) -> "CircuitDAG":
        """Create a deep copy of this circuit."""
        result = CircuitDAG(self.num_qubits, self.num_classical_bits)
        
        for gate in self.get_gate_sequence():
            result.add_gate(
                name=gate.name,
                qubits=gate.qubits.copy(),
                params=gate.params.copy() if gate.params else None,
                classical_bits=gate.classical_bits.copy() if gate.classical_bits else None,
                condition=gate.condition.copy() if gate.condition else None
            )
        
        return result
