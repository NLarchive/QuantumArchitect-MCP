"""
Beginner Templates - Pre-built quantum circuits for learning.
Bell States, GHZ States, and other fundamental circuits.
"""

from typing import Any
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.qasm2 import dumps as qasm2_dumps
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


def _circuit_to_qasm(qc) -> str:
    """Convert Qiskit circuit to QASM string (Qiskit 2.x compatible)."""
    try:
        return qasm2_dumps(qc)
    except Exception:
        # Fallback to manual generation
        return ""


def create_bell_state(variant: int = 0) -> dict[str, Any]:
    """
    Create a Bell state circuit.
    
    Args:
        variant: Which Bell state to create (0-3)
            0: |Φ+⟩ = (|00⟩ + |11⟩)/√2
            1: |Φ-⟩ = (|00⟩ - |11⟩)/√2
            2: |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            3: |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    
    Returns:
        Dictionary with circuit data and QASM
    """
    # Build gates list - works with or without Qiskit
    gates = [{"name": "h", "qubits": [0], "params": []}]
    if variant in (1, 3):
        gates.append({"name": "z", "qubits": [0], "params": []})
    if variant in (2, 3):
        gates.append({"name": "x", "qubits": [0], "params": []})
    gates.append({"name": "cx", "qubits": [0, 1], "params": []})
    
    bell_names = ["Φ+", "Φ-", "Ψ+", "Ψ-"]
    qasm = _generate_bell_qasm(variant)
    
    result = {
        "name": f"Bell State |{bell_names[variant]}⟩",
        "num_qubits": 2,
        "num_classical_bits": 2,
        "gates": gates,
        "description": f"Creates the Bell state |{bell_names[variant]}⟩ = maximally entangled 2-qubit state",
        "qasm": qasm,
        "depth": len(gates),  # Simple depth estimate
        "gate_count": len(gates),
    }
    
    # Add Qiskit circuit diagram if available
    if QISKIT_AVAILABLE:
        try:
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            if variant in (1, 3):
                qc.z(0)
            if variant in (2, 3):
                qc.x(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            result["circuit_diagram"] = str(qc.draw(output='text'))
            result["qasm"] = _circuit_to_qasm(qc)
            result["depth"] = qc.depth()
            result["gate_count"] = qc.size()
        except Exception:
            pass
    
    return result


def _generate_bell_qasm(variant: int) -> str:
    """Generate QASM for Bell state without Qiskit."""
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
"""
    if variant in (1, 3):
        qasm += "z q[0];\n"
    if variant in (2, 3):
        qasm += "x q[0];\n"
    qasm += """cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
"""
    return qasm


def create_ghz_state(num_qubits: int = 3) -> dict[str, Any]:
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state circuit.
    
    GHZ state: (|00...0⟩ + |11...1⟩)/√2
    
    Args:
        num_qubits: Number of qubits (minimum 2)
    
    Returns:
        Dictionary with circuit data and QASM
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits")
    
    # Build gates list
    gates = [{"name": "h", "qubits": [0], "params": []}]
    for i in range(num_qubits - 1):
        gates.append({"name": "cx", "qubits": [i, i + 1], "params": []})
    
    qasm = _generate_ghz_qasm(num_qubits)
    
    result = {
        "name": f"{num_qubits}-qubit GHZ State",
        "num_qubits": num_qubits,
        "num_classical_bits": num_qubits,
        "gates": gates,
        "qasm": qasm,
        "description": f"GHZ state: (|{'0'*num_qubits}⟩ + |{'1'*num_qubits}⟩)/√2",
        "depth": num_qubits,  # H + n-1 CX gates in sequence
        "gate_count": num_qubits,
    }
    
    if QISKIT_AVAILABLE:
        try:
            qc = QuantumCircuit(num_qubits, num_qubits)
            qc.h(0)
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
            qc.measure(range(num_qubits), range(num_qubits))
            result["circuit_diagram"] = str(qc.draw(output='text'))
            result["qasm"] = _circuit_to_qasm(qc)
            result["depth"] = qc.depth()
            result["gate_count"] = qc.size()
        except Exception:
            pass
    
    return result


def _generate_ghz_qasm(num_qubits: int) -> str:
    """Generate QASM for GHZ state without Qiskit."""
    qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
h q[0];
"""
    for i in range(num_qubits - 1):
        qasm += f"cx q[{i}],q[{i+1}];\n"
    for i in range(num_qubits):
        qasm += f"measure q[{i}] -> c[{i}];\n"
    return qasm


def create_superposition(num_qubits: int = 1) -> dict[str, Any]:
    """
    Create a uniform superposition state using Hadamard gates.
    
    Creates the state: |+⟩^⊗n = (1/√2^n) Σ|x⟩
    
    Args:
        num_qubits: Number of qubits
    
    Returns:
        Dictionary with circuit data
    """
    if num_qubits < 1:
        raise ValueError("Need at least 1 qubit")
    
    if not QISKIT_AVAILABLE:
        gates = [{"name": "h", "qubits": [i]} for i in range(num_qubits)]
        return {
            "name": f"{num_qubits}-qubit Uniform Superposition",
            "num_qubits": num_qubits,
            "num_classical_bits": num_qubits,
            "gates": gates,
            "description": f"Uniform superposition over {2**num_qubits} basis states",
        }
    
    qc = QuantumCircuit(num_qubits, num_qubits)
    for i in range(num_qubits):
        qc.h(i)
    qc.measure(range(num_qubits), range(num_qubits))
    
    return {
        "name": f"{num_qubits}-qubit Uniform Superposition",
        "num_qubits": num_qubits,
        "num_classical_bits": num_qubits,
        "depth": qc.depth(),
        "gate_count": qc.size(),
        "qasm": _circuit_to_qasm(qc),
        "circuit_diagram": str(qc.draw(output='text')),
        "description": f"Uniform superposition over {2**num_qubits} basis states",
    }


def create_w_state(num_qubits: int = 3) -> dict[str, Any]:
    """
    Create a W state circuit.
    
    W state: (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n
    
    Args:
        num_qubits: Number of qubits (minimum 3)
    
    Returns:
        Dictionary with circuit data
    """
    if num_qubits < 3:
        raise ValueError("W state requires at least 3 qubits")
    
    if not QISKIT_AVAILABLE:
        # Simplified W state construction
        import math
        gates = []
        # First rotation
        theta = 2 * math.acos(1 / math.sqrt(num_qubits))
        gates.append({"name": "ry", "qubits": [0], "params": [theta]})
        
        for i in range(1, num_qubits):
            theta = 2 * math.acos(1 / math.sqrt(num_qubits - i))
            gates.append({"name": "cx", "qubits": [i-1, i]})
            if i < num_qubits - 1:
                gates.append({"name": "ry", "qubits": [i], "params": [theta]})
        
        return {
            "name": f"{num_qubits}-qubit W State",
            "num_qubits": num_qubits,
            "num_classical_bits": num_qubits,
            "gates": gates,
            "description": f"W state with {num_qubits} qubits - balanced superposition of single-excitation states",
        }
    
    import numpy as np
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # W state preparation using rotations
    theta = 2 * np.arccos(1 / np.sqrt(num_qubits))
    qc.ry(theta, 0)
    
    for i in range(1, num_qubits):
        theta = 2 * np.arccos(1 / np.sqrt(num_qubits - i))
        qc.cx(i - 1, i)
        if i < num_qubits - 1:
            qc.cry(theta, i, i - 1)
    
    qc.measure(range(num_qubits), range(num_qubits))
    
    return {
        "name": f"{num_qubits}-qubit W State",
        "num_qubits": num_qubits,
        "num_classical_bits": num_qubits,
        "depth": qc.depth(),
        "gate_count": qc.size(),
        "qasm": _circuit_to_qasm(qc),
        "circuit_diagram": str(qc.draw(output='text')),
        "description": f"W state with {num_qubits} qubits",
    }


# =============================================================================
# CLASS INTERFACE FOR MCP ENDPOINTS
# =============================================================================

class BeginnerTemplates:
    """
    Class interface for beginner quantum circuit templates.
    Wraps the standalone functions for MCP endpoint compatibility.
    """
    
    @staticmethod
    def bell_state(variant: int = 0) -> dict[str, Any]:
        """Create a Bell state circuit."""
        return create_bell_state(variant)
    
    @staticmethod
    def ghz_state(num_qubits: int = 3) -> dict[str, Any]:
        """Create a GHZ state circuit."""
        return create_ghz_state(num_qubits)
    
    @staticmethod
    def w_state(num_qubits: int = 3) -> dict[str, Any]:
        """Create a W state circuit."""
        return create_w_state(num_qubits)
    
    @staticmethod
    def uniform_superposition(num_qubits: int = 1) -> dict[str, Any]:
        """Create a uniform superposition circuit."""
        return create_superposition(num_qubits)
