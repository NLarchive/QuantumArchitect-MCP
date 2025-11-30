"""
Context Provider - Functions that provide Resources to the Agent.
Loads documentation on gates, hardware topologies, and reference circuits.
"""

from typing import Any
import json
import os
from pathlib import Path


# Base path for data files
DATA_DIR = Path(__file__).parent.parent / "data"


def get_hardware_profile(hardware_name: str) -> dict[str, Any] | None:
    """
    Load a hardware profile by name.
    
    Args:
        hardware_name: Name of the hardware profile
    
    Returns:
        Hardware profile dictionary or None if not found
    """
    profiles_dir = DATA_DIR / "hardware_profiles"
    
    # Try exact match first
    profile_path = profiles_dir / f"{hardware_name}.json"
    if profile_path.exists():
        with open(profile_path, "r") as f:
            return json.load(f)
    
    # Try case-insensitive match
    for file_path in profiles_dir.glob("*.json"):
        if file_path.stem.lower() == hardware_name.lower():
            with open(file_path, "r") as f:
                return json.load(f)
    
    return None


def list_hardware_profiles() -> list[dict[str, Any]]:
    """
    List all available hardware profiles.
    
    Returns:
        List of hardware profile summaries
    """
    profiles_dir = DATA_DIR / "hardware_profiles"
    profiles = []
    
    if profiles_dir.exists():
        for file_path in profiles_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    profiles.append({
                        "id": file_path.stem,
                        "name": data.get("display_name", data.get("name", file_path.stem)),
                        "num_qubits": data.get("num_qubits", 0),
                        "provider": data.get("provider", "Unknown"),
                    })
            except Exception:
                pass
    
    return profiles


def get_reference_circuit(circuit_name: str) -> dict[str, Any] | None:
    """
    Load a reference circuit by name.
    
    Args:
        circuit_name: Name of the reference circuit
    
    Returns:
        Reference circuit dictionary or None if not found
    """
    circuits_dir = DATA_DIR / "reference_circuits"
    
    circuit_path = circuits_dir / f"{circuit_name}.json"
    if circuit_path.exists():
        with open(circuit_path, "r") as f:
            return json.load(f)
    
    return None


def list_reference_circuits() -> list[dict[str, Any]]:
    """
    List all available reference circuits.
    
    Returns:
        List of reference circuit summaries
    """
    circuits_dir = DATA_DIR / "reference_circuits"
    circuits = []
    
    if circuits_dir.exists():
        for file_path in circuits_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    circuits.append({
                        "id": file_path.stem,
                        "name": data.get("name", file_path.stem),
                        "description": data.get("description", ""),
                        "num_qubits": data.get("num_qubits", 0),
                        "algorithm_type": data.get("algorithm_type", "unknown"),
                    })
            except Exception:
                pass
    
    return circuits


def get_gate_documentation(gate_name: str) -> dict[str, Any]:
    """
    Get documentation for a quantum gate.
    
    Args:
        gate_name: Name of the gate
    
    Returns:
        Gate documentation
    """
    gate_docs = {
        "h": {
            "name": "Hadamard",
            "symbol": "H",
            "qubits": 1,
            "parameters": 0,
            "matrix": "[[1,1],[1,-1]]/√2",
            "description": "Creates superposition from basis states",
            "bloch_action": "Rotation of π around the (X+Z)/√2 axis",
            "use_cases": ["Creating superposition", "Quantum Fourier Transform"],
        },
        "x": {
            "name": "Pauli-X",
            "symbol": "X",
            "qubits": 1,
            "parameters": 0,
            "matrix": "[[0,1],[1,0]]",
            "description": "Bit flip gate (quantum NOT)",
            "bloch_action": "Rotation of π around the X axis",
            "use_cases": ["State preparation", "Error correction"],
        },
        "y": {
            "name": "Pauli-Y",
            "symbol": "Y",
            "qubits": 1,
            "parameters": 0,
            "matrix": "[[0,-i],[i,0]]",
            "description": "Combined bit and phase flip",
            "bloch_action": "Rotation of π around the Y axis",
        },
        "z": {
            "name": "Pauli-Z",
            "symbol": "Z",
            "qubits": 1,
            "parameters": 0,
            "matrix": "[[1,0],[0,-1]]",
            "description": "Phase flip gate",
            "bloch_action": "Rotation of π around the Z axis",
        },
        "s": {
            "name": "S Gate",
            "symbol": "S",
            "qubits": 1,
            "parameters": 0,
            "matrix": "[[1,0],[0,i]]",
            "description": "√Z gate (phase gate with π/2 rotation)",
            "bloch_action": "Rotation of π/2 around the Z axis",
        },
        "t": {
            "name": "T Gate",
            "symbol": "T",
            "qubits": 1,
            "parameters": 0,
            "matrix": "[[1,0],[0,e^(iπ/4)]]",
            "description": "π/8 gate, crucial for universal quantum computation",
            "bloch_action": "Rotation of π/4 around the Z axis",
            "use_cases": ["Fault-tolerant computing", "Magic state distillation"],
        },
        "rx": {
            "name": "Rotation X",
            "symbol": "Rx(θ)",
            "qubits": 1,
            "parameters": 1,
            "matrix": "[[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]",
            "description": "Rotation around X axis by angle θ",
            "use_cases": ["Variational circuits", "State preparation"],
        },
        "ry": {
            "name": "Rotation Y",
            "symbol": "Ry(θ)",
            "qubits": 1,
            "parameters": 1,
            "matrix": "[[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]",
            "description": "Rotation around Y axis by angle θ",
            "use_cases": ["Variational circuits", "Amplitude encoding"],
        },
        "rz": {
            "name": "Rotation Z",
            "symbol": "Rz(θ)",
            "qubits": 1,
            "parameters": 1,
            "matrix": "[[e^(-iθ/2), 0], [0, e^(iθ/2)]]",
            "description": "Rotation around Z axis by angle θ",
            "use_cases": ["Phase control", "Native gate on most hardware"],
        },
        "cx": {
            "name": "Controlled-X (CNOT)",
            "symbol": "CX",
            "qubits": 2,
            "parameters": 0,
            "description": "Flips target qubit if control qubit is |1⟩",
            "use_cases": ["Entanglement creation", "Universal gate set"],
            "connectivity": "Requires connected qubits on hardware",
        },
        "cz": {
            "name": "Controlled-Z",
            "symbol": "CZ",
            "qubits": 2,
            "parameters": 0,
            "description": "Applies Z to target if control is |1⟩",
            "use_cases": ["Entanglement", "Native gate on some hardware"],
        },
        "swap": {
            "name": "SWAP",
            "symbol": "SWAP",
            "qubits": 2,
            "parameters": 0,
            "description": "Exchanges states of two qubits",
            "decomposition": "3 CNOT gates",
        },
        "ccx": {
            "name": "Toffoli (CCX)",
            "symbol": "CCX",
            "qubits": 3,
            "parameters": 0,
            "description": "Controlled-controlled-X, flips target if both controls are |1⟩",
            "use_cases": ["Classical logic in quantum", "Error correction"],
            "decomposition": "~6-15 two-qubit gates",
        },
    }
    
    gate_key = gate_name.lower()
    if gate_key in gate_docs:
        return gate_docs[gate_key]
    
    # Aliases
    aliases = {
        "cnot": "cx",
        "toffoli": "ccx",
        "hadamard": "h",
    }
    if gate_key in aliases:
        return gate_docs[aliases[gate_key]]
    
    return {
        "name": gate_name,
        "error": "Documentation not found",
        "suggestion": f"Available gates: {', '.join(gate_docs.keys())}",
    }


def get_algorithm_explanation(algorithm_name: str) -> dict[str, Any]:
    """
    Get explanation of a quantum algorithm.
    
    Args:
        algorithm_name: Name of the algorithm
    
    Returns:
        Algorithm explanation
    """
    algorithms = {
        "bell_state": {
            "name": "Bell State",
            "category": "Entanglement",
            "level": "Beginner",
            "description": "Creates maximally entangled 2-qubit states",
            "steps": [
                "Apply Hadamard to first qubit (superposition)",
                "Apply CNOT with first as control (entanglement)",
            ],
            "output": "Correlated measurements: 00 or 11 with equal probability",
            "applications": ["Quantum teleportation", "Superdense coding", "Bell tests"],
        },
        "ghz_state": {
            "name": "GHZ State",
            "category": "Entanglement",
            "level": "Beginner",
            "description": "Multi-qubit generalization of Bell state",
            "formula": "(|00...0⟩ + |11...1⟩)/√2",
            "applications": ["Quantum sensing", "Secret sharing"],
        },
        "qft": {
            "name": "Quantum Fourier Transform",
            "category": "Algorithm Building Block",
            "level": "Intermediate",
            "description": "Quantum analog of discrete Fourier transform",
            "complexity": "O(n²) gates vs O(n·2ⁿ) classical",
            "applications": ["Shor's algorithm", "Phase estimation", "Period finding"],
        },
        "grover": {
            "name": "Grover's Search",
            "category": "Search Algorithm",
            "level": "Intermediate",
            "description": "Searches unstructured database with quadratic speedup",
            "speedup": "O(√N) vs O(N) classical",
            "steps": [
                "Initialize uniform superposition",
                "Repeat √N times: Oracle + Diffusion",
                "Measure",
            ],
            "applications": ["Database search", "Constraint satisfaction", "Optimization"],
        },
        "vqe": {
            "name": "Variational Quantum Eigensolver",
            "category": "Variational Algorithm",
            "level": "Advanced",
            "description": "Finds ground state energy of Hamiltonians",
            "approach": "Hybrid quantum-classical optimization",
            "applications": ["Quantum chemistry", "Materials science"],
        },
        "qaoa": {
            "name": "Quantum Approximate Optimization Algorithm",
            "category": "Variational Algorithm",
            "level": "Advanced",
            "description": "Solves combinatorial optimization problems",
            "approach": "Alternating cost and mixer Hamiltonians",
            "applications": ["Max-Cut", "Traveling salesman", "Portfolio optimization"],
        },
    }
    
    key = algorithm_name.lower().replace(" ", "_").replace("-", "_")
    if key in algorithms:
        return algorithms[key]
    
    return {
        "name": algorithm_name,
        "error": "Algorithm not found",
        "available": list(algorithms.keys()),
    }


def get_learning_resources(topic: str) -> dict[str, Any]:
    """
    Get learning resources for a quantum computing topic.
    
    Args:
        topic: Topic name
    
    Returns:
        Learning resources and recommendations
    """
    resources = {
        "beginner": {
            "prerequisites": ["Linear algebra basics", "Complex numbers"],
            "concepts": ["Qubits", "Superposition", "Measurement", "Basic gates"],
            "recommended_tools": ["IBM Quantum Composer", "Quirk"],
            "circuits_to_build": ["Bell state", "Superposition", "Simple measurement"],
            "roi_score": "10/10 - Foundation for everything",
        },
        "intermediate": {
            "prerequisites": ["Basic quantum gates", "Circuit diagrams"],
            "concepts": ["Entanglement", "Quantum algorithms", "Noise basics"],
            "recommended_tools": ["Qiskit", "PennyLane"],
            "circuits_to_build": ["Grover's search", "Deutsch-Jozsa", "Teleportation"],
            "roi_score": "9/10 - Essential for practical work",
        },
        "advanced": {
            "prerequisites": ["Framework experience", "Algorithm design"],
            "concepts": ["VQE", "QAOA", "Noise mitigation", "Hardware constraints"],
            "recommended_tools": ["Qiskit Runtime", "AWS Braket"],
            "circuits_to_build": ["VQE ansatz", "QAOA for Max-Cut"],
            "roi_score": "10/10 - This is where jobs are",
        },
        "phd": {
            "prerequisites": ["Advanced quantum mechanics", "Complexity theory"],
            "concepts": ["Error correction", "Fault tolerance", "Novel algorithms"],
            "recommended_tools": ["Stim", "Qiskit Pulse", "Custom simulators"],
            "circuits_to_build": ["Surface code", "Custom error correction"],
            "roi_score": "Variable - Research frontier",
        },
    }
    
    key = topic.lower()
    if key in resources:
        return resources[key]
    
    return {
        "error": f"Topic '{topic}' not found",
        "available_topics": list(resources.keys()),
    }
