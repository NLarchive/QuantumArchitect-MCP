"""
Connectivity Validator - Checks if 2-qubit gates are adjacent on specific hardware topologies.
Crucial for real hardware execution.
"""

from typing import Any
import json
import os


# Default hardware topologies (coupling maps)
DEFAULT_TOPOLOGIES: dict[str, dict[str, Any]] = {
    "ibm_brisbane": {
        "name": "IBM Brisbane",
        "num_qubits": 127,
        "native_gates": ["cx", "id", "rz", "sx", "x"],
        "coupling_map": [
            [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
            [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5],
            [6, 7], [7, 6], [7, 8], [8, 7], [8, 9], [9, 8],
            # Heavy-hex lattice structure (simplified for first 20 qubits)
            [0, 14], [14, 0], [4, 15], [15, 4], [8, 16], [16, 8],
            [12, 17], [17, 12], [1, 18], [18, 1], [5, 19], [19, 5],
        ],
        "description": "127-qubit Eagle processor",
    },
    "ibm_sherbrooke": {
        "name": "IBM Sherbrooke",
        "num_qubits": 127,
        "native_gates": ["cx", "id", "rz", "sx", "x"],
        "coupling_map": [
            [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
            [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5],
        ],
        "description": "127-qubit Eagle processor",
    },
    "rigetti_aspen": {
        "name": "Rigetti Aspen-M",
        "num_qubits": 80,
        "native_gates": ["cz", "rx", "rz"],
        "coupling_map": [
            # Octagonal lattice (simplified)
            [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
            [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5],
            [6, 7], [7, 6], [7, 0], [0, 7],  # Ring
            [0, 8], [8, 0], [2, 9], [9, 2], [4, 10], [10, 4], [6, 11], [11, 6],
        ],
        "description": "80-qubit Aspen processor",
    },
    "google_sycamore": {
        "name": "Google Sycamore",
        "num_qubits": 53,
        "native_gates": ["fsim", "phxz", "syc"],
        "coupling_map": [
            # 2D grid with nearest-neighbor coupling
            [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2],
            [3, 4], [4, 3], [4, 5], [5, 4], [5, 6], [6, 5],
            [0, 7], [7, 0], [1, 8], [8, 1], [2, 9], [9, 2],
            [3, 10], [10, 3], [4, 11], [11, 4], [5, 12], [12, 5],
        ],
        "description": "53-qubit Sycamore processor",
    },
    "ionq_harmony": {
        "name": "IonQ Harmony",
        "num_qubits": 11,
        "native_gates": ["gpi", "gpi2", "ms"],
        "coupling_map": "all_to_all",  # Ion traps have full connectivity
        "description": "11-qubit ion trap processor (full connectivity)",
    },
    "quantinuum_h1": {
        "name": "Quantinuum H1",
        "num_qubits": 20,
        "native_gates": ["rz", "ry", "zz"],
        "coupling_map": "all_to_all",
        "description": "20-qubit ion trap processor (full connectivity)",
    },
    "linear_5": {
        "name": "Linear 5-qubit",
        "num_qubits": 5,
        "native_gates": ["cx", "id", "rz", "sx", "x"],
        "coupling_map": [
            [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]
        ],
        "description": "5-qubit linear chain topology",
    },
}


def validate_connectivity(
    circuit_data: dict[str, Any],
    hardware_name: str | None = None,
    custom_coupling_map: list[list[int]] | None = None
) -> dict[str, Any]:
    """
    Validate that all 2-qubit gates respect hardware connectivity.
    
    Args:
        circuit_data: Circuit dictionary
        hardware_name: Name of hardware profile to use
        custom_coupling_map: Custom coupling map as list of [control, target] pairs
    
    Returns:
        Validation result with specific error locations
    """
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    
    # Get coupling map
    if custom_coupling_map:
        coupling_map = set(tuple(pair) for pair in custom_coupling_map)
        hardware_info = {"name": "Custom", "num_qubits": max(max(p) for p in custom_coupling_map) + 1}
    elif hardware_name:
        if hardware_name.lower() not in {k.lower() for k in DEFAULT_TOPOLOGIES}:
            available = list(DEFAULT_TOPOLOGIES.keys())
            return {
                "valid": False,
                "errors": [{"message": f"Unknown hardware: {hardware_name}", "available": available}],
                "warnings": [],
            }
        
        # Case-insensitive lookup
        hardware_key = next(k for k in DEFAULT_TOPOLOGIES if k.lower() == hardware_name.lower())
        hardware_info = DEFAULT_TOPOLOGIES[hardware_key]
        
        if hardware_info.get("coupling_map") == "all_to_all":
            # Full connectivity - all pairs valid
            coupling_map = None
        else:
            coupling_map = set(tuple(pair) for pair in hardware_info["coupling_map"])
    else:
        # No hardware specified - check for obvious issues only
        coupling_map = None
        hardware_info = {"name": "No specific hardware", "num_qubits": 1000}
    
    gates = circuit_data.get("gates", [])
    num_qubits = circuit_data.get("num_qubits", 0)
    
    # Check qubit count against hardware
    if num_qubits > hardware_info.get("num_qubits", 1000):
        errors.append({
            "gate_idx": -1,
            "message": f"Circuit uses {num_qubits} qubits but {hardware_info['name']} has only {hardware_info['num_qubits']}",
            "suggestion": "Reduce circuit size or use different hardware"
        })
    
    # Check each 2-qubit gate
    swap_suggestions: list[dict[str, Any]] = []
    
    for idx, gate in enumerate(gates):
        name = gate.get("name", "").lower()
        qubits = gate.get("qubits", [])
        
        if len(qubits) == 2:
            q0, q1 = qubits
            
            if coupling_map is not None:
                # Check if this edge exists
                if (q0, q1) not in coupling_map and (q1, q0) not in coupling_map:
                    errors.append({
                        "gate_idx": idx,
                        "gate": name,
                        "qubits": [q0, q1],
                        "message": f"Qubits {q0} and {q1} are not connected on {hardware_info['name']}",
                        "suggestion": f"Insert SWAP gates to route this connection"
                    })
                    
                    # Try to find a path
                    path = _find_qubit_path(q0, q1, coupling_map, num_qubits)
                    if path:
                        swap_suggestions.append({
                            "original_gate_idx": idx,
                            "path": path,
                            "swaps_needed": len(path) - 2
                        })
        
        elif len(qubits) == 3:
            # Three-qubit gates need all pairs connected or decomposition
            q0, q1, q2 = qubits
            if coupling_map is not None:
                missing_pairs = []
                for pair in [(q0, q1), (q1, q2), (q0, q2)]:
                    if pair not in coupling_map and (pair[1], pair[0]) not in coupling_map:
                        missing_pairs.append(pair)
                
                if missing_pairs:
                    warnings.append(
                        f"Gate {idx} ({name}): 3-qubit gate may need decomposition. "
                        f"Missing connections: {missing_pairs}"
                    )
    
    is_valid = len(errors) == 0
    
    result: dict[str, Any] = {
        "valid": is_valid,
        "hardware": hardware_info.get("name", "Unknown"),
        "hardware_qubits": hardware_info.get("num_qubits", 0),
        "circuit_qubits": num_qubits,
        "errors": errors,
        "warnings": warnings,
        "connectivity_type": "all_to_all" if coupling_map is None else "restricted",
    }
    
    if swap_suggestions:
        result["swap_suggestions"] = swap_suggestions
    
    if is_valid:
        result["summary"] = f"Circuit is compatible with {hardware_info['name']}"
    else:
        result["summary"] = f"Found {len(errors)} connectivity violation(s)"
    
    return result


def _find_qubit_path(
    start: int,
    end: int,
    coupling_map: set[tuple[int, int]],
    max_qubits: int
) -> list[int] | None:
    """Find shortest path between two qubits using BFS."""
    if not coupling_map:
        return None
    
    # Build adjacency list
    adj: dict[int, list[int]] = {}
    for q0, q1 in coupling_map:
        if q0 not in adj:
            adj[q0] = []
        adj[q0].append(q1)
    
    # BFS
    visited = {start}
    queue = [(start, [start])]
    
    while queue:
        current, path = queue.pop(0)
        
        if current == end:
            return path
        
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


def get_available_hardware() -> list[dict[str, Any]]:
    """Get list of available hardware profiles."""
    return [
        {
            "id": key,
            "name": info["name"],
            "num_qubits": info["num_qubits"],
            "native_gates": info["native_gates"],
            "connectivity": "all_to_all" if info.get("coupling_map") == "all_to_all" else "restricted",
            "description": info.get("description", ""),
        }
        for key, info in DEFAULT_TOPOLOGIES.items()
    ]


def check_native_gates(
    circuit_data: dict[str, Any],
    hardware_name: str
) -> dict[str, Any]:
    """
    Check if circuit uses only native gates for the specified hardware.
    
    Args:
        circuit_data: Circuit dictionary
        hardware_name: Hardware profile name
    
    Returns:
        Validation result with non-native gates listed
    """
    hardware_key = next(
        (k for k in DEFAULT_TOPOLOGIES if k.lower() == hardware_name.lower()),
        None
    )
    
    if not hardware_key:
        return {
            "valid": False,
            "error": f"Unknown hardware: {hardware_name}",
            "available": list(DEFAULT_TOPOLOGIES.keys()),
        }
    
    hardware_info = DEFAULT_TOPOLOGIES[hardware_key]
    native_gates = set(g.lower() for g in hardware_info["native_gates"])
    
    # Common gate translations
    gate_aliases = {
        "cnot": "cx",
        "toffoli": "ccx",
        "fredkin": "cswap",
    }
    
    non_native: list[dict[str, Any]] = []
    gates = circuit_data.get("gates", [])
    
    for idx, gate in enumerate(gates):
        name = gate.get("name", "").lower()
        canonical_name = gate_aliases.get(name, name)
        
        if name in ("barrier", "measure", "reset"):
            continue
        
        if canonical_name not in native_gates:
            non_native.append({
                "gate_idx": idx,
                "gate": name,
                "qubits": gate.get("qubits", []),
            })
    
    return {
        "valid": len(non_native) == 0,
        "hardware": hardware_info["name"],
        "native_gates": list(native_gates),
        "non_native_gates": non_native,
        "summary": f"Found {len(non_native)} non-native gate(s)" if non_native else "All gates are native",
        "suggestion": "Use transpilation to decompose non-native gates" if non_native else None,
    }
