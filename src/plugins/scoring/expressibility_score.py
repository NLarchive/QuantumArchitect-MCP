"""
Expressibility Score - Evaluates how well variational circuits can cover the Hilbert space.
Research-level metric for Quantum Machine Learning (QML) applications.
"""

from typing import Any
import numpy as np
import math


def score_expressibility(
    circuit_data: dict[str, Any],
    num_samples: int = 1000
) -> dict[str, Any]:
    """
    Score the expressibility of a variational quantum circuit.
    
    Expressibility measures how uniformly a parameterized circuit can sample
    from the space of quantum states. Higher expressibility = better for QML.
    
    Based on: "Expressibility and Entangling Capability of Parameterized 
    Quantum Circuits for Hybrid Quantum-Classical Algorithms"
    
    Args:
        circuit_data: Circuit dictionary (should be parameterized)
        num_samples: Number of random parameter samples for estimation
    
    Returns:
        Expressibility analysis
    """
    gates = circuit_data.get("gates", [])
    num_qubits = circuit_data.get("num_qubits", 0)
    parameters = circuit_data.get("parameters", [])
    
    # Analyze circuit structure
    structure_analysis = _analyze_ansatz_structure(gates, num_qubits)
    
    # Calculate theoretical expressibility bounds
    num_params = len(parameters) if parameters else structure_analysis["estimated_parameters"]
    
    # Expressibility capacity
    # Maximum expressibility requires 2^(2n) - 1 parameters for n qubits
    # (full SU(2^n) coverage)
    max_params_needed = 4 ** num_qubits - 1
    param_sufficiency = min(1.0, num_params / max_params_needed) if max_params_needed > 0 else 0
    
    # Entangling capability analysis
    entangling_score = _calculate_entangling_capability(gates, num_qubits)
    
    # Estimate expressibility score (0-1)
    # Based on: parameter count, entangling gates, layer structure
    expressibility_score = _estimate_expressibility(
        num_params,
        num_qubits,
        structure_analysis,
        entangling_score
    )
    
    # Meyer-Wallach entanglement measure estimate
    mw_measure = _estimate_meyer_wallach(structure_analysis, num_qubits)
    
    # Classification
    if expressibility_score > 0.8:
        classification = "Highly Expressive"
        recommendation = "Suitable for complex learning tasks"
    elif expressibility_score > 0.5:
        classification = "Moderately Expressive"
        recommendation = "Good for moderate complexity problems"
    elif expressibility_score > 0.2:
        classification = "Low Expressibility"
        recommendation = "May struggle with complex patterns"
    else:
        classification = "Very Low Expressibility"
        recommendation = "Consider adding more parameters or entangling gates"
    
    return {
        "expressibility_score": round(expressibility_score, 4),
        "classification": classification,
        "recommendation": recommendation,
        "analysis": {
            "num_parameters": num_params,
            "max_parameters_for_full_coverage": max_params_needed,
            "parameter_sufficiency": round(param_sufficiency, 4),
            "entangling_capability": round(entangling_score, 4),
            "estimated_meyer_wallach": round(mw_measure, 4),
        },
        "structure": {
            "num_layers": structure_analysis["num_layers"],
            "entangling_gates_per_layer": structure_analysis["entangling_per_layer"],
            "rotation_gates_per_layer": structure_analysis["rotations_per_layer"],
            "entanglement_pattern": structure_analysis["entanglement_pattern"],
        },
        "qml_suitability": {
            "classification_tasks": "Good" if expressibility_score > 0.3 else "Limited",
            "regression_tasks": "Good" if expressibility_score > 0.4 else "Limited",
            "generative_tasks": "Good" if expressibility_score > 0.6 else "Limited",
        },
    }


def _analyze_ansatz_structure(
    gates: list[dict[str, Any]],
    num_qubits: int
) -> dict[str, Any]:
    """Analyze the structure of a variational ansatz."""
    rotation_gates = {"rx", "ry", "rz", "u", "u1", "u2", "u3", "p"}
    entangling_gates = {"cx", "cnot", "cz", "cy", "swap", "crx", "cry", "crz", "cp"}
    
    rotation_count = 0
    entangling_count = 0
    barriers = 0
    estimated_params = 0
    
    # Track layers (separated by barriers or pattern)
    current_layer_rotations = 0
    current_layer_entangling = 0
    layers: list[dict[str, int]] = []
    
    # Entanglement pattern detection
    entangling_pairs: list[tuple[int, int]] = []
    
    for gate in gates:
        name = gate.get("name", "").lower()
        qubits = gate.get("qubits", [])
        
        if name == "barrier":
            if current_layer_rotations > 0 or current_layer_entangling > 0:
                layers.append({
                    "rotations": current_layer_rotations,
                    "entangling": current_layer_entangling
                })
                current_layer_rotations = 0
                current_layer_entangling = 0
            barriers += 1
            continue
        
        if name in rotation_gates:
            rotation_count += 1
            current_layer_rotations += 1
            # Estimate parameters
            if name in ("rx", "ry", "rz", "p", "u1"):
                estimated_params += 1
            elif name in ("u2",):
                estimated_params += 2
            elif name in ("u", "u3"):
                estimated_params += 3
        
        elif name in entangling_gates:
            entangling_count += 1
            current_layer_entangling += 1
            if len(qubits) == 2:
                entangling_pairs.append((qubits[0], qubits[1]))
    
    # Add final layer if exists
    if current_layer_rotations > 0 or current_layer_entangling > 0:
        layers.append({
            "rotations": current_layer_rotations,
            "entangling": current_layer_entangling
        })
    
    # Detect entanglement pattern
    pattern = _detect_entanglement_pattern(entangling_pairs, num_qubits)
    
    num_layers = len(layers) if layers else 1
    
    return {
        "rotation_count": rotation_count,
        "entangling_count": entangling_count,
        "estimated_parameters": estimated_params,
        "num_layers": num_layers,
        "rotations_per_layer": rotation_count / num_layers if num_layers > 0 else 0,
        "entangling_per_layer": entangling_count / num_layers if num_layers > 0 else 0,
        "entanglement_pattern": pattern,
        "layers_detail": layers,
    }


def _detect_entanglement_pattern(
    pairs: list[tuple[int, int]],
    num_qubits: int
) -> str:
    """Detect the entanglement pattern."""
    if not pairs:
        return "none"
    
    # Check for linear pattern
    linear_pairs = [(i, i+1) for i in range(num_qubits - 1)]
    linear_pairs_rev = [(i+1, i) for i in range(num_qubits - 1)]
    
    if all(p in pairs or (p[1], p[0]) in pairs for p in linear_pairs):
        # Check if circular
        if (0, num_qubits - 1) in pairs or (num_qubits - 1, 0) in pairs:
            return "circular"
        return "linear"
    
    # Check for all-to-all
    all_pairs = [(i, j) for i in range(num_qubits) for j in range(i+1, num_qubits)]
    if all(p in pairs or (p[1], p[0]) in pairs for p in all_pairs):
        return "full"
    
    # Check for alternating
    even_pairs = [(i, i+1) for i in range(0, num_qubits - 1, 2)]
    odd_pairs = [(i, i+1) for i in range(1, num_qubits - 1, 2)]
    if all(p in pairs or (p[1], p[0]) in pairs for p in even_pairs):
        if all(p in pairs or (p[1], p[0]) in pairs for p in odd_pairs):
            return "alternating"
    
    return "custom"


def _calculate_entangling_capability(
    gates: list[dict[str, Any]],
    num_qubits: int
) -> float:
    """Calculate entangling capability score."""
    entangling_gates = {"cx", "cnot", "cz", "cy", "swap", "crx", "cry", "crz", "cp"}
    
    entangling_count = 0
    qubit_pairs_entangled: set[tuple[int, int]] = set()
    
    for gate in gates:
        name = gate.get("name", "").lower()
        qubits = gate.get("qubits", [])
        
        if name in entangling_gates and len(qubits) >= 2:
            entangling_count += 1
            pair = (min(qubits[0], qubits[1]), max(qubits[0], qubits[1]))
            qubit_pairs_entangled.add(pair)
    
    # Maximum possible pairs
    max_pairs = num_qubits * (num_qubits - 1) // 2
    
    # Coverage score
    coverage = len(qubit_pairs_entangled) / max_pairs if max_pairs > 0 else 0
    
    # Depth score (more entangling layers = more capability)
    depth_factor = min(1.0, entangling_count / (num_qubits * 2))
    
    return (coverage + depth_factor) / 2


def _estimate_expressibility(
    num_params: int,
    num_qubits: int,
    structure: dict[str, Any],
    entangling_score: float
) -> float:
    """Estimate expressibility score."""
    # Base score from parameter count
    dim = 4 ** num_qubits
    param_score = min(1.0, num_params / (dim / 4))
    
    # Structure bonus
    structure_score = 0.0
    if structure["num_layers"] >= 2:
        structure_score += 0.1
    if structure["entangling_per_layer"] >= num_qubits - 1:
        structure_score += 0.2
    if structure["entanglement_pattern"] in ("full", "circular"):
        structure_score += 0.1
    
    # Combine scores
    score = 0.4 * param_score + 0.3 * entangling_score + 0.3 * min(1.0, structure_score)
    
    return min(1.0, score)


def _estimate_meyer_wallach(structure: dict[str, Any], num_qubits: int) -> float:
    """Estimate Meyer-Wallach entanglement measure."""
    # Simplified estimation based on entanglement structure
    entangling_count = structure["entangling_count"]
    
    if entangling_count == 0:
        return 0.0
    
    # Rough estimate: MW measure approaches 0.5 for maximally entangling circuits
    max_entangling = num_qubits * (num_qubits - 1)
    coverage = min(1.0, entangling_count / max_entangling)
    
    return 0.5 * coverage


def analyze_ansatz_trainability(circuit_data: dict[str, Any]) -> dict[str, Any]:
    """
    Analyze potential trainability issues like barren plateaus.
    
    Barren plateaus occur when gradients vanish exponentially with
    circuit depth/width, making training difficult.
    """
    gates = circuit_data.get("gates", [])
    num_qubits = circuit_data.get("num_qubits", 0)
    
    structure = _analyze_ansatz_structure(gates, num_qubits)
    
    # Risk factors for barren plateaus
    risk_factors = []
    risk_score = 0.0
    
    # Factor 1: Circuit depth
    depth_ratio = structure["num_layers"] / num_qubits if num_qubits > 0 else 0
    if depth_ratio > 10:
        risk_factors.append("Very deep circuit (high barren plateau risk)")
        risk_score += 0.3
    elif depth_ratio > 5:
        risk_factors.append("Deep circuit (moderate barren plateau risk)")
        risk_score += 0.15
    
    # Factor 2: Global entanglement
    if structure["entanglement_pattern"] == "full":
        risk_factors.append("Full entanglement pattern increases gradient variance")
        risk_score += 0.2
    
    # Factor 3: Many random parameters
    if structure["estimated_parameters"] > 4 ** num_qubits // 4:
        risk_factors.append("High parameter count relative to Hilbert space")
        risk_score += 0.15
    
    # Factor 4: Large qubit count
    if num_qubits > 10:
        risk_factors.append(f"Large qubit count ({num_qubits}) increases plateau risk")
        risk_score += 0.2
    
    # Recommendations
    recommendations = []
    if risk_score > 0.3:
        recommendations.append("Consider parameter initialization strategies (e.g., identity initialization)")
        recommendations.append("Use local cost functions when possible")
        recommendations.append("Consider layer-wise training")
    
    if structure["entanglement_pattern"] == "full":
        recommendations.append("Consider using linear or alternating entanglement instead")
    
    if not risk_factors:
        risk_factors.append("No major barren plateau risk factors detected")
    
    return {
        "barren_plateau_risk": round(min(1.0, risk_score), 2),
        "risk_level": "High" if risk_score > 0.4 else "Moderate" if risk_score > 0.2 else "Low",
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "trainability_friendly_features": _identify_good_features(structure),
    }


def _identify_good_features(structure: dict[str, Any]) -> list[str]:
    """Identify features that help trainability."""
    good_features = []
    
    if structure["entanglement_pattern"] == "linear":
        good_features.append("Linear entanglement pattern (good for avoiding barren plateaus)")
    
    if structure["num_layers"] <= 3:
        good_features.append("Shallow circuit depth")
    
    if structure["rotations_per_layer"] > 0:
        good_features.append("Sufficient rotation gates for expressivity")
    
    return good_features
