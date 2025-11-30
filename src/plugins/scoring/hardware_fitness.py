"""
Hardware Fitness Score - Evaluates how well a circuit fits specific quantum hardware.
"""

from typing import Any

from ..validation.connectivity_validator import (
    DEFAULT_TOPOLOGIES,
    validate_connectivity,
    check_native_gates,
)


def score_hardware_fitness(
    circuit_data: dict[str, Any],
    hardware_name: str
) -> dict[str, Any]:
    """
    Score how well a circuit fits a specific quantum hardware platform.
    
    Args:
        circuit_data: Circuit dictionary
        hardware_name: Target hardware name
    
    Returns:
        Hardware fitness score and analysis
    """
    # Validate hardware exists
    hardware_key = next(
        (k for k in DEFAULT_TOPOLOGIES if k.lower() == hardware_name.lower()),
        None
    )
    
    if not hardware_key:
        return {
            "error": f"Unknown hardware: {hardware_name}",
            "available_hardware": list(DEFAULT_TOPOLOGIES.keys()),
        }
    
    hardware_info = DEFAULT_TOPOLOGIES[hardware_key]
    
    # Run validations
    connectivity_result = validate_connectivity(circuit_data, hardware_name)
    native_gates_result = check_native_gates(circuit_data, hardware_name)
    
    # Calculate component scores
    scores = {}
    
    # Qubit fit score
    circuit_qubits = circuit_data.get("num_qubits", 0)
    hw_qubits = hardware_info.get("num_qubits", 0)
    qubit_utilization = circuit_qubits / hw_qubits if hw_qubits > 0 else 1
    
    if qubit_utilization <= 0.5:
        scores["qubit_fit"] = 1.0  # Good - plenty of room
    elif qubit_utilization <= 0.8:
        scores["qubit_fit"] = 0.8
    elif qubit_utilization <= 1.0:
        scores["qubit_fit"] = 0.6
    else:
        scores["qubit_fit"] = 0.0  # Circuit too big
    
    # Connectivity score
    if connectivity_result["valid"]:
        scores["connectivity"] = 1.0
    else:
        num_errors = len(connectivity_result.get("errors", []))
        # Penalize based on number of connectivity violations
        scores["connectivity"] = max(0, 1 - num_errors * 0.2)
    
    # Native gates score
    if native_gates_result["valid"]:
        scores["native_gates"] = 1.0
    else:
        num_non_native = len(native_gates_result.get("non_native_gates", []))
        total_gates = sum(1 for g in circuit_data.get("gates", [])
                        if g.get("name", "").lower() not in ("barrier", "measure", "reset"))
        if total_gates > 0:
            non_native_ratio = num_non_native / total_gates
            scores["native_gates"] = max(0, 1 - non_native_ratio)
        else:
            scores["native_gates"] = 1.0
    
    # Depth fitness
    gates = circuit_data.get("gates", [])
    depth = _calculate_depth(gates, circuit_qubits)
    
    # Different hardware has different depth tolerances
    depth_tolerance = _get_depth_tolerance(hardware_key)
    if depth <= depth_tolerance["excellent"]:
        scores["depth"] = 1.0
    elif depth <= depth_tolerance["good"]:
        scores["depth"] = 0.8
    elif depth <= depth_tolerance["fair"]:
        scores["depth"] = 0.5
    else:
        scores["depth"] = max(0, 1 - (depth - depth_tolerance["fair"]) / 100)
    
    # Calculate overall fitness
    weights = {
        "qubit_fit": 0.2,
        "connectivity": 0.35,
        "native_gates": 0.25,
        "depth": 0.2,
    }
    
    overall_score = sum(scores[k] * weights[k] for k in scores)
    
    # Generate grade
    if overall_score >= 0.9:
        grade = "A"
        fitness_level = "Excellent"
    elif overall_score >= 0.75:
        grade = "B"
        fitness_level = "Good"
    elif overall_score >= 0.6:
        grade = "C"
        fitness_level = "Fair"
    elif overall_score >= 0.4:
        grade = "D"
        fitness_level = "Poor"
    else:
        grade = "F"
        fitness_level = "Not Recommended"
    
    # Generate recommendations
    recommendations = _generate_recommendations(
        scores, hardware_info, connectivity_result, native_gates_result
    )
    
    return {
        "hardware": hardware_info["name"],
        "overall_score": round(overall_score * 10, 1),  # 0-10 scale
        "grade": grade,
        "fitness_level": fitness_level,
        "component_scores": {
            "qubit_fit": round(scores["qubit_fit"] * 10, 1),
            "connectivity": round(scores["connectivity"] * 10, 1),
            "native_gates": round(scores["native_gates"] * 10, 1),
            "depth_tolerance": round(scores["depth"] * 10, 1),
        },
        "details": {
            "circuit_qubits": circuit_qubits,
            "hardware_qubits": hw_qubits,
            "circuit_depth": depth,
            "connectivity_errors": len(connectivity_result.get("errors", [])),
            "non_native_gates": len(native_gates_result.get("non_native_gates", [])),
        },
        "hardware_info": {
            "native_gates": hardware_info.get("native_gates", []),
            "connectivity_type": hardware_info.get("description", ""),
        },
        "recommendations": recommendations,
    }


def _calculate_depth(gates: list[dict[str, Any]], num_qubits: int) -> int:
    """Calculate circuit depth."""
    if num_qubits == 0:
        return 0
    
    qubit_depths = [0] * num_qubits
    
    for gate in gates:
        name = gate.get("name", "").lower()
        qubits = gate.get("qubits", [])
        
        if name == "barrier":
            continue
        
        max_depth = max((qubit_depths[q] for q in qubits if q < num_qubits), default=0)
        for q in qubits:
            if q < num_qubits:
                qubit_depths[q] = max_depth + 1
    
    return max(qubit_depths) if qubit_depths else 0


def _get_depth_tolerance(hardware_key: str) -> dict[str, int]:
    """Get depth tolerance thresholds for hardware."""
    tolerances = {
        "ibm_brisbane": {"excellent": 50, "good": 100, "fair": 200},
        "ibm_sherbrooke": {"excellent": 50, "good": 100, "fair": 200},
        "rigetti_aspen": {"excellent": 30, "good": 60, "fair": 120},
        "google_sycamore": {"excellent": 40, "good": 80, "fair": 150},
        "ionq_harmony": {"excellent": 100, "good": 200, "fair": 400},
        "quantinuum_h1": {"excellent": 150, "good": 300, "fair": 500},
        "linear_5": {"excellent": 30, "good": 60, "fair": 100},
    }
    return tolerances.get(hardware_key, {"excellent": 50, "good": 100, "fair": 200})


def _generate_recommendations(
    scores: dict[str, float],
    hardware_info: dict[str, Any],
    connectivity_result: dict[str, Any],
    native_gates_result: dict[str, Any]
) -> list[str]:
    """Generate improvement recommendations."""
    recommendations = []
    
    if scores["connectivity"] < 1.0:
        num_errors = len(connectivity_result.get("errors", []))
        recommendations.append(
            f"Insert SWAP gates to fix {num_errors} connectivity violation(s), "
            "or use automatic routing/transpilation"
        )
    
    if scores["native_gates"] < 1.0:
        non_native = native_gates_result.get("non_native_gates", [])
        unique_gates = set(g["gate"] for g in non_native)
        recommendations.append(
            f"Transpile non-native gates to native basis: {', '.join(unique_gates)}"
        )
    
    if scores["depth"] < 0.6:
        recommendations.append(
            "Consider circuit optimization to reduce depth - "
            "deep circuits accumulate more errors"
        )
    
    if scores["qubit_fit"] < 1.0:
        recommendations.append(
            "Circuit uses a high percentage of available qubits - "
            "less room for error correction or routing"
        )
    
    if not recommendations:
        recommendations.append(
            f"Circuit is well-suited for {hardware_info['name']}"
        )
    
    return recommendations


def compare_hardware_fitness(
    circuit_data: dict[str, Any],
    hardware_list: list[str] | None = None
) -> dict[str, Any]:
    """
    Compare circuit fitness across multiple hardware platforms.
    
    Args:
        circuit_data: Circuit dictionary
        hardware_list: List of hardware names (default: all available)
    
    Returns:
        Comparative analysis
    """
    if hardware_list is None:
        hardware_list = list(DEFAULT_TOPOLOGIES.keys())
    
    results = []
    
    for hw_name in hardware_list:
        fitness = score_hardware_fitness(circuit_data, hw_name)
        if "error" not in fitness:
            results.append({
                "hardware": fitness["hardware"],
                "score": fitness["overall_score"],
                "grade": fitness["grade"],
                "fitness_level": fitness["fitness_level"],
            })
    
    # Sort by score
    results.sort(key=lambda x: -x["score"])
    
    return {
        "comparisons": results,
        "best_fit": results[0] if results else None,
        "worst_fit": results[-1] if results else None,
        "recommendation": (
            f"Best hardware match: {results[0]['hardware']} (Score: {results[0]['score']}/10)"
            if results else "No compatible hardware found"
        ),
    }


def get_hardware_suggestions(circuit_data: dict[str, Any]) -> dict[str, Any]:
    """
    Get hardware suggestions based on circuit requirements.
    
    Args:
        circuit_data: Circuit dictionary
    
    Returns:
        Suggested hardware with reasoning
    """
    num_qubits = circuit_data.get("num_qubits", 0)
    gates = circuit_data.get("gates", [])
    
    # Analyze circuit characteristics
    depth = _calculate_depth(gates, num_qubits)
    
    two_qubit_gates = sum(1 for g in gates
                         if g.get("name", "").lower() in 
                         {"cx", "cnot", "cz", "cy", "swap"})
    
    suggestions = []
    
    # Check each hardware
    for hw_key, hw_info in DEFAULT_TOPOLOGIES.items():
        hw_qubits = hw_info.get("num_qubits", 0)
        
        if hw_qubits < num_qubits:
            continue  # Too small
        
        reasons = []
        score = 0
        
        # Connectivity advantage
        if hw_info.get("coupling_map") == "all_to_all":
            reasons.append("Full connectivity (no SWAP overhead)")
            score += 3
        elif two_qubit_gates < 10:
            reasons.append("Low two-qubit gate count suits limited connectivity")
            score += 1
        
        # Depth tolerance
        if "ionq" in hw_key.lower() or "quantinuum" in hw_key.lower():
            if depth > 100:
                reasons.append("Ion trap has better coherence for deep circuits")
                score += 2
        else:
            if depth < 50:
                reasons.append("Shallow circuit works well on superconducting hardware")
                score += 1
        
        # Size match
        qubit_ratio = num_qubits / hw_qubits if hw_qubits > 0 else 1
        if 0.1 <= qubit_ratio <= 0.5:
            reasons.append("Good qubit utilization ratio")
            score += 1
        
        if reasons:
            suggestions.append({
                "hardware": hw_info["name"],
                "score": score,
                "reasons": reasons,
            })
    
    suggestions.sort(key=lambda x: -x["score"])
    
    return {
        "circuit_requirements": {
            "qubits_needed": num_qubits,
            "depth": depth,
            "two_qubit_gates": two_qubit_gates,
        },
        "suggestions": suggestions[:5],  # Top 5
        "top_recommendation": suggestions[0] if suggestions else None,
    }
