"""
Noise Estimator - Estimates how much error a circuit will accumulate.
Based on circuit depth, gate types, and decoherence times.
"""

from typing import Any


# Default noise parameters (inspired by typical superconducting qubits)
DEFAULT_NOISE_PARAMS = {
    "t1_us": 100.0,           # T1 relaxation time in microseconds
    "t2_us": 80.0,            # T2 dephasing time in microseconds
    "gate_time_1q_ns": 35.0,  # Single-qubit gate time in nanoseconds
    "gate_time_2q_ns": 300.0, # Two-qubit gate time in nanoseconds
    "gate_error_1q": 0.001,   # Single-qubit gate error rate
    "gate_error_2q": 0.01,    # Two-qubit gate error rate
    "readout_error": 0.02,    # Measurement error rate
}


def estimate_noise(
    circuit_data: dict[str, Any],
    noise_params: dict[str, float] | None = None
) -> dict[str, Any]:
    """
    Estimate the expected noise/error for a quantum circuit.
    
    Args:
        circuit_data: Circuit dictionary with gates
        noise_params: Custom noise parameters (optional)
    
    Returns:
        Noise estimation with breakdown by source
    """
    params = {**DEFAULT_NOISE_PARAMS, **(noise_params or {})}
    gates = circuit_data.get("gates", [])
    num_qubits = circuit_data.get("num_qubits", 0)
    
    # Count gates by type
    single_qubit_gates = 0
    two_qubit_gates = 0
    multi_qubit_gates = 0
    measurements = 0
    
    # Track depth per qubit
    qubit_depths: dict[int, int] = {i: 0 for i in range(num_qubits)}
    
    two_qubit_gate_names = {"cx", "cnot", "cy", "cz", "swap", "iswap", "cp", "crx", "cry", "crz"}
    three_qubit_gate_names = {"ccx", "toffoli", "cswap", "fredkin"}
    
    for gate in gates:
        name = gate.get("name", "").lower()
        qubits = gate.get("qubits", [])
        
        if name in ("barrier",):
            continue
        
        if name == "measure":
            measurements += 1
            continue
        
        # Update depth
        max_depth = max((qubit_depths.get(q, 0) for q in qubits), default=0)
        for q in qubits:
            qubit_depths[q] = max_depth + 1
        
        # Count by type
        if name in two_qubit_gate_names:
            two_qubit_gates += 1
        elif name in three_qubit_gate_names:
            # Treat as multiple 2-qubit gates (typical decomposition)
            two_qubit_gates += 6  # CCX ≈ 6 CNOTs
        elif len(qubits) == 1:
            single_qubit_gates += 1
        else:
            multi_qubit_gates += 1
    
    circuit_depth = max(qubit_depths.values()) if qubit_depths else 0
    
    # Calculate gate errors
    gate_error_1q = single_qubit_gates * params["gate_error_1q"]
    gate_error_2q = two_qubit_gates * params["gate_error_2q"]
    total_gate_error = gate_error_1q + gate_error_2q
    
    # Calculate circuit duration (simplified)
    total_time_ns = (
        single_qubit_gates * params["gate_time_1q_ns"] +
        two_qubit_gates * params["gate_time_2q_ns"]
    )
    total_time_us = total_time_ns / 1000.0
    
    # Decoherence error (simplified model)
    # Probability of coherence loss per qubit over circuit duration
    t1_error_per_qubit = 1 - (2.718 ** (-total_time_us / params["t1_us"]))
    t2_error_per_qubit = 1 - (2.718 ** (-total_time_us / params["t2_us"]))
    
    decoherence_error = num_qubits * (t1_error_per_qubit + t2_error_per_qubit) / 2
    
    # Readout error
    readout_error_total = measurements * params["readout_error"]
    
    # Total error estimation (simplified - assumes independent errors)
    total_estimated_error = min(1.0, total_gate_error + decoherence_error + readout_error_total)
    
    # Expected fidelity
    expected_fidelity = max(0.0, 1.0 - total_estimated_error)
    
    # Quality assessment
    if expected_fidelity > 0.99:
        quality = "Excellent"
    elif expected_fidelity > 0.95:
        quality = "Good"
    elif expected_fidelity > 0.8:
        quality = "Fair"
    elif expected_fidelity > 0.5:
        quality = "Poor"
    else:
        quality = "Very Poor - Consider simplifying circuit"
    
    return {
        "estimated_fidelity": round(expected_fidelity, 4),
        "total_error": round(total_estimated_error, 4),
        "quality_assessment": quality,
        "error_breakdown": {
            "gate_errors": round(total_gate_error, 4),
            "single_qubit_gate_error": round(gate_error_1q, 4),
            "two_qubit_gate_error": round(gate_error_2q, 4),
            "decoherence_error": round(decoherence_error, 4),
            "readout_error": round(readout_error_total, 4),
        },
        "circuit_metrics": {
            "depth": circuit_depth,
            "single_qubit_gates": single_qubit_gates,
            "two_qubit_gates": two_qubit_gates,
            "total_gates": single_qubit_gates + two_qubit_gates,
            "measurements": measurements,
            "estimated_duration_us": round(total_time_us, 2),
        },
        "noise_parameters_used": params,
        "recommendations": _get_recommendations(
            circuit_depth, two_qubit_gates, expected_fidelity, total_time_us
        ),
    }


def _get_recommendations(
    depth: int,
    two_qubit_gates: int,
    fidelity: float,
    duration_us: float
) -> list[str]:
    """Generate recommendations based on circuit analysis."""
    recommendations = []
    
    if depth > 100:
        recommendations.append(
            f"Circuit depth ({depth}) is very high. Consider circuit optimization "
            "or error mitigation techniques."
        )
    elif depth > 50:
        recommendations.append(
            f"Circuit depth ({depth}) is moderately high. May benefit from optimization."
        )
    
    if two_qubit_gates > 50:
        recommendations.append(
            f"High number of 2-qubit gates ({two_qubit_gates}). These are the main "
            "source of errors. Consider reducing entanglement where possible."
        )
    
    if fidelity < 0.5:
        recommendations.append(
            "Expected fidelity is very low. Consider using error mitigation techniques "
            "like Zero-Noise Extrapolation (ZNE) or Probabilistic Error Cancellation (PEC)."
        )
    
    if duration_us > 100:
        recommendations.append(
            f"Circuit duration ({duration_us:.1f} μs) approaches coherence times. "
            "Decoherence will significantly affect results."
        )
    
    if not recommendations:
        recommendations.append(
            "Circuit appears well-suited for NISQ hardware. Consider running with "
            "sufficient shots for statistical significance."
        )
    
    return recommendations


def estimate_fidelity(
    circuit_data: dict[str, Any],
    noise_params: dict[str, float] | None = None
) -> dict[str, Any]:
    """
    Estimate the expected fidelity of a quantum circuit.
    
    This is a convenience wrapper around estimate_noise that focuses
    on fidelity estimation.
    
    Args:
        circuit_data: Circuit dictionary with gates
        noise_params: Custom noise parameters (optional)
    
    Returns:
        Fidelity estimation with breakdown
    """
    full_result = estimate_noise(circuit_data, noise_params)
    
    return {
        "fidelity": full_result.get("estimated_fidelity", 0.0),
        "quality": full_result.get("quality_assessment", "Unknown"),
        "error_breakdown": full_result.get("error_breakdown", {}),
        "circuit_metrics": full_result.get("circuit_metrics", {}),
        "recommendations": full_result.get("recommendations", []),
    }


def estimate_required_shots(
    circuit_data: dict[str, Any],
    target_precision: float = 0.01,
    confidence_level: float = 0.95
) -> dict[str, Any]:
    """
    Estimate the number of shots required for desired precision.
    
    Args:
        circuit_data: Circuit dictionary
        target_precision: Desired probability precision (e.g., 0.01 = 1%)
        confidence_level: Statistical confidence (e.g., 0.95 = 95%)
    
    Returns:
        Shot estimation with statistical analysis
    """
    import math
    
    num_qubits = circuit_data.get("num_qubits", 0)
    num_outcomes = 2 ** num_qubits
    
    # Get noise estimation for error-adjusted calculations
    noise_result = estimate_noise(circuit_data)
    expected_fidelity = noise_result["estimated_fidelity"]
    
    # Z-score for confidence level
    z_scores = {
        0.90: 1.645,
        0.95: 1.960,
        0.99: 2.576,
    }
    z = z_scores.get(confidence_level, 1.960)
    
    # Basic shot estimation using binomial proportion confidence interval
    # n = (z^2 * p * (1-p)) / E^2
    # Assuming worst case p = 0.5
    basic_shots = int(math.ceil((z ** 2 * 0.5 * 0.5) / (target_precision ** 2)))
    
    # Adjust for number of outcomes (need more shots to resolve more states)
    adjusted_shots = basic_shots * min(num_outcomes, 16)  # Cap at 16 outcomes
    
    # Adjust for expected noise (need more shots for noisy circuits)
    noise_factor = 1.0 / max(expected_fidelity, 0.1)
    final_shots = int(math.ceil(adjusted_shots * noise_factor))
    
    # Practical bounds
    min_shots = 100
    max_practical = 1_000_000
    recommended_shots = max(min_shots, min(final_shots, max_practical))
    
    return {
        "recommended_shots": recommended_shots,
        "minimum_shots": min_shots,
        "statistical_minimum": basic_shots,
        "target_precision": target_precision,
        "confidence_level": confidence_level,
        "expected_fidelity": expected_fidelity,
        "noise_factor": round(noise_factor, 2),
        "num_possible_outcomes": num_outcomes,
        "practical_note": _get_shot_note(recommended_shots, num_qubits, expected_fidelity),
    }


def _get_shot_note(shots: int, num_qubits: int, fidelity: float) -> str:
    """Generate practical note about shot count."""
    if fidelity < 0.5:
        return (
            f"Due to low expected fidelity ({fidelity:.2f}), even {shots:,} shots "
            "may not yield reliable results. Consider circuit simplification."
        )
    elif num_qubits > 10:
        return (
            f"With {num_qubits} qubits and {2**num_qubits:,} possible outcomes, "
            f"{shots:,} shots provides statistical sampling of the distribution."
        )
    else:
        return f"{shots:,} shots should provide reliable probability estimates."


def compare_noise_models(
    circuit_data: dict[str, Any],
    hardware_configs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """
    Compare expected noise across different hardware configurations.
    
    Args:
        circuit_data: Circuit dictionary
        hardware_configs: List of hardware noise configurations
    
    Returns:
        Comparison of expected performance across hardware
    """
    if hardware_configs is None:
        # Default comparisons
        hardware_configs = [
            {
                "name": "IBM Eagle (Typical)",
                "params": {
                    "t1_us": 100.0, "t2_us": 80.0,
                    "gate_error_1q": 0.001, "gate_error_2q": 0.01,
                }
            },
            {
                "name": "Rigetti Aspen (Typical)",
                "params": {
                    "t1_us": 30.0, "t2_us": 25.0,
                    "gate_error_1q": 0.002, "gate_error_2q": 0.02,
                }
            },
            {
                "name": "IonQ (Typical)",
                "params": {
                    "t1_us": 1000.0, "t2_us": 500.0,
                    "gate_error_1q": 0.0005, "gate_error_2q": 0.005,
                    "gate_time_1q_ns": 10000, "gate_time_2q_ns": 100000,
                }
            },
            {
                "name": "Ideal (No Noise)",
                "params": {
                    "t1_us": 1e9, "t2_us": 1e9,
                    "gate_error_1q": 0.0, "gate_error_2q": 0.0,
                    "readout_error": 0.0,
                }
            },
        ]
    
    results = []
    for config in hardware_configs:
        noise_result = estimate_noise(circuit_data, config.get("params", {}))
        results.append({
            "hardware": config["name"],
            "estimated_fidelity": noise_result["estimated_fidelity"],
            "quality": noise_result["quality_assessment"],
            "dominant_error_source": _get_dominant_error(noise_result["error_breakdown"]),
        })
    
    # Sort by fidelity
    results.sort(key=lambda x: -x["estimated_fidelity"])
    
    return {
        "comparisons": results,
        "best_hardware": results[0]["hardware"] if results else None,
        "worst_hardware": results[-1]["hardware"] if results else None,
    }


def _get_dominant_error(breakdown: dict[str, float]) -> str:
    """Identify the dominant error source."""
    sources = [
        ("Gate errors", breakdown.get("gate_errors", 0)),
        ("Decoherence", breakdown.get("decoherence_error", 0)),
        ("Readout", breakdown.get("readout_error", 0)),
    ]
    sources.sort(key=lambda x: -x[1])
    return sources[0][0] if sources[0][1] > 0 else "None"
