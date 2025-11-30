"""
Resource Estimator - Calculates computational resources required for circuits.
Includes shot estimation, memory requirements, and execution time estimates.
"""

from typing import Any
import math


def estimate_resources(
    circuit_data: dict[str, Any],
    execution_mode: str = "simulation"
) -> dict[str, Any]:
    """
    Estimate computational resources required for a quantum circuit.
    
    Args:
        circuit_data: Circuit dictionary
        execution_mode: "simulation" or "hardware"
    
    Returns:
        Resource estimation including memory, time, and shots
    """
    num_qubits = circuit_data.get("num_qubits", 0)
    gates = circuit_data.get("gates", [])
    
    # Gate counting
    gate_counts = _count_gates(gates)
    total_gates = sum(gate_counts.values())
    
    # Depth calculation
    depth = _calculate_depth(gates, num_qubits)
    
    if execution_mode == "simulation":
        return _estimate_simulation_resources(num_qubits, total_gates, depth, gate_counts)
    else:
        return _estimate_hardware_resources(num_qubits, total_gates, depth, gate_counts)


def _count_gates(gates: list[dict[str, Any]]) -> dict[str, int]:
    """Count gates by type."""
    counts: dict[str, int] = {}
    for gate in gates:
        name = gate.get("name", "unknown").lower()
        if name not in ("barrier",):
            counts[name] = counts.get(name, 0) + 1
    return counts


def _calculate_depth(gates: list[dict[str, Any]], num_qubits: int) -> int:
    """Calculate circuit depth."""
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


def _estimate_simulation_resources(
    num_qubits: int,
    total_gates: int,
    depth: int,
    gate_counts: dict[str, int]
) -> dict[str, Any]:
    """Estimate resources for classical simulation."""
    
    # Statevector simulation
    dim = 2 ** num_qubits
    complex_size = 16  # bytes for complex128
    
    statevector_memory_bytes = dim * complex_size
    statevector_memory_mb = statevector_memory_bytes / (1024 ** 2)
    statevector_memory_gb = statevector_memory_bytes / (1024 ** 3)
    
    # Operation complexity
    single_qubit_ops = gate_counts.get("h", 0) + gate_counts.get("x", 0) + \
                       gate_counts.get("y", 0) + gate_counts.get("z", 0) + \
                       gate_counts.get("rx", 0) + gate_counts.get("ry", 0) + \
                       gate_counts.get("rz", 0) + gate_counts.get("s", 0) + \
                       gate_counts.get("t", 0)
    
    two_qubit_ops = gate_counts.get("cx", 0) + gate_counts.get("cnot", 0) + \
                    gate_counts.get("cz", 0) + gate_counts.get("swap", 0)
    
    # Rough FLOP estimate
    flops_1q = single_qubit_ops * dim * 8  # 8 complex ops per amplitude
    flops_2q = two_qubit_ops * dim * 16    # 16 complex ops per amplitude
    total_flops = flops_1q + flops_2q
    
    # Time estimate (assuming 10 GFLOPs for typical hardware)
    gflops_available = 10.0
    time_seconds = total_flops / (gflops_available * 1e9) if total_flops > 0 else 0.001
    
    # Feasibility assessment
    if num_qubits <= 20:
        feasibility = "Easy - Standard laptop"
    elif num_qubits <= 30:
        feasibility = "Moderate - Requires HPC cluster"
    elif num_qubits <= 40:
        feasibility = "Hard - Requires specialized supercomputer"
    elif num_qubits <= 50:
        feasibility = "Very Hard - Frontier-class supercomputer"
    else:
        feasibility = "Infeasible - Beyond current classical computing"
    
    return {
        "mode": "simulation",
        "num_qubits": num_qubits,
        "circuit_depth": depth,
        "total_gates": total_gates,
        "memory": {
            "statevector_bytes": statevector_memory_bytes,
            "statevector_mb": round(statevector_memory_mb, 2),
            "statevector_gb": round(statevector_memory_gb, 4),
            "hilbert_space_dimension": dim,
        },
        "computation": {
            "estimated_flops": total_flops,
            "estimated_time_seconds": round(time_seconds, 4),
            "estimated_time_human": _format_time(time_seconds),
        },
        "feasibility": feasibility,
        "recommendations": _get_simulation_recommendations(num_qubits, depth),
    }


def _estimate_hardware_resources(
    num_qubits: int,
    total_gates: int,
    depth: int,
    gate_counts: dict[str, int]
) -> dict[str, Any]:
    """Estimate resources for quantum hardware execution."""
    
    # Shots recommendation
    measurements = gate_counts.get("measure", 0)
    if measurements == 0:
        measurements = num_qubits  # Assume full measurement
    
    # Base shots for statistical significance
    base_shots = 1024
    
    # Adjust for circuit complexity
    if depth > 100:
        shots_multiplier = 4
    elif depth > 50:
        shots_multiplier = 2
    else:
        shots_multiplier = 1
    
    recommended_shots = base_shots * shots_multiplier
    
    # Queue time estimates (typical for cloud quantum services)
    queue_estimates = {
        "ibm_free": "5-30 minutes",
        "ibm_premium": "1-5 minutes",
        "aws_braket": "1-10 minutes",
        "azure_quantum": "1-10 minutes",
    }
    
    # Execution time estimate
    gate_time_ns = 35 * (total_gates - gate_counts.get("cx", 0) - gate_counts.get("cnot", 0)) + \
                   300 * (gate_counts.get("cx", 0) + gate_counts.get("cnot", 0))
    execution_time_per_shot_us = gate_time_ns / 1000
    total_execution_time_ms = (execution_time_per_shot_us * recommended_shots) / 1000
    
    # Cost estimates (rough, varies by provider)
    cost_per_shot_usd = 0.00003  # Rough average
    estimated_cost = recommended_shots * cost_per_shot_usd
    
    return {
        "mode": "hardware",
        "num_qubits": num_qubits,
        "circuit_depth": depth,
        "total_gates": total_gates,
        "execution": {
            "recommended_shots": recommended_shots,
            "min_shots": 100,
            "max_shots": 100000,
            "time_per_shot_us": round(execution_time_per_shot_us, 2),
            "total_execution_time_ms": round(total_execution_time_ms, 2),
        },
        "queue_estimates": queue_estimates,
        "cost_estimate": {
            "per_shot_usd": cost_per_shot_usd,
            "total_estimated_usd": round(estimated_cost, 4),
            "note": "Actual costs vary by provider and plan",
        },
        "hardware_requirements": {
            "min_qubits_needed": num_qubits,
            "connectivity_critical": depth > 20 or gate_counts.get("cx", 0) > 20,
        },
        "recommendations": _get_hardware_recommendations(num_qubits, depth, total_gates),
    }


def _format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hours"
    else:
        return f"{seconds / 86400:.2f} days"


def _get_simulation_recommendations(num_qubits: int, depth: int) -> list[str]:
    """Get recommendations for simulation."""
    recs = []
    
    if num_qubits > 25:
        recs.append("Consider using tensor network methods (e.g., MPS) instead of statevector")
    
    if num_qubits > 30:
        recs.append("Use sparse simulation if circuit has limited entanglement")
        recs.append("Consider GPU acceleration (e.g., cuQuantum)")
    
    if depth < 10 and num_qubits < 30:
        recs.append("Circuit is well-suited for standard statevector simulation")
    
    if not recs:
        recs.append("Standard simulation should work efficiently")
    
    return recs


def _get_hardware_recommendations(num_qubits: int, depth: int, gates: int) -> list[str]:
    """Get recommendations for hardware execution."""
    recs = []
    
    if depth > 100:
        recs.append("High depth circuit - consider error mitigation")
        recs.append("Use Zero-Noise Extrapolation (ZNE) if available")
    
    if num_qubits > 50:
        recs.append("Large circuit - check hardware availability")
        recs.append("Consider circuit partitioning techniques")
    
    if gates > 1000:
        recs.append("Many gates - transpilation optimization critical")
    
    if depth < 20 and num_qubits < 30:
        recs.append("Circuit is well-suited for current NISQ hardware")
    
    if not recs:
        recs.append("Circuit should run on most quantum hardware providers")
    
    return recs


def estimate_quantum_volume_requirement(circuit_data: dict[str, Any]) -> dict[str, Any]:
    """
    Estimate the minimum Quantum Volume required to run this circuit.
    
    Quantum Volume (QV) is a metric that measures the largest random circuit
    that a quantum computer can successfully implement.
    """
    num_qubits = circuit_data.get("num_qubits", 0)
    gates = circuit_data.get("gates", [])
    
    depth = _calculate_depth(gates, num_qubits)
    
    # Effective circuit width (considering qubit utilization)
    used_qubits = set()
    for gate in gates:
        used_qubits.update(gate.get("qubits", []))
    effective_width = len(used_qubits)
    
    # QV is roughly 2^(min(width, depth))
    effective_dimension = min(effective_width, depth)
    required_qv = 2 ** effective_dimension if effective_dimension > 0 else 1
    
    # Cap at practical values
    required_qv = min(required_qv, 2 ** 20)
    
    # Compare with known QV values
    qv_benchmarks = [
        ("IBM Brisbane", 128),
        ("IBM Sherbrooke", 127),
        ("IonQ Aria", 25),
        ("Quantinuum H1-1", 32768),
        ("Rigetti Aspen-M", 8),
    ]
    
    compatible_hardware = [
        hw for hw, qv in qv_benchmarks if qv >= required_qv
    ]
    
    return {
        "circuit_width": effective_width,
        "circuit_depth": depth,
        "effective_dimension": effective_dimension,
        "estimated_required_qv": required_qv,
        "qv_log2": effective_dimension,
        "compatible_hardware": compatible_hardware,
        "note": "Quantum Volume is a rough metric; actual performance depends on many factors",
    }
