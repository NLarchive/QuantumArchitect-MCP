"""
Complexity Score - Calculates circuit complexity metrics.
Includes depth, gate count, width, and derived metrics.
"""

from typing import Any
import math


def score_complexity(circuit_data: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate comprehensive complexity metrics for a quantum circuit.
    
    Args:
        circuit_data: Circuit dictionary with gates
    
    Returns:
        Complexity scores and metrics
    """
    gates = circuit_data.get("gates", [])
    num_qubits = circuit_data.get("num_qubits", 0)
    
    # Basic metrics
    gate_analysis = _analyze_gates(gates, num_qubits)
    depth = gate_analysis["depth"]
    
    # Derived metrics
    circuit_volume = num_qubits * depth
    gate_density = gate_analysis["total_gates"] / circuit_volume if circuit_volume > 0 else 0
    
    # Two-qubit gate ratio (indicator of entanglement complexity)
    two_qubit_ratio = (
        gate_analysis["two_qubit_gates"] / gate_analysis["total_gates"]
        if gate_analysis["total_gates"] > 0 else 0
    )
    
    # T-gate count (relevant for fault-tolerant quantum computing)
    t_count = gate_analysis["gate_counts"].get("t", 0) + gate_analysis["gate_counts"].get("tdg", 0)
    
    # Calculate overall complexity score (0-100)
    complexity_score = _calculate_complexity_score(
        num_qubits, depth, gate_analysis["total_gates"],
        gate_analysis["two_qubit_gates"], t_count
    )
    
    # Complexity class
    if complexity_score < 20:
        complexity_class = "Trivial"
    elif complexity_score < 40:
        complexity_class = "Simple"
    elif complexity_score < 60:
        complexity_class = "Moderate"
    elif complexity_score < 80:
        complexity_class = "Complex"
    else:
        complexity_class = "Very Complex"
    
    return {
        "complexity_score": round(complexity_score, 1),
        "complexity_class": complexity_class,
        "basic_metrics": {
            "width": num_qubits,
            "depth": depth,
            "total_gates": gate_analysis["total_gates"],
            "single_qubit_gates": gate_analysis["single_qubit_gates"],
            "two_qubit_gates": gate_analysis["two_qubit_gates"],
            "multi_qubit_gates": gate_analysis["multi_qubit_gates"],
        },
        "derived_metrics": {
            "circuit_volume": circuit_volume,
            "gate_density": round(gate_density, 3),
            "two_qubit_ratio": round(two_qubit_ratio, 3),
            "t_count": t_count,
            "cnot_count": gate_analysis["gate_counts"].get("cx", 0) + 
                         gate_analysis["gate_counts"].get("cnot", 0),
        },
        "gate_breakdown": gate_analysis["gate_counts"],
        "qubit_utilization": gate_analysis["qubit_utilization"],
        "critical_path": {
            "max_depth_qubit": gate_analysis["max_depth_qubit"],
            "min_depth_qubit": gate_analysis["min_depth_qubit"],
            "depth_variance": gate_analysis["depth_variance"],
        },
    }


def _analyze_gates(gates: list[dict[str, Any]], num_qubits: int) -> dict[str, Any]:
    """Analyze gate structure of the circuit."""
    gate_counts: dict[str, int] = {}
    qubit_depths: list[int] = [0] * num_qubits
    qubit_gate_counts: list[int] = [0] * num_qubits
    
    single_qubit = 0
    two_qubit = 0
    multi_qubit = 0
    total = 0
    
    two_qubit_names = {"cx", "cnot", "cy", "cz", "swap", "iswap", "cp", "crx", "cry", "crz"}
    three_qubit_names = {"ccx", "toffoli", "cswap", "fredkin"}
    
    for gate in gates:
        name = gate.get("name", "unknown").lower()
        qubits = gate.get("qubits", [])
        
        if name in ("barrier", "measure", "reset"):
            continue
        
        gate_counts[name] = gate_counts.get(name, 0) + 1
        total += 1
        
        # Update depths
        max_depth = max((qubit_depths[q] for q in qubits if q < num_qubits), default=0)
        for q in qubits:
            if q < num_qubits:
                qubit_depths[q] = max_depth + 1
                qubit_gate_counts[q] += 1
        
        # Count by type
        if name in two_qubit_names:
            two_qubit += 1
        elif name in three_qubit_names:
            multi_qubit += 1
        elif len(qubits) == 1:
            single_qubit += 1
        elif len(qubits) == 2:
            two_qubit += 1
        else:
            multi_qubit += 1
    
    depth = max(qubit_depths) if qubit_depths else 0
    
    # Calculate variance in depth
    if num_qubits > 0:
        mean_depth = sum(qubit_depths) / num_qubits
        variance = sum((d - mean_depth) ** 2 for d in qubit_depths) / num_qubits
    else:
        variance = 0
    
    return {
        "gate_counts": gate_counts,
        "total_gates": total,
        "single_qubit_gates": single_qubit,
        "two_qubit_gates": two_qubit,
        "multi_qubit_gates": multi_qubit,
        "depth": depth,
        "qubit_utilization": {
            "depths": qubit_depths,
            "gate_counts": qubit_gate_counts,
        },
        "max_depth_qubit": qubit_depths.index(max(qubit_depths)) if qubit_depths and max(qubit_depths) > 0 else 0,
        "min_depth_qubit": qubit_depths.index(min(qubit_depths)) if qubit_depths else 0,
        "depth_variance": round(variance, 2),
    }


def _calculate_complexity_score(
    width: int,
    depth: int,
    total_gates: int,
    two_qubit_gates: int,
    t_count: int
) -> float:
    """Calculate overall complexity score (0-100)."""
    # Width contribution (log scale)
    width_score = min(25, math.log2(width + 1) * 5)
    
    # Depth contribution
    depth_score = min(25, math.log2(depth + 1) * 4)
    
    # Gate count contribution
    gate_score = min(25, math.log2(total_gates + 1) * 3)
    
    # Two-qubit gate contribution (most impactful for complexity)
    two_q_score = min(15, math.log2(two_qubit_gates + 1) * 4)
    
    # T-count contribution (for fault-tolerance)
    t_score = min(10, math.log2(t_count + 1) * 3)
    
    return width_score + depth_score + gate_score + two_q_score + t_score


def compare_circuits_complexity(circuits: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compare complexity across multiple circuits.
    
    Args:
        circuits: List of circuit dictionaries
    
    Returns:
        Comparative analysis
    """
    results = []
    
    for i, circuit in enumerate(circuits):
        name = circuit.get("name", f"Circuit {i+1}")
        score = score_complexity(circuit)
        results.append({
            "name": name,
            "complexity_score": score["complexity_score"],
            "class": score["complexity_class"],
            "width": score["basic_metrics"]["width"],
            "depth": score["basic_metrics"]["depth"],
            "gates": score["basic_metrics"]["total_gates"],
        })
    
    # Sort by complexity
    results.sort(key=lambda x: x["complexity_score"])
    
    return {
        "circuits_analyzed": len(circuits),
        "comparison": results,
        "simplest": results[0] if results else None,
        "most_complex": results[-1] if results else None,
    }


def estimate_transpilation_overhead(
    circuit_data: dict[str, Any],
    target_basis: list[str] | None = None
) -> dict[str, Any]:
    """
    Estimate overhead from transpiling to a native gate set.
    
    Args:
        circuit_data: Circuit dictionary
        target_basis: Target native gate set (default: IBM native gates)
    
    Returns:
        Estimated transpilation overhead
    """
    if target_basis is None:
        target_basis = ["cx", "id", "rz", "sx", "x"]
    
    gates = circuit_data.get("gates", [])
    
    # Count gates that need decomposition
    needs_decomposition = 0
    estimated_additional_gates = 0
    
    # Decomposition estimates (rough)
    decomposition_costs = {
        "h": 2,      # H = Rz(π) · SX · Rz(π/2)
        "y": 2,      # Y needs decomposition
        "s": 1,      # S = Rz(π/2)
        "sdg": 1,
        "t": 1,      # T = Rz(π/4)
        "tdg": 1,
        "rx": 3,     # Rx needs decomposition
        "ry": 4,     # Ry needs decomposition
        "swap": 3,   # SWAP = 3 CNOTs
        "iswap": 4,
        "cz": 3,     # CZ = H·CX·H
        "cy": 4,
        "ccx": 15,   # Toffoli ≈ 6-15 two-qubit gates
        "cswap": 20,
    }
    
    for gate in gates:
        name = gate.get("name", "").lower()
        
        if name in ("barrier", "measure", "reset"):
            continue
        
        if name not in target_basis:
            needs_decomposition += 1
            estimated_additional_gates += decomposition_costs.get(name, 2)
    
    original_gate_count = len([g for g in gates 
                              if g.get("name", "").lower() not in ("barrier", "measure", "reset")])
    
    estimated_final_count = original_gate_count + estimated_additional_gates - needs_decomposition
    overhead_ratio = estimated_final_count / original_gate_count if original_gate_count > 0 else 1
    
    return {
        "target_basis": target_basis,
        "original_gate_count": original_gate_count,
        "gates_needing_decomposition": needs_decomposition,
        "estimated_additional_gates": estimated_additional_gates,
        "estimated_final_gate_count": estimated_final_count,
        "overhead_ratio": round(overhead_ratio, 2),
        "overhead_percentage": round((overhead_ratio - 1) * 100, 1),
        "note": "Actual transpilation may vary based on optimization level",
    }
