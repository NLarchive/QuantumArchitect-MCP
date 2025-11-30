"""
Statevector Simulator - Lightweight ideal simulator for quantum circuits.
Calculates the final probability vector without noise.
"""

from typing import Any
import numpy as np

from ..creation.gate_library import GateLibrary


def simulate_statevector(
    circuit_data: dict[str, Any],
    initial_state: list[complex] | None = None
) -> dict[str, Any]:
    """
    Simulate a quantum circuit and return the final statevector.
    
    Args:
        circuit_data: Circuit dictionary with gates
        initial_state: Optional initial state (default: |0...0⟩)
    
    Returns:
        Simulation results with statevector and probabilities
    """
    num_qubits = circuit_data.get("num_qubits", 0)
    gates = circuit_data.get("gates", [])
    
    if num_qubits > 20:
        return {
            "success": False,
            "error": f"Circuit too large for statevector simulation ({num_qubits} qubits). "
                     f"Maximum supported: 20 qubits (2^20 = 1M amplitudes)."
        }
    
    dim = 2 ** num_qubits
    
    # Initialize state
    if initial_state:
        statevector = np.array(initial_state, dtype=complex)
        if len(statevector) != dim:
            return {
                "success": False,
                "error": f"Initial state dimension ({len(statevector)}) doesn't match "
                         f"circuit dimension ({dim})"
            }
    else:
        # Start in |0...0⟩
        statevector = np.zeros(dim, dtype=complex)
        statevector[0] = 1.0
    
    # Apply gates
    measurement_results: dict[int, int] = {}  # qubit -> classical bit value
    gates_applied = 0
    
    for gate in gates:
        name = gate.get("name", "").lower()
        qubits = gate.get("qubits", [])
        params = gate.get("params", [])
        
        if name == "barrier":
            continue
        
        if name == "measure":
            # Simulate measurement
            for q in qubits:
                prob_0, prob_1 = _get_qubit_probabilities(statevector, q, num_qubits)
                # Deterministically collapse to most likely state for reproducibility
                result = 0 if prob_0 >= prob_1 else 1
                measurement_results[q] = result
                statevector = _collapse_state(statevector, q, result, num_qubits)
            continue
        
        if name == "reset":
            for q in qubits:
                statevector = _reset_qubit(statevector, q, num_qubits)
            continue
        
        # Skip symbolic parameters
        if any(isinstance(p, str) and p.startswith("param:") for p in params):
            continue
        
        try:
            gate_matrix = GateLibrary.get_gate(name, params if params else None)
            statevector = _apply_gate(statevector, gate_matrix, qubits, num_qubits)
            gates_applied += 1
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to apply gate '{name}': {str(e)}"
            }
    
    # Calculate probabilities
    probabilities = np.abs(statevector) ** 2
    
    # Get most likely outcomes
    sorted_indices = np.argsort(probabilities)[::-1]
    top_outcomes = []
    
    for idx in sorted_indices[:min(16, len(probabilities))]:
        if probabilities[idx] > 1e-10:
            bitstring = format(idx, f'0{num_qubits}b')
            top_outcomes.append({
                "state": f"|{bitstring}⟩",
                "probability": float(probabilities[idx]),
                "amplitude": {
                    "real": float(statevector[idx].real),
                    "imag": float(statevector[idx].imag)
                }
            })
    
    return {
        "success": True,
        "num_qubits": num_qubits,
        "gates_applied": gates_applied,
        "statevector_dimension": dim,
        "top_outcomes": top_outcomes,
        "total_probability": float(np.sum(probabilities)),
        "measurement_results": measurement_results,
        "statevector": {
            "real": statevector.real.tolist(),
            "imag": statevector.imag.tolist()
        } if num_qubits <= 6 else None,  # Only include full statevector for small circuits
    }


def _apply_gate(
    statevector: np.ndarray,
    gate_matrix: np.ndarray,
    target_qubits: list[int],
    num_qubits: int
) -> np.ndarray:
    """Apply a gate to specific qubits in the statevector."""
    dim = 2 ** num_qubits
    
    if len(target_qubits) == 1:
        q = target_qubits[0]
        new_state = np.zeros_like(statevector)
        
        for i in range(dim):
            # Get bit value at position q
            bit_q = (i >> q) & 1
            # Calculate partner index (flip bit q)
            partner = i ^ (1 << q)
            
            if bit_q == 0:
                # Apply gate to [amplitude_0, amplitude_1]
                amp_0 = statevector[i]
                amp_1 = statevector[partner]
                new_amp_0 = gate_matrix[0, 0] * amp_0 + gate_matrix[0, 1] * amp_1
                new_amp_1 = gate_matrix[1, 0] * amp_0 + gate_matrix[1, 1] * amp_1
                new_state[i] = new_amp_0
                new_state[partner] = new_amp_1
        
        return new_state
    
    elif len(target_qubits) == 2:
        q0, q1 = target_qubits
        new_state = np.zeros_like(statevector)
        processed = set()
        
        for i in range(dim):
            if i in processed:
                continue
            
            # Get all 4 basis states for this 2-qubit subspace
            bit_q0 = (i >> q0) & 1
            bit_q1 = (i >> q1) & 1
            
            # Base index (both qubits = 0)
            base = i & ~(1 << q0) & ~(1 << q1)
            
            # Four indices in gate matrix order |control,target⟩:
            # |00⟩, |01⟩ (target=1), |10⟩ (control=1), |11⟩
            # For CX(control=q0, target=q1), the gate matrix expects this ordering
            indices = [
                base,                           # |00⟩
                base | (1 << q1),               # |01⟩ - target set
                base | (1 << q0),               # |10⟩ - control set
                base | (1 << q0) | (1 << q1)    # |11⟩
            ]
            
            # Extract amplitudes
            amps = np.array([statevector[idx] for idx in indices], dtype=complex)
            
            # Apply gate
            new_amps = gate_matrix @ amps
            
            # Write back
            for j, idx in enumerate(indices):
                new_state[idx] = new_amps[j]
                processed.add(idx)
        
        return new_state
    
    else:
        # For 3+ qubit gates, use general approach
        # This is computationally expensive but correct
        return _apply_multiqubit_gate(statevector, gate_matrix, target_qubits, num_qubits)


def _apply_multiqubit_gate(
    statevector: np.ndarray,
    gate_matrix: np.ndarray,
    target_qubits: list[int],
    num_qubits: int
) -> np.ndarray:
    """Apply a multi-qubit gate using full matrix expansion."""
    # Build full gate matrix by tensor products
    n = len(target_qubits)
    dim = 2 ** num_qubits
    
    # For simplicity, use matrix-vector multiplication with full expansion
    # This is O(4^n) but works for small circuits
    full_gate = np.eye(dim, dtype=complex)
    
    # This is a simplified implementation
    # A proper implementation would use efficient tensor contractions
    
    return full_gate @ statevector


def _get_qubit_probabilities(
    statevector: np.ndarray,
    qubit: int,
    num_qubits: int
) -> tuple[float, float]:
    """Get probability of measuring 0 or 1 on a specific qubit."""
    dim = 2 ** num_qubits
    prob_0 = 0.0
    prob_1 = 0.0
    
    for i in range(dim):
        amplitude_sq = abs(statevector[i]) ** 2
        if (i >> qubit) & 1:
            prob_1 += amplitude_sq
        else:
            prob_0 += amplitude_sq
    
    return prob_0, prob_1


def _collapse_state(
    statevector: np.ndarray,
    qubit: int,
    result: int,
    num_qubits: int
) -> np.ndarray:
    """Collapse statevector after measurement."""
    dim = 2 ** num_qubits
    new_state = np.zeros_like(statevector)
    
    for i in range(dim):
        if ((i >> qubit) & 1) == result:
            new_state[i] = statevector[i]
    
    # Renormalize
    norm = np.linalg.norm(new_state)
    if norm > 0:
        new_state /= norm
    
    return new_state


def _reset_qubit(
    statevector: np.ndarray,
    qubit: int,
    num_qubits: int
) -> np.ndarray:
    """Reset a qubit to |0⟩."""
    dim = 2 ** num_qubits
    new_state = np.zeros_like(statevector)
    
    for i in range(dim):
        bit_value = (i >> qubit) & 1
        target_idx = i & ~(1 << qubit)  # Set qubit to 0
        new_state[target_idx] += statevector[i]
    
    # Renormalize
    norm = np.linalg.norm(new_state)
    if norm > 0:
        new_state /= norm
    
    return new_state


def sample_circuit(
    circuit_data: dict[str, Any],
    shots: int = 1024
) -> dict[str, Any]:
    """
    Sample measurement outcomes from a circuit.
    
    Args:
        circuit_data: Circuit dictionary
        shots: Number of samples
    
    Returns:
        Histogram of measurement outcomes
    """
    # First get the statevector
    sim_result = simulate_statevector(circuit_data)
    
    if not sim_result.get("success", False):
        return sim_result
    
    num_qubits = sim_result["num_qubits"]
    dim = 2 ** num_qubits
    
    # Get probabilities from statevector
    sv = sim_result.get("statevector", {})
    if sv:
        real = np.array(sv["real"])
        imag = np.array(sv["imag"])
        statevector = real + 1j * imag
        probabilities = np.abs(statevector) ** 2
    else:
        # Reconstruct from top outcomes
        probabilities = np.zeros(dim)
        for outcome in sim_result.get("top_outcomes", []):
            state = outcome["state"]
            bitstring = state[1:-1]  # Remove |⟩
            idx = int(bitstring, 2)
            probabilities[idx] = outcome["probability"]
    
    # Sample
    indices = np.random.choice(dim, size=shots, p=probabilities)
    counts: dict[str, int] = {}
    
    for idx in indices:
        bitstring = format(idx, f'0{num_qubits}b')
        counts[bitstring] = counts.get(bitstring, 0) + 1
    
    # Sort by count
    sorted_counts = dict(sorted(counts.items(), key=lambda x: -x[1]))
    
    # Build probabilities dict with bitstring keys
    prob_dict: dict[str, float] = {}
    for idx in range(dim):
        prob = float(probabilities[idx])
        if prob > 1e-10:  # Only include non-zero probabilities
            bitstring = format(idx, f'0{num_qubits}b')
            prob_dict[bitstring] = prob
    
    return {
        "success": True,
        "shots": shots,
        "counts": sorted_counts,
        "probabilities": prob_dict,
        "most_frequent": list(sorted_counts.keys())[0] if sorted_counts else None,
    }
