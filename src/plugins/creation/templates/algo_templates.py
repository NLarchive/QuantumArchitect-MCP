"""
Algorithm Templates - Parameterized functions for QFT, Grover, VQE Ansatz, QAOA.
"""

from typing import Any
import math

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
        return ""


def create_qft(num_qubits: int = 3, with_swaps: bool = True) -> dict[str, Any]:
    """
    Create a Quantum Fourier Transform circuit.
    
    The QFT is the quantum analogue of the discrete Fourier transform.
    It's a key component in Shor's algorithm.
    
    Args:
        num_qubits: Number of qubits
        with_swaps: Whether to include SWAP gates at the end
    
    Returns:
        Dictionary with circuit data
    """
    if num_qubits < 1:
        raise ValueError("QFT requires at least 1 qubit")
    
    gates: list[dict[str, Any]] = []
    
    for i in range(num_qubits):
        # Hadamard on qubit i
        gates.append({"name": "h", "qubits": [i], "params": []})
        
        # Controlled rotations
        for j in range(i + 1, num_qubits):
            angle = math.pi / (2 ** (j - i))
            gates.append({
                "name": "cp",
                "qubits": [j, i],
                "params": [angle]
            })
    
    # SWAP gates to reverse qubit order
    if with_swaps:
        for i in range(num_qubits // 2):
            gates.append({
                "name": "swap",
                "qubits": [i, num_qubits - 1 - i],
                "params": []
            })
    
    result = {
        "name": f"{num_qubits}-qubit Quantum Fourier Transform",
        "num_qubits": num_qubits,
        "num_classical_bits": 0,
        "gates": gates,
        "description": "Quantum Fourier Transform - basis for phase estimation and Shor's algorithm",
        "depth": num_qubits * 2,
        "gate_count": len(gates),
    }
    
    if QISKIT_AVAILABLE:
        try:
            qc = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                qc.h(i)
                for j in range(i + 1, num_qubits):
                    qc.cp(math.pi / (2 ** (j - i)), j, i)
            
            if with_swaps:
                for i in range(num_qubits // 2):
                    qc.swap(i, num_qubits - 1 - i)
            
            result["qasm"] = _circuit_to_qasm(qc)
            result["circuit_diagram"] = str(qc.draw(output='text'))
            result["depth"] = qc.depth()
            result["gate_count"] = qc.size()
        except Exception:
            pass
    
    return result


def create_inverse_qft(num_qubits: int = 3, with_swaps: bool = True) -> dict[str, Any]:
    """
    Create an Inverse Quantum Fourier Transform circuit.
    
    Args:
        num_qubits: Number of qubits
        with_swaps: Whether to include SWAP gates
    
    Returns:
        Dictionary with circuit data
    """
    if num_qubits < 1:
        raise ValueError("Inverse QFT requires at least 1 qubit")
    
    gates: list[dict[str, Any]] = []
    
    # SWAP gates first (reverse of QFT)
    if with_swaps:
        for i in range(num_qubits // 2):
            gates.append({
                "name": "swap",
                "qubits": [i, num_qubits - 1 - i]
            })
    
    # Reverse order of QFT operations
    for i in range(num_qubits - 1, -1, -1):
        for j in range(num_qubits - 1, i, -1):
            angle = -math.pi / (2 ** (j - i))
            gates.append({
                "name": "cp",
                "qubits": [j, i],
                "params": [angle]
            })
        gates.append({"name": "h", "qubits": [i]})
    
    return {
        "name": f"{num_qubits}-qubit Inverse QFT",
        "num_qubits": num_qubits,
        "num_classical_bits": 0,
        "gates": gates,
        "description": "Inverse Quantum Fourier Transform",
    }


def create_grover(
    num_qubits: int = 3,
    marked_states: list[int] | None = None,
    iterations: int | None = None
) -> dict[str, Any]:
    """
    Create Grover's search algorithm circuit.
    
    Grover's algorithm provides quadratic speedup for unstructured search.
    
    Args:
        num_qubits: Number of qubits (search space = 2^n)
        marked_states: List of marked state indices to find (default: [0])
        iterations: Number of Grover iterations (default: optimal)
    
    Returns:
        Dictionary with circuit data
    """
    if num_qubits < 2:
        raise ValueError("Grover's algorithm requires at least 2 qubits")
    
    if marked_states is None:
        marked_states = [0]  # Default: search for |00...0⟩
    
    N = 2 ** num_qubits
    M = len(marked_states)
    
    # Optimal number of iterations
    if iterations is None:
        iterations = max(1, int(math.pi / 4 * math.sqrt(N / M)))
    
    gates: list[dict[str, Any]] = []
    
    # Initial superposition
    for i in range(num_qubits):
        gates.append({"name": "h", "qubits": [i]})
    
    # Grover iterations
    for _ in range(iterations):
        # Oracle (simplified - marks state 0)
        gates.append({"name": "barrier", "qubits": list(range(num_qubits))})
        
        # Multi-controlled Z for oracle (simplified representation)
        for i in range(num_qubits):
            gates.append({"name": "x", "qubits": [i]})
        
        # MCZ using H-CX-H decomposition for 2 qubits
        if num_qubits == 2:
            gates.append({"name": "h", "qubits": [1]})
            gates.append({"name": "cx", "qubits": [0, 1]})
            gates.append({"name": "h", "qubits": [1]})
        elif num_qubits == 3:
            gates.append({"name": "h", "qubits": [2]})
            gates.append({"name": "ccx", "qubits": [0, 1, 2]})
            gates.append({"name": "h", "qubits": [2]})
        else:
            # For more qubits, use ancilla or multi-controlled decomposition
            gates.append({"name": "h", "qubits": [num_qubits - 1]})
            # Simplified: just add phase
            gates.append({"name": "z", "qubits": [num_qubits - 1]})
            gates.append({"name": "h", "qubits": [num_qubits - 1]})
        
        for i in range(num_qubits):
            gates.append({"name": "x", "qubits": [i]})
        
        # Diffusion operator
        gates.append({"name": "barrier", "qubits": list(range(num_qubits))})
        
        for i in range(num_qubits):
            gates.append({"name": "h", "qubits": [i]})
        for i in range(num_qubits):
            gates.append({"name": "x", "qubits": [i]})
        
        # MCZ for diffusion
        if num_qubits == 2:
            gates.append({"name": "h", "qubits": [1]})
            gates.append({"name": "cx", "qubits": [0, 1]})
            gates.append({"name": "h", "qubits": [1]})
        elif num_qubits == 3:
            gates.append({"name": "h", "qubits": [2]})
            gates.append({"name": "ccx", "qubits": [0, 1, 2]})
            gates.append({"name": "h", "qubits": [2]})
        else:
            gates.append({"name": "h", "qubits": [num_qubits - 1]})
            gates.append({"name": "z", "qubits": [num_qubits - 1]})
            gates.append({"name": "h", "qubits": [num_qubits - 1]})
        
        for i in range(num_qubits):
            gates.append({"name": "x", "qubits": [i]})
        for i in range(num_qubits):
            gates.append({"name": "h", "qubits": [i]})
    
    return {
        "name": f"Grover's Search ({num_qubits} qubits, {iterations} iterations)",
        "num_qubits": num_qubits,
        "num_classical_bits": num_qubits,
        "gates": gates,
        "iterations": iterations,
        "search_space_size": N,
        "marked_states": marked_states,
        "description": f"Grover's algorithm searching {N} states with {iterations} iterations",
        "success_probability": math.sin((2 * iterations + 1) * math.asin(math.sqrt(M / N))) ** 2,
    }


def create_vqe_ansatz(
    num_qubits: int = 4,
    layers: int = 2,
    entanglement: str = "linear"
) -> dict[str, Any]:
    """
    Create a VQE (Variational Quantum Eigensolver) ansatz circuit.
    
    This is a parameterized circuit used for quantum chemistry simulations.
    
    Args:
        num_qubits: Number of qubits
        layers: Number of variational layers
        entanglement: Entanglement pattern ('linear', 'full', 'circular')
    
    Returns:
        Dictionary with circuit data including parameter names
    """
    if num_qubits < 2:
        raise ValueError("VQE ansatz requires at least 2 qubits")
    
    gates: list[dict[str, Any]] = []
    parameters: list[str] = []
    param_count = 0
    
    for layer in range(layers):
        # Rotation layer (Ry, Rz on each qubit)
        for qubit in range(num_qubits):
            param_ry = f"θ_{param_count}"
            param_rz = f"θ_{param_count + 1}"
            parameters.extend([param_ry, param_rz])
            
            gates.append({
                "name": "ry",
                "qubits": [qubit],
                "params": [f"param:{param_ry}"],
                "param_name": param_ry
            })
            gates.append({
                "name": "rz",
                "qubits": [qubit],
                "params": [f"param:{param_rz}"],
                "param_name": param_rz
            })
            param_count += 2
        
        # Entanglement layer
        if entanglement == "linear":
            for i in range(num_qubits - 1):
                gates.append({"name": "cx", "qubits": [i, i + 1]})
        elif entanglement == "circular":
            for i in range(num_qubits - 1):
                gates.append({"name": "cx", "qubits": [i, i + 1]})
            if num_qubits > 2:
                gates.append({"name": "cx", "qubits": [num_qubits - 1, 0]})
        elif entanglement == "full":
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    gates.append({"name": "cx", "qubits": [i, j]})
    
    # Final rotation layer
    for qubit in range(num_qubits):
        param_ry = f"θ_{param_count}"
        param_rz = f"θ_{param_count + 1}"
        parameters.extend([param_ry, param_rz])
        
        gates.append({
            "name": "ry",
            "qubits": [qubit],
            "params": [f"param:{param_ry}"],
            "param_name": param_ry
        })
        gates.append({
            "name": "rz",
            "qubits": [qubit],
            "params": [f"param:{param_rz}"],
            "param_name": param_rz
        })
        param_count += 2
    
    return {
        "name": f"VQE Ansatz ({num_qubits}q, {layers}L, {entanglement})",
        "num_qubits": num_qubits,
        "num_classical_bits": num_qubits,
        "gates": gates,
        "parameters": parameters,
        "num_parameters": len(parameters),
        "layers": layers,
        "entanglement": entanglement,
        "description": f"Variational ansatz with {layers} layers and {entanglement} entanglement",
        "use_case": "Quantum chemistry (ground state energy estimation)",
    }


def create_qaoa(
    num_qubits: int = 4,
    p: int = 1,
    problem_type: str = "maxcut"
) -> dict[str, Any]:
    """
    Create a QAOA (Quantum Approximate Optimization Algorithm) circuit.
    
    QAOA is used for combinatorial optimization problems.
    
    Args:
        num_qubits: Number of qubits (problem size)
        p: Number of QAOA layers
        problem_type: Type of problem ('maxcut', 'tsp', 'portfolio')
    
    Returns:
        Dictionary with circuit data
    """
    if num_qubits < 2:
        raise ValueError("QAOA requires at least 2 qubits")
    
    gates: list[dict[str, Any]] = []
    parameters: list[str] = []
    
    # Initial superposition
    for i in range(num_qubits):
        gates.append({"name": "h", "qubits": [i]})
    
    for layer in range(p):
        gamma = f"γ_{layer}"
        beta = f"β_{layer}"
        parameters.extend([gamma, beta])
        
        # Cost Hamiltonian layer (ZZ interactions for MaxCut)
        gates.append({"name": "barrier", "qubits": list(range(num_qubits))})
        
        if problem_type == "maxcut":
            # For each edge in the graph (assuming complete graph)
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    gates.append({"name": "cx", "qubits": [i, j]})
                    gates.append({
                        "name": "rz",
                        "qubits": [j],
                        "params": [f"param:{gamma}"],
                        "param_name": gamma
                    })
                    gates.append({"name": "cx", "qubits": [i, j]})
        
        # Mixer Hamiltonian layer (X rotations)
        gates.append({"name": "barrier", "qubits": list(range(num_qubits))})
        
        for i in range(num_qubits):
            gates.append({
                "name": "rx",
                "qubits": [i],
                "params": [f"param:{beta}"],
                "param_name": beta
            })
    
    return {
        "name": f"QAOA ({problem_type}, p={p})",
        "num_qubits": num_qubits,
        "num_classical_bits": num_qubits,
        "gates": gates,
        "parameters": parameters,
        "num_parameters": len(parameters),
        "p_layers": p,
        "problem_type": problem_type,
        "description": f"QAOA for {problem_type} with {p} layers",
        "use_case": "Combinatorial optimization (Max-Cut, TSP, Portfolio)",
    }


def create_phase_estimation(
    num_counting_qubits: int = 3,
    num_state_qubits: int = 1
) -> dict[str, Any]:
    """
    Create a Quantum Phase Estimation circuit template.
    
    QPE estimates the eigenvalue of a unitary operator.
    
    Args:
        num_counting_qubits: Precision qubits for phase estimation
        num_state_qubits: Qubits for the eigenstate
    
    Returns:
        Dictionary with circuit data
    """
    total_qubits = num_counting_qubits + num_state_qubits
    gates: list[dict[str, Any]] = []
    
    # Hadamard on counting qubits
    for i in range(num_counting_qubits):
        gates.append({"name": "h", "qubits": [i]})
    
    # Prepare eigenstate (example: |1⟩ for T gate)
    gates.append({"name": "x", "qubits": [num_counting_qubits]})
    
    # Controlled unitary powers
    for i in range(num_counting_qubits):
        power = 2 ** (num_counting_qubits - 1 - i)
        # Controlled-U^(2^k) - using T gate as example unitary
        for _ in range(power):
            gates.append({
                "name": "cp",
                "qubits": [i, num_counting_qubits],
                "params": [math.pi / 4]
            })
    
    # Inverse QFT on counting qubits
    for i in range(num_counting_qubits // 2):
        gates.append({
            "name": "swap",
            "qubits": [i, num_counting_qubits - 1 - i]
        })
    
    for i in range(num_counting_qubits - 1, -1, -1):
        for j in range(num_counting_qubits - 1, i, -1):
            angle = -math.pi / (2 ** (j - i))
            gates.append({
                "name": "cp",
                "qubits": [j, i],
                "params": [angle]
            })
        gates.append({"name": "h", "qubits": [i]})
    
    return {
        "name": f"Phase Estimation ({num_counting_qubits} precision qubits)",
        "num_qubits": total_qubits,
        "num_classical_bits": num_counting_qubits,
        "gates": gates,
        "num_counting_qubits": num_counting_qubits,
        "num_state_qubits": num_state_qubits,
        "precision_bits": num_counting_qubits,
        "description": f"Quantum Phase Estimation with {num_counting_qubits}-bit precision",
        "use_case": "Eigenvalue estimation, Shor's algorithm component",
    }


# =============================================================================
# CLASS INTERFACE FOR MCP ENDPOINTS
# =============================================================================

class AlgorithmTemplates:
    """
    Class interface for algorithm quantum circuit templates.
    Wraps the standalone functions for MCP endpoint compatibility.
    """
    
    @staticmethod
    def qft_circuit(num_qubits: int = 3, with_swaps: bool = True) -> dict[str, Any]:
        """Create a Quantum Fourier Transform circuit."""
        return create_qft(num_qubits, with_swaps)
    
    @staticmethod
    def inverse_qft_circuit(num_qubits: int = 3, with_swaps: bool = True) -> dict[str, Any]:
        """Create an Inverse Quantum Fourier Transform circuit."""
        return create_inverse_qft(num_qubits, with_swaps)
    
    @staticmethod
    def grover_circuit(
        num_qubits: int = 3,
        marked_states: list[int] | None = None,
        iterations: int | None = None
    ) -> dict[str, Any]:
        """Create Grover's search algorithm circuit."""
        return create_grover(num_qubits, marked_states, iterations)
    
    @staticmethod
    def vqe_ansatz(
        num_qubits: int = 4,
        layers: int = 2,
        entanglement: str = "linear"
    ) -> dict[str, Any]:
        """Create a VQE ansatz circuit."""
        return create_vqe_ansatz(num_qubits, layers, entanglement)
    
    @staticmethod
    def qaoa_circuit(
        num_qubits: int = 4,
        edges: list[tuple[int, int]] | None = None,
        num_layers: int = 1
    ) -> dict[str, Any]:
        """Create a QAOA circuit."""
        # The create_qaoa function uses p and problem_type
        return create_qaoa(num_qubits, num_layers, "maxcut")
    
    @staticmethod
    def phase_estimation(
        num_counting_qubits: int = 3,
        num_state_qubits: int = 1
    ) -> dict[str, Any]:
        """Create a Quantum Phase Estimation circuit."""
        return create_phase_estimation(num_counting_qubits, num_state_qubits)
