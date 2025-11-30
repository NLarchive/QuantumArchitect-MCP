"""
MCP Endpoint Handlers - Tool functions exposed to the Agent.
These are the core MCP tools for quantum circuit operations.
"""

from typing import Any
import json
import traceback

from .schemas import (
    CircuitSchema,
    QASMInput,
    HardwareTarget,
    ValidationResponse,
    SimulationResponse,
    ScoreResponse,
)
from .context_provider import (
    get_hardware_profile,
    list_hardware_profiles,
    get_reference_circuit,
    list_reference_circuits,
    get_gate_documentation,
    get_algorithm_explanation,
    get_learning_resources,
)

# Import core modules
from ..core.circuit_parser import CircuitParser
from ..core.dag_representation import CircuitDAG

# Import plugins
from ..plugins.creation.gate_library import GateLibrary
from ..plugins.creation.templates.beginner_templates import BeginnerTemplates
from ..plugins.creation.templates.algo_templates import AlgorithmTemplates
from ..plugins.creation.visualizers import visualize_circuit_ascii, circuit_summary

from ..plugins.validation.syntax_checker import check_syntax, validate_qasm_syntax
from ..plugins.validation.connectivity_validator import validate_connectivity
from ..plugins.validation.unitary_check import check_unitarity, compute_circuit_unitary

from ..plugins.evaluation.statevector_sim import simulate_statevector, sample_circuit
from ..plugins.evaluation.noise_estimator import estimate_noise, estimate_fidelity
from ..plugins.evaluation.resource_estimator import estimate_resources

from ..plugins.scoring.complexity_score import score_complexity, compare_circuits_complexity
from ..plugins.scoring.hardware_fitness import score_hardware_fitness


# =============================================================================
# CREATION TOOLS
# =============================================================================

def create_circuit_from_template(
    template_name: str,
    num_qubits: int = 2,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a quantum circuit from a predefined template.
    
    Args:
        template_name: Name of the template (bell_state, ghz_state, qft, grover, vqe)
        num_qubits: Number of qubits for the circuit
        parameters: Optional parameters for parameterized templates
    
    Returns:
        Created circuit with QASM representation
    """
    try:
        params = parameters or {}
        template_name_lower = template_name.lower().replace(" ", "_").replace("-", "_")
        
        # Beginner templates (return dictionaries)
        if template_name_lower in ["bell", "bell_state"]:
            circuit_data = BeginnerTemplates.bell_state()
            
        elif template_name_lower in ["ghz", "ghz_state"]:
            circuit_data = BeginnerTemplates.ghz_state(num_qubits)
            
        elif template_name_lower in ["w", "w_state"]:
            circuit_data = BeginnerTemplates.w_state(num_qubits)
            
        elif template_name_lower in ["superposition", "uniform_superposition"]:
            circuit_data = BeginnerTemplates.uniform_superposition(num_qubits)
            
        # Algorithm templates (return dictionaries)
        elif template_name_lower == "qft":
            circuit_data = AlgorithmTemplates.qft_circuit(num_qubits)
            
        elif template_name_lower in ["grover", "grovers"]:
            marked_states = params.get("marked_states", [0])
            iterations = params.get("iterations", None)
            circuit_data = AlgorithmTemplates.grover_circuit(num_qubits, marked_states, iterations)
            
        elif template_name_lower in ["vqe", "vqe_ansatz"]:
            ansatz_depth = params.get("depth", 2)
            circuit_data = AlgorithmTemplates.vqe_ansatz(num_qubits, ansatz_depth)
            
        elif template_name_lower == "qaoa":
            edges = params.get("edges", [(0, 1)])
            num_layers = params.get("num_layers", 1)
            circuit_data = AlgorithmTemplates.qaoa_circuit(num_qubits, edges, num_layers)
            
        else:
            return {
                "success": False,
                "error": f"Unknown template: {template_name}",
                "available_templates": [
                    "bell_state", "ghz_state", "w_state", "superposition",
                    "qft", "grover", "vqe", "qaoa"
                ],
            }
        
        # Templates return dictionaries directly
        return {
            "success": True,
            "template": template_name,
            "num_qubits": circuit_data.get("num_qubits", num_qubits),
            "circuit": circuit_data,
            "qasm": circuit_data.get("qasm", ""),
            "visualization": circuit_data.get("circuit_diagram", ""),
            "summary": {
                "name": circuit_data.get("name", template_name),
                "description": circuit_data.get("description", ""),
                "depth": circuit_data.get("depth", 0),
                "gate_count": circuit_data.get("gate_count", len(circuit_data.get("gates", []))),
            },
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def parse_qasm_circuit(
    qasm_code: str,
) -> dict[str, Any]:
    """
    Parse a QASM string into internal circuit representation.

    Args:
        qasm_code: OpenQASM code string

    Returns:
        Parsed circuit data
    """
    try:
        parser = CircuitParser()
        dag = parser.parse_qasm(qasm_code)

        return {
            "success": True,
            "num_qubits": dag.num_qubits,
            "num_clbits": dag.num_classical_bits,
            "circuit": dag.to_dict(),
            "visualization": visualize_circuit_ascii(dag),
            "summary": circuit_summary(dag),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def build_custom_circuit(
    num_qubits: int,
    gates: list[dict[str, Any]],
    measurements: list[dict[str, int]] | None = None,
) -> dict[str, Any]:
    """
    Build a custom circuit from gate specifications.
    
    Args:
        num_qubits: Number of qubits
        gates: List of gate operations [{"name": "h", "qubits": [0]}, ...]
        measurements: Optional list of measurements [{"qubit": 0, "clbit": 0}, ...]
    
    Returns:
        Built circuit data
    """
    try:
        dag = CircuitDAG(num_qubits, len(measurements) if measurements else 0)
        
        for gate in gates:
            name = gate.get("name", "").lower()
            qubits = gate.get("qubits", [])
            params = gate.get("parameters", [])
            
            if not qubits:
                continue
                
            dag.add_gate(name, qubits, params)
        
        if measurements:
            for meas in measurements:
                qubit = meas.get("qubit", 0)
                clbit = meas.get("clbit", 0)
                dag.add_measurement(qubit, clbit)
        
        return {
            "success": True,
            "num_qubits": dag.num_qubits,
            "circuit": dag.to_dict(),
            "qasm": dag.to_qasm(),
            "visualization": visualize_circuit_ascii(dag),
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# VALIDATION TOOLS
# =============================================================================

def validate_circuit(
    qasm_code: str | None = None,
    circuit_dict: dict[str, Any] | None = None,
    hardware_target: str | None = None,
    check_connectivity: bool = True,
    check_unitary: bool = True,
) -> dict[str, Any]:
    """
    Comprehensive circuit validation including syntax, connectivity, and unitarity.
    
    Args:
        qasm_code: QASM code to validate
        circuit_dict: Alternatively, circuit dictionary
        hardware_target: Target hardware for connectivity check
        check_connectivity: Whether to check hardware connectivity
        check_unitary: Whether to verify unitarity
    
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "checks": {},
        "errors": [],
        "warnings": [],
    }
    
    try:
        # Parse circuit
        if qasm_code:
            syntax_result = validate_qasm_syntax(qasm_code)
            results["checks"]["syntax"] = syntax_result
            
            if not syntax_result["valid"]:
                results["valid"] = False
                results["errors"].extend(syntax_result.get("errors", []))
            
            if syntax_result.get("warnings"):
                results["warnings"].extend(syntax_result["warnings"])
            
            parser = CircuitParser()
            dag = parser.parse_qasm(qasm_code)
            
        elif circuit_dict:
            dag = CircuitDAG.from_dict(circuit_dict)
            results["checks"]["syntax"] = {"valid": True, "source": "dictionary"}
            
        else:
            return {
                "valid": False,
                "error": "Either qasm_code or circuit_dict must be provided",
            }
        
        # Check connectivity
        if check_connectivity and hardware_target:
            # validate_connectivity expects (circuit_data: dict, hardware_name: str)
            circuit_data = dag.to_dict()
            connectivity_result = validate_connectivity(circuit_data, hardware_target)
            results["checks"]["connectivity"] = connectivity_result
            
            if not connectivity_result["valid"]:
                results["valid"] = False
                results["errors"].extend(connectivity_result.get("violations", []))
        
        # Check unitarity
        if check_unitary:
            try:
                unitarity_result = check_unitarity(dag)
                results["checks"]["unitarity"] = unitarity_result
                
                if not unitarity_result.get("is_unitary", True):
                    results["warnings"].append("Circuit may not preserve unitarity")
            except Exception as e:
                results["warnings"].append(f"Unitarity check failed: {str(e)}")
        
        # Add circuit info
        results["circuit_info"] = {
            "num_qubits": dag.num_qubits,
            "num_gates": len(dag.gates),
            "depth": dag.depth,
        }
        
    except Exception as e:
        results["valid"] = False
        results["errors"].append(str(e))
        results["traceback"] = traceback.format_exc()
    
    return results


def check_hardware_compatibility(
    qasm_code: str,
    hardware_name: str,
) -> dict[str, Any]:
    """
    Check if a circuit is compatible with specific hardware.
    
    Args:
        qasm_code: QASM code to check
        hardware_name: Name of target hardware
    
    Returns:
        Compatibility analysis
    """
    try:
        parser = CircuitParser()
        dag = parser.parse_qasm(qasm_code)
        
        hardware_profile = get_hardware_profile(hardware_name)
        if not hardware_profile:
            return {
                "compatible": False,
                "error": f"Hardware profile '{hardware_name}' not found",
                "available_hardware": [p["id"] for p in list_hardware_profiles()],
            }
        
        # Check connectivity
        connectivity_result = validate_connectivity(dag.to_dict(), hardware_name)

        # Check qubit count
        hw_qubits = hardware_profile.get("num_qubits", 0)
        circuit_qubits = dag.num_qubits

        # Score hardware fitness
        fitness = score_hardware_fitness(dag.to_dict(), hardware_name)

        return {
            "compatible": connectivity_result["valid"] and circuit_qubits <= hw_qubits,
            "hardware": {
                "name": hardware_profile.get("name", hardware_name),
                "provider": hardware_profile.get("provider", "Unknown"),
                "available_qubits": hw_qubits,
            },
            "circuit": {
                "required_qubits": circuit_qubits,
                "within_qubit_limit": circuit_qubits <= hw_qubits,
            },
            "connectivity": connectivity_result,
            "fitness_score": fitness,
            "recommendations": fitness.get("recommendations", []),
        }
        
    except Exception as e:
        return {
            "compatible": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# SIMULATION TOOLS
# =============================================================================

def simulate_circuit(
    qasm_code: str,
    shots: int = 1024,
    include_statevector: bool = True,
    noise_model: str | None = None,
) -> dict[str, Any]:
    """
    Simulate a quantum circuit and return results.
    
    Args:
        qasm_code: QASM code to simulate
        shots: Number of measurement shots
        include_statevector: Whether to include statevector in results
        noise_model: Optional noise model name
    
    Returns:
        Simulation results
    """
    try:
        parser = CircuitParser()
        dag = parser.parse_qasm(qasm_code)
        
        results = {
            "success": True,
            "num_qubits": dag.num_qubits,
        }
        
        # Convert DAG to dict for simulation functions
        circuit_dict = dag.to_dict()
        
        # Get statevector
        if include_statevector:
            sv_result = simulate_statevector(circuit_dict)
            results["statevector"] = sv_result
        
        # Sample measurements
        sample_result = sample_circuit(circuit_dict, shots=shots)
        results["counts"] = sample_result.get("counts", {})
        results["probabilities"] = sample_result.get("probabilities", {})
        
        # Estimate noise if model specified
        if noise_model:
            hardware_profile = get_hardware_profile(noise_model)
            if hardware_profile:
                noise_result = estimate_noise(circuit_dict, hardware_profile)
                results["noise_estimate"] = noise_result
                
                fidelity = estimate_fidelity(circuit_dict, hardware_profile)
                results["estimated_fidelity"] = fidelity
        
        # Resource estimation
        resources = estimate_resources(circuit_dict, "simulation")
        results["resources"] = resources
        
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def get_statevector(qasm_code: str) -> dict[str, Any]:
    """
    Get the statevector of a circuit (ideal simulation).
    
    Args:
        qasm_code: QASM code
    
    Returns:
        Statevector information
    """
    try:
        parser = CircuitParser()
        dag = parser.parse_qasm(qasm_code)
        
        # Convert DAG to dict for simulation function
        circuit_dict = dag.to_dict()
        result = simulate_statevector(circuit_dict)
        
        return {
            "success": True,
            "num_qubits": dag.num_qubits,
            **result,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def estimate_circuit_fidelity(
    qasm_code: str,
    hardware_name: str,
) -> dict[str, Any]:
    """
    Estimate the fidelity of a circuit on specific hardware.
    
    Args:
        qasm_code: QASM code
        hardware_name: Target hardware name
    
    Returns:
        Fidelity estimation
    """
    try:
        parser = CircuitParser()
        dag = parser.parse_qasm(qasm_code)
        
        hardware_profile = get_hardware_profile(hardware_name)
        if not hardware_profile:
            return {
                "error": f"Hardware profile '{hardware_name}' not found",
            }
        
        circuit_dict = dag.to_dict()
        fidelity = estimate_fidelity(circuit_dict, hardware_profile)
        noise = estimate_noise(circuit_dict, hardware_profile)
        
        return {
            "success": True,
            "hardware": hardware_name,
            "estimated_fidelity": fidelity,
            "noise_analysis": noise,
            "interpretation": (
                "Excellent" if fidelity > 0.95 else
                "Good" if fidelity > 0.90 else
                "Acceptable" if fidelity > 0.80 else
                "Poor" if fidelity > 0.60 else
                "Very Poor"
            ),
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# SCORING TOOLS
# =============================================================================

def score_circuit(
    qasm_code: str,
    hardware_name: str | None = None,
) -> dict[str, Any]:
    """
    Comprehensive scoring of a quantum circuit.
    
    Args:
        qasm_code: QASM code to score
        hardware_name: Optional hardware for fitness scoring
    
    Returns:
        Comprehensive scores
    """
    try:
        parser = CircuitParser()
        dag = parser.parse_qasm(qasm_code)
        
        # Convert DAG to dict
        circuit_dict = dag.to_dict()
        
        # Complexity score
        complexity = score_complexity(circuit_dict)
        
        scores = {
            "success": True,
            "complexity": complexity,
        }
        
        # Hardware fitness if specified
        if hardware_name:
            hardware_profile = get_hardware_profile(hardware_name)
            if hardware_profile:
                fitness = score_hardware_fitness(circuit_dict, hardware_name)
                scores["hardware_fitness"] = fitness
                
                fidelity = estimate_fidelity(circuit_dict, hardware_profile)
                scores["estimated_fidelity"] = fidelity
        
        # Overall score
        base_score = complexity.get("normalized_score", 0.5)
        if hardware_name and "hardware_fitness" in scores:
            hw_score = scores["hardware_fitness"].get("overall_score", 0.5)
            fidelity_dict = scores.get("estimated_fidelity", {})
            fidelity_score = fidelity_dict.get("fidelity", 0.5) if isinstance(fidelity_dict, dict) else 0.5
            scores["overall_score"] = (base_score + hw_score + fidelity_score) / 3
        else:
            scores["overall_score"] = base_score
        
        return scores
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def compare_circuits(
    qasm_circuits: list[str],
    hardware_name: str | None = None,
) -> dict[str, Any]:
    """
    Compare multiple circuits and rank them.
    
    Args:
        qasm_circuits: List of QASM codes to compare
        hardware_name: Optional hardware for comparison
    
    Returns:
        Comparison results with ranking
    """
    try:
        results = []
        
        for i, qasm in enumerate(qasm_circuits):
            parser = CircuitParser()
            dag = parser.parse_qasm(qasm)
            circuit_dict = dag.to_dict()
            
            complexity = score_complexity(circuit_dict)
            
            result = {
                "index": i,
                "num_qubits": dag.num_qubits,
                "depth": dag.depth,
                "gate_count": dag.gate_count,
                "complexity": complexity,
            }
            
            if hardware_name:
                hardware_profile = get_hardware_profile(hardware_name)
                if hardware_profile:
                    result["hardware_fitness"] = score_hardware_fitness(circuit_dict, hardware_name)
                    result["estimated_fidelity"] = estimate_fidelity(circuit_dict, hardware_profile)
            
            results.append(result)
        
        # Rank by overall effectiveness
        def get_rank_score(r):
            score = r["complexity"].get("normalized_score", 0)
            if "hardware_fitness" in r:
                score += r["hardware_fitness"].get("overall_score", 0)
            if "estimated_fidelity" in r:
                score += r["estimated_fidelity"]
            return score
        
        results.sort(key=get_rank_score, reverse=True)
        
        for rank, r in enumerate(results, 1):
            r["rank"] = rank
        
        return {
            "success": True,
            "num_circuits": len(qasm_circuits),
            "rankings": results,
            "best_circuit_index": results[0]["index"] if results else None,
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# DOCUMENTATION TOOLS (Resources)
# =============================================================================

def get_gate_info(gate_name: str) -> dict[str, Any]:
    """
    Get detailed information about a quantum gate.        

    Args:
        gate_name: Name of the gate

    Returns:
        Gate documentation
    """
    try:
        result = get_gate_documentation(gate_name)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
def get_algorithm_info(algorithm_name: str) -> dict[str, Any]:
    """
    Get information about a quantum algorithm.

    Args:
        algorithm_name: Name of the algorithm

    Returns:
        Algorithm explanation
    """
    try:
        result = get_algorithm_explanation(algorithm_name)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
def list_available_hardware() -> dict[str, Any]:
    """
    List all available hardware profiles.

    Returns:
        List of hardware profiles
    """
    try:
        return {
            "success": True,
            "hardware": list_hardware_profiles(),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
def list_circuit_templates() -> dict[str, Any]:
    """
    List all available circuit templates.

    Returns:
        List of templates with descriptions
    """
    try:
        return {
            "success": True,
            "templates": [
                {
                    "name": "bell_state",
                    "category": "beginner",
                    "qubits": 2,
                    "description": "Creates maximally entangled Bell state",
                },
                {
                    "name": "ghz_state",
                    "category": "beginner",
                    "qubits": "variable",
                    "description": "Creates N-qubit GHZ state",
                },
                {
                    "name": "w_state",
                    "category": "beginner",
                    "qubits": "variable",
                    "description": "Creates N-qubit W state",
                },
                {
                    "name": "superposition",
                    "category": "beginner",
                    "qubits": "variable",
                    "description": "Creates uniform superposition",
                },
                {
                    "name": "qft",
                    "category": "algorithm",
                    "qubits": "variable",
                    "description": "Quantum Fourier Transform",
                },
                {
                    "name": "grover",
                    "category": "algorithm",
                    "qubits": "variable",
                    "description": "Grover's search algorithm",
                    "parameters": ["marked_states", "iterations"],
                },
                {
                    "name": "vqe",
                    "category": "variational",
                    "qubits": "variable",
                    "description": "VQE ansatz circuit",
                    "parameters": ["depth"],
                },
                {
                    "name": "qaoa",
                    "category": "variational",
                    "qubits": "variable",
                    "description": "QAOA circuit for optimization",
                    "parameters": ["edges", "num_layers"],
                },
            ],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
def get_learning_path(level: str) -> dict[str, Any]:
    """
    Get recommended learning resources for a skill level.

    Args:
        level: Skill level (beginner, intermediate, advanced, phd)

    Returns:
        Learning resources
    """
    return get_learning_resources(level)


def get_hardware_suggestions(circuit_qasm: str, top_n: int = 3) -> dict[str, Any]:
    """
    Get hardware recommendations based on circuit characteristics.
    
    Args:
        circuit_qasm: Circuit in OpenQASM format
        top_n: Number of top recommendations to return
    
    Returns:
        Hardware recommendations with scores
    """
    try:
        # Parse the circuit
        dag = CircuitParser.parse_qasm(circuit_qasm)
        circuit_dict = dag.to_dict()
        
        # Score hardware fitness for all available hardware
        from ..plugins.scoring.hardware_fitness import score_hardware_fitness
        
        hardware_list = list_hardware_profiles()
        suggestions = []
        
        for hw_data in hardware_list:
            hw_id = hw_data.get('id')
            try:
                fitness = score_hardware_fitness(circuit_dict, hw_id)
                if isinstance(fitness, dict) and 'fitness_score' in fitness:
                    suggestions.append({
                        'hardware_id': hw_id,
                        'hardware_name': hw_data.get('name', hw_id),
                        'num_qubits': hw_data.get('num_qubits'),
                        'fitness_score': fitness.get('fitness_score', 0),
                        'reasoning': fitness.get('explanation', ''),
                        'details': fitness
                    })
            except Exception:
                pass
        
        # Sort by fitness score and return top_n
        suggestions.sort(key=lambda x: x.get('fitness_score', 0), reverse=True)
        top_suggestions = suggestions[:top_n]
        
        return {
            "success": True,
            "recommendations": top_suggestions,
            "total_evaluated": len(suggestions)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_circuits_complexity(circuits: list[dict]) -> dict[str, Any]:
    """
    Compare complexity of multiple circuits.
    
    Args:
        circuits: List of circuit dicts with keys:
                 - qasm: OpenQASM string
                 - name: Circuit name (optional)
    
    Returns:
        Comparison of circuit complexities
    """
    try:
        from ..plugins.scoring.complexity_score import compare_circuits_complexity as compare_func
        
        parsed_circuits = []
        
        for circuit_input in circuits:
            qasm = circuit_input.get('qasm') or circuit_input.get('code')
            name = circuit_input.get('name', 'Unknown')
            dag = CircuitParser.parse_qasm(qasm)
            circuit_dict = dag.to_dict()
            parsed_circuits.append({
                'name': name,
                'circuit': circuit_dict,
                'qasm': qasm
            })
        
        comparison = compare_func([c['circuit'] for c in parsed_circuits])
        
        # Enrich with circuit names
        result = {
            "success": True,
            "comparison": comparison,
            "circuits": [
                {
                    "name": pc['name'],
                    "qasm_preview": pc['qasm'][:200] + "..." if len(pc['qasm']) > 200 else pc['qasm']
                }
                for pc in parsed_circuits
            ]
        }
        
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def compute_circuit_unitary(circuit_qasm: str) -> dict[str, Any]:
    """
    Compute the full unitary matrix of a circuit.
    
    Args:
        circuit_qasm: Circuit in OpenQASM format
    
    Returns:
        Unitary matrix and metadata
    """
    try:
        from ..plugins.validation.unitary_check import compute_circuit_unitary as compute_func
        
        dag = CircuitParser.parse_qasm(circuit_qasm)
        circuit_dict = dag.to_dict()
        
        unitary_result = compute_func(circuit_dict)
        
        # Format unitary as serializable list
        if isinstance(unitary_result, dict):
            unitary_matrix = unitary_result.get('unitary')
        else:
            unitary_matrix = unitary_result
        
        if unitary_matrix is not None:
            import numpy as np
            # Convert numpy array to nested lists
            if hasattr(unitary_matrix, 'tolist'):
                unitary_list = unitary_matrix.tolist()
            else:
                unitary_list = [[complex(x) for x in row] for row in unitary_matrix]
        else:
            unitary_list = None
        
        return {
            "success": True,
            "unitary_matrix": unitary_list,
            "dimension": len(unitary_list) if unitary_list else 0,
            "is_unitary": unitary_result.get('is_unitary', True) if isinstance(unitary_result, dict) else True
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def estimate_required_shots(circuit_qasm: str, target_std_error: float = 0.01) -> dict[str, Any]:
    """
    Estimate required number of shots for statistical significance.
    
    Args:
        circuit_qasm: Circuit in OpenQASM format
        target_std_error: Target standard error for measurement results
    
    Returns:
        Required shots and recommendations
    """
    try:
        from ..plugins.evaluation.noise_estimator import estimate_required_shots as estimate_func
        
        dag = CircuitParser.parse_qasm(circuit_qasm)
        circuit_dict = dag.to_dict()
        
        required_shots = estimate_func(circuit_dict, target_std_error)
        
        # Handle if result is a dict
        if isinstance(required_shots, dict):
            required_shots = required_shots.get('required_shots', 100)
        
        required_shots = int(required_shots) if required_shots else 100
        
        # Generate recommendations
        recommendations = []
        if required_shots < 100:
            recommendations.append("Moderate shot count recommended for good statistics")
        elif required_shots < 10000:
            recommendations.append("High shot count needed; consider longer runs")
        else:
            recommendations.append("Very high shot count required; consider circuit simplification")
        
        return {
            "success": True,
            "required_shots": int(required_shots),
            "target_std_error": target_std_error,
            "recommendations": recommendations,
            "shot_recommendations": {
                "minimum": int(required_shots * 0.8),
                "recommended": int(required_shots),
                "high_confidence": int(required_shots * 1.5)
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_entanglement_structure(circuit_qasm: str) -> dict[str, Any]:
    """
    Analyze the entanglement structure of a circuit.
    
    Args:
        circuit_qasm: Circuit in OpenQASM format
    
    Returns:
        Detailed entanglement analysis
    """
    try:
        from ..plugins.validation.unitary_check import analyze_entanglement_structure as analyze_func
        
        dag = CircuitParser.parse_qasm(circuit_qasm)
        circuit_dict = dag.to_dict()
        
        entanglement_info = analyze_func(circuit_dict)
        
        return {
            "success": True,
            "entanglement_analysis": entanglement_info
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def compose_circuits(circuit1_qasm: str, circuit2_qasm: str, 
                     qubit_mapping: dict[int, int] | None = None) -> dict[str, Any]:
    """
    Compose two circuits sequentially (append circuit2 after circuit1).
    
    Args:
        circuit1_qasm: First circuit in OpenQASM format
        circuit2_qasm: Second circuit in OpenQASM format
        qubit_mapping: Optional mapping from circuit2's qubits to circuit1's qubits.
                       If None, identity mapping is used.
    
    Returns:
        Composed circuit in OpenQASM format
    """
    try:
        dag1 = CircuitParser.parse_qasm(circuit1_qasm)
        dag2 = CircuitParser.parse_qasm(circuit2_qasm)
        
        composed = dag1.compose(dag2, qubit_mapping)
        
        return {
            "success": True,
            "composed_qasm": composed.to_qasm(),
            "composed_circuit": composed.to_dict(),
            "num_qubits": composed.num_qubits,
            "depth": composed.depth,
            "gate_count": composed.gate_count
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_inverse_circuit(circuit_qasm: str) -> dict[str, Any]:
    """
    Generate the inverse (adjoint/dagger) of a circuit.
    
    The inverse circuit reverses all gates and applies conjugate transpose
    to each gate. This is useful for:
    - Uncomputation in quantum algorithms
    - Implementing controlled versions of circuits
    - Verifying unitarity (circuit + inverse = identity)
    
    Args:
        circuit_qasm: Circuit in OpenQASM format
    
    Returns:
        Inverse circuit in OpenQASM format
    
    Note:
        Measurements and resets are NOT included in the inverse
        as they are not unitary operations.
    """
    try:
        dag = CircuitParser.parse_qasm(circuit_qasm)
        inverse_dag = dag.inverse()
        
        return {
            "success": True,
            "inverse_qasm": inverse_dag.to_qasm(),
            "inverse_circuit": inverse_dag.to_dict(),
            "original_gate_count": dag.gate_count,
            "inverse_gate_count": inverse_dag.gate_count,
            "notes": [
                "Gate order is reversed",
                "Parameterized gates have negated angles",
                "S -> Sdg, T -> Tdg conversions applied",
                "Measurements and resets are excluded"
            ]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def tensor_circuits(circuit1_qasm: str, circuit2_qasm: str) -> dict[str, Any]:
    """
    Tensor product of two circuits (parallel composition).
    
    Places circuit2's qubits after circuit1's qubits, allowing
    both circuits to run in parallel on separate qubit registers.
    
    Args:
        circuit1_qasm: First circuit in OpenQASM format
        circuit2_qasm: Second circuit in OpenQASM format
    
    Returns:
        Tensored circuit in OpenQASM format
    """
    try:
        dag1 = CircuitParser.parse_qasm(circuit1_qasm)
        dag2 = CircuitParser.parse_qasm(circuit2_qasm)
        
        tensored = dag1.tensor(dag2)
        
        return {
            "success": True,
            "tensored_qasm": tensored.to_qasm(),
            "tensored_circuit": tensored.to_dict(),
            "num_qubits": tensored.num_qubits,
            "circuit1_qubits": list(range(dag1.num_qubits)),
            "circuit2_qubits": list(range(dag1.num_qubits, tensored.num_qubits)),
            "depth": tensored.depth
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def repeat_circuit(circuit_qasm: str, repetitions: int) -> dict[str, Any]:
    """
    Repeat a circuit multiple times (sequential self-composition).
    
    Useful for:
    - Grover iterations
    - Trotter-Suzuki decomposition steps
    - Variational circuit layers
    
    Args:
        circuit_qasm: Circuit in OpenQASM format
        repetitions: Number of times to repeat the circuit
    
    Returns:
        Repeated circuit in OpenQASM format
    """
    try:
        if repetitions <= 0:
            return {"success": False, "error": "Repetitions must be positive"}
        if repetitions > 100:
            return {"success": False, "error": "Maximum 100 repetitions allowed"}
        
        dag = CircuitParser.parse_qasm(circuit_qasm)
        repeated = dag.repeat(repetitions)
        
        return {
            "success": True,
            "repeated_qasm": repeated.to_qasm(),
            "repeated_circuit": repeated.to_dict(),
            "original_depth": dag.depth,
            "repeated_depth": repeated.depth,
            "original_gate_count": dag.gate_count,
            "repeated_gate_count": repeated.gate_count,
            "repetitions": repetitions
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
