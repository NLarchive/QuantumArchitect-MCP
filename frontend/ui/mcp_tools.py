# =============================================================================
# MCP TOOL FUNCTIONS - Exposed to AI Agents
# =============================================================================
"""
MCP (Model Context Protocol) tool functions for quantum circuit operations.
These functions wrap the backend handlers and expose them for AI agent use.
"""

import json

from src.mcp_server import (
    # Creation tools
    create_circuit_from_template,
    parse_qasm_circuit,
    build_custom_circuit,
    # Validation tools
    validate_circuit,
    check_hardware_compatibility,
    # Simulation tools
    simulate_circuit,
    get_statevector,
    estimate_circuit_fidelity,
    # Scoring tools
    score_circuit,
    compare_circuits,
    # Documentation tools
    get_gate_info,
    get_algorithm_info,
    list_available_hardware,
    list_circuit_templates,
    get_learning_path,
)


def mcp_create_circuit(
    template_name: str,
    num_qubits: int = 2,
    parameters_json: str = "{}",
) -> str:
    """
    Create a quantum circuit from a predefined template.
    
    Args:
        template_name: Template name (bell_state, ghz_state, qft, grover, vqe, qaoa)
        num_qubits: Number of qubits (default: 2)
        parameters_json: JSON string of additional parameters
    
    Returns:
        JSON string with created circuit
    """
    try:
        params = json.loads(parameters_json) if parameters_json and parameters_json.strip() else {}
    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON in parameters: {e}"
        }, indent=2)

    result = create_circuit_from_template(template_name, num_qubits, params)
    return json.dumps(result, indent=2, default=str)
def mcp_parse_qasm(qasm_code: str, qasm_version: str = "2.0") -> str:
    """
    Parse OpenQASM code into internal representation.
    
    Args:
        qasm_code: OpenQASM 2.0 or 3.0 code string
        qasm_version: QASM version ("2.0" or "3.0")
    
    Returns:
        JSON string with parsed circuit
    """
    result = parse_qasm_circuit(qasm_code, qasm_version)
    return json.dumps(result, indent=2, default=str)


def mcp_build_circuit(
    num_qubits: int,
    gates_json: str,
    measurements_json: str = "[]",
) -> str:
    """
    Build a custom quantum circuit from gate specifications.
    
    Args:
        num_qubits: Number of qubits
        gates_json: JSON array of gates [{"name": "h", "qubits": [0]}, ...]
        measurements_json: JSON array of measurements [{"qubit": 0, "clbit": 0}, ...]
    
    Returns:
        JSON string with built circuit
    """
    try:
        gates = json.loads(gates_json)
        measurements = json.loads(measurements_json) if measurements_json else []
    except json.JSONDecodeError as e:
        return json.dumps({"success": False, "error": f"JSON parse error: {e}"})
    
    result = build_custom_circuit(num_qubits, gates, measurements)
    return json.dumps(result, indent=2, default=str)


def mcp_validate_circuit(
    qasm_code: str,
    hardware_target: str = "",
    check_connectivity: bool = True,
    check_unitary: bool = True,
) -> str:
    """
    Validate a quantum circuit for syntax, connectivity, and unitarity.
    
    Args:
        qasm_code: OpenQASM code to validate
        hardware_target: Target hardware name for connectivity check
        check_connectivity: Whether to check hardware connectivity
        check_unitary: Whether to verify unitarity
    
    Returns:
        JSON string with validation results
    """
    result = validate_circuit(
        qasm_code=qasm_code,
        hardware_target=hardware_target if hardware_target else None,
        check_connectivity=check_connectivity,
        check_unitary=check_unitary,
    )
    return json.dumps(result, indent=2, default=str)


def mcp_check_hardware(qasm_code: str, hardware_name: str) -> str:
    """
    Check circuit compatibility with specific quantum hardware.
    
    Args:
        qasm_code: OpenQASM code
        hardware_name: Hardware profile name (e.g., "ibm_eagle", "rigetti_aspen")
    
    Returns:
        JSON string with compatibility analysis
    """
    result = check_hardware_compatibility(qasm_code, hardware_name)
    return json.dumps(result, indent=2, default=str)


def mcp_simulate(
    qasm_code: str,
    shots: int = 1024,
    include_statevector: bool = True,
    noise_model: str = "",
) -> str:
    """
    Simulate a quantum circuit and return measurement results.
    
    Args:
        qasm_code: OpenQASM code to simulate
        shots: Number of measurement shots
        include_statevector: Include full statevector in results
        noise_model: Hardware name for noise modeling (optional)
    
    Returns:
        JSON string with simulation results
    """
    result = simulate_circuit(
        qasm_code=qasm_code,
        shots=shots,
        include_statevector=include_statevector,
        noise_model=noise_model if noise_model else None,
    )
    return json.dumps(result, indent=2, default=str)


def mcp_get_statevector(qasm_code: str) -> str:
    """
    Get the ideal statevector of a quantum circuit.
    
    Args:
        qasm_code: OpenQASM code
    
    Returns:
        JSON string with statevector data
    """
    result = get_statevector(qasm_code)
    return json.dumps(result, indent=2, default=str)


def mcp_estimate_fidelity(qasm_code: str, hardware_name: str) -> str:
    """
    Estimate circuit fidelity on specific hardware.
    
    Args:
        qasm_code: OpenQASM code
        hardware_name: Target hardware name
    
    Returns:
        JSON string with fidelity estimation
    """
    result = estimate_circuit_fidelity(qasm_code, hardware_name)
    return json.dumps(result, indent=2, default=str)


def mcp_score_circuit(qasm_code: str, hardware_name: str = "") -> str:
    """
    Get comprehensive scoring for a quantum circuit.
    
    Args:
        qasm_code: OpenQASM code
        hardware_name: Target hardware for fitness scoring (optional)
    
    Returns:
        JSON string with complexity and fitness scores
    """
    result = score_circuit(
        qasm_code=qasm_code,
        hardware_name=hardware_name if hardware_name else None,
    )
    return json.dumps(result, indent=2, default=str)


def mcp_compare_circuits(circuits_json: str, hardware_name: str = "") -> str:
    """
    Compare multiple circuits and rank them by effectiveness.
    
    Args:
        circuits_json: JSON array of QASM code strings
        hardware_name: Target hardware for comparison (optional)
    
    Returns:
        JSON string with comparison and rankings
    """
    try:
        circuits = json.loads(circuits_json)
    except json.JSONDecodeError as e:
        return json.dumps({"success": False, "error": f"JSON parse error: {e}"})
    
    result = compare_circuits(
        qasm_circuits=circuits,
        hardware_name=hardware_name if hardware_name else None,
    )
    return json.dumps(result, indent=2, default=str)


def mcp_get_gate_info(gate_name: str) -> str:
    """
    Get documentation for a quantum gate.
    
    Args:
        gate_name: Gate name (h, x, y, z, cx, rx, ry, rz, etc.)
    
    Returns:
        JSON string with gate documentation
    """
    result = get_gate_info(gate_name)
    return json.dumps(result, indent=2, default=str)


def mcp_get_algorithm_info(algorithm_name: str) -> str:
    """
    Get explanation of a quantum algorithm.
    
    Args:
        algorithm_name: Algorithm name (bell_state, qft, grover, vqe, qaoa, etc.)
    
    Returns:
        JSON string with algorithm explanation
    """
    result = get_algorithm_info(algorithm_name)
    return json.dumps(result, indent=2, default=str)


def mcp_list_hardware() -> str:
    """
    List all available hardware profiles.
    
    Returns:
        JSON string with hardware profiles
    """
    result = list_available_hardware()
    return json.dumps(result, indent=2, default=str)


def mcp_list_templates() -> str:
    """
    List all available circuit templates.
    
    Returns:
        JSON string with template information
    """
    result = list_circuit_templates()
    return json.dumps(result, indent=2, default=str)


def mcp_get_learning_path(level: str) -> str:
    """
    Get recommended learning resources for a skill level.
    
    Args:
        level: Skill level (beginner, intermediate, advanced, phd)
    
    Returns:
        JSON string with learning resources
    """
    result = get_learning_path(level)
    return json.dumps(result, indent=2, default=str)
