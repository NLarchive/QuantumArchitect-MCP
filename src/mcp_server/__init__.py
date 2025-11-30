"""
MCP Server Module Initialization.
Exports all MCP tools and resources.
"""

from .schemas import (
    GateSchema,
    CircuitSchema,
    QASMInput,
    HardwareTarget,
    SimulationRequest,
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

from .endpoint_handlers import (
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

__all__ = [
    # Schemas
    "GateSchema",
    "CircuitSchema",
    "QASMInput",
    "HardwareTarget",
    "SimulationRequest",
    "ValidationResponse",
    "SimulationResponse",
    "ScoreResponse",
    # Context providers
    "get_hardware_profile",
    "list_hardware_profiles",
    "get_reference_circuit",
    "list_reference_circuits",
    "get_gate_documentation",
    "get_algorithm_explanation",
    "get_learning_resources",
    # Creation tools
    "create_circuit_from_template",
    "parse_qasm_circuit",
    "build_custom_circuit",
    # Validation tools
    "validate_circuit",
    "check_hardware_compatibility",
    # Simulation tools
    "simulate_circuit",
    "get_statevector",
    "estimate_circuit_fidelity",
    # Scoring tools
    "score_circuit",
    "compare_circuits",
    # Documentation tools
    "get_gate_info",
    "get_algorithm_info",
    "list_available_hardware",
    "list_circuit_templates",
    "get_learning_path",
]
