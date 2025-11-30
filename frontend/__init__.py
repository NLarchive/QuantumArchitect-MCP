# =============================================================================
# Frontend Package - Modular UI Components for QuantumArchitect-MCP
# =============================================================================
"""
This package contains all frontend components for the Gradio UI organized into subfolders:

- core/: Constants, circuits, and helper functions
- ui/: Styles, visualizations, UI handlers, and MCP tools  
- tabs/: Individual tab modules for the Gradio interface
"""

# Import from core subfolder
from .core import (
    GATE_LIBRARY,
    GATE_CATEGORIES,
    TEMPLATE_PARAMS,
    PREDEFINED_CIRCUITS,
    load_test_circuit,
    EXAMPLE_QASM,
    get_template_params,
    get_template_info,
    add_gate_to_json,
    clear_gates,
    make_gate_handler,
    clear_circuit_handler,
    undo_handler,
)

# Import from ui subfolder
from .ui import (
    IBM_COMPOSER_CSS,
    render_visual_circuit,
    render_qsphere_svg,
    render_probability_bars,
    render_statevector_amplitudes,
    plot_bloch_sphere_plotly,
    ui_create_circuit,
    ui_validate_circuit,
    ui_simulate_circuit,
    ui_score_circuit,
    ui_simulate_visual,
    mcp_get_gate_info,
    mcp_get_algorithm_info,
)

# Import from tabs subfolder
from .tabs import (
    add_circuit_builder_tab,
    add_templates_tab,
    add_validate_tab,
    add_simulate_tab,
    add_score_tab,
    add_getting_started_tab,
    add_bell_state_study_tab,
    add_ghz_state_study_tab,
    add_qft_study_tab,
    add_reference_tab,
    add_mcp_endpoints_tab,
    add_dirac_notation_study_tab,
    add_measurement_study_tab,
    add_grover_study_tab,
    add_vqe_study_tab,
    add_transpilation_study_tab,
)

# For backwards compatibility, import additional items from ui
from .ui.visualizations import (
    render_bloch_sphere_svg,
    render_gate_palette,
    render_bloch_sphere_placeholder,
)

from .ui.mcp_tools import (
    mcp_create_circuit,
    mcp_parse_qasm,
    mcp_build_circuit,
    mcp_validate_circuit,
    mcp_check_hardware,
    mcp_simulate,
    mcp_get_statevector,
    mcp_estimate_fidelity,
    mcp_score_circuit,
    mcp_compare_circuits,
    mcp_list_hardware,
    mcp_list_templates,
    mcp_get_learning_path,
)

from .core.circuits import (
    get_predefined_circuit,
    render_predefined_circuit_card,
)

__all__ = [
    # Constants
    "GATE_LIBRARY",
    "GATE_CATEGORIES", 
    "TEMPLATE_PARAMS",
    # Styles
    "IBM_COMPOSER_CSS",
    # Visualizations
    "render_bloch_sphere_svg",
    "render_qsphere_svg",
    "render_statevector_amplitudes",
    "render_visual_circuit",
    "render_gate_palette",
    "render_bloch_sphere_placeholder",
    "render_probability_bars",
    "plot_bloch_sphere_plotly",
    # MCP Tools
    "mcp_create_circuit",
    "mcp_parse_qasm",
    "mcp_build_circuit",
    "mcp_validate_circuit",
    "mcp_check_hardware",
    "mcp_simulate",
    "mcp_get_statevector",
    "mcp_estimate_fidelity",
    "mcp_score_circuit",
    "mcp_compare_circuits",
    "mcp_get_gate_info",
    "mcp_get_algorithm_info",
    "mcp_list_hardware",
    "mcp_list_templates",
    "mcp_get_learning_path",
    # UI Handlers
    "ui_create_circuit",
    "ui_validate_circuit",
    "ui_simulate_circuit",
    "ui_score_circuit",
    "ui_simulate_visual",
    # Circuits
    "PREDEFINED_CIRCUITS",
    "get_predefined_circuit",
    "render_predefined_circuit_card",
    "load_test_circuit",
    # Helpers
    "EXAMPLE_QASM",
    "get_template_params",
    "get_template_info",
    "add_gate_to_json",
    "clear_gates",
    "make_gate_handler",
    "clear_circuit_handler",
    "undo_handler",
]
