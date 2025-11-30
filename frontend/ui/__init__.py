"""
UI components for QuantumArchitect-MCP.
Contains styles, visualizations, UI handlers, and MCP tools.
"""

from .styles import IBM_COMPOSER_CSS
from .visualizations import (
    render_visual_circuit,
    render_qsphere_svg,
    render_probability_bars,
    render_statevector_amplitudes,
    plot_bloch_sphere_plotly,
    create_placeholder_plot,
)
from .ui_handlers import (
    ui_create_circuit,
    ui_validate_circuit,
    ui_simulate_circuit,
    ui_score_circuit,
    ui_simulate_visual,
)
from .mcp_tools import (
    mcp_get_gate_info,
    mcp_get_algorithm_info,
)

__all__ = [
    # Styles
    "IBM_COMPOSER_CSS",
    # Visualizations
    "render_visual_circuit",
    "render_qsphere_svg",
    "render_probability_bars",
    "render_statevector_amplitudes",
    "plot_bloch_sphere_plotly",
    "create_placeholder_plot",
    # UI Handlers
    "ui_create_circuit",
    "ui_validate_circuit",
    "ui_simulate_circuit",
    "ui_score_circuit",
    "ui_simulate_visual",
    # MCP Tools
    "mcp_get_gate_info",
    "mcp_get_algorithm_info",
]
