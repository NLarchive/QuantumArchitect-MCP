"""
Tab modules for QuantumArchitect-MCP interface.
Each module contains a function to add a specific tab to the Gradio interface.
"""

from .tab_circuit_builder import add_circuit_builder_tab
from .tab_templates import add_templates_tab
from .tab_validate import add_validate_tab
from .tab_simulate import add_simulate_tab
from .tab_score import add_score_tab
from .tab_getting_started import add_getting_started_tab
from .tab_bell_state_study import add_bell_state_study_tab
from .tab_ghz_state_study import add_ghz_state_study_tab
from .tab_qft_study import add_qft_study_tab
from .tab_reference import add_reference_tab
from .tab_mcp_endpoints import add_mcp_endpoints_tab
from .tab_dirac_notation_study import add_dirac_notation_study_tab
from .tab_measurement_study import add_measurement_study_tab
from .tab_grover_study import add_grover_study_tab
from .tab_vqe_study import add_vqe_study_tab
from .tab_transpilation_study import add_transpilation_study_tab

__all__ = [
    "add_circuit_builder_tab",
    "add_templates_tab",
    "add_validate_tab",
    "add_simulate_tab",
    "add_score_tab",
    "add_getting_started_tab",
    "add_bell_state_study_tab",
    "add_ghz_state_study_tab",
    "add_qft_study_tab",
    "add_reference_tab",
    "add_mcp_endpoints_tab",
    "add_dirac_notation_study_tab",
    "add_measurement_study_tab",
    "add_grover_study_tab",
    "add_vqe_study_tab",
    "add_transpilation_study_tab",
]
