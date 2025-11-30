"""
made with gradio 6 https://github.com/gradio-app/gradio 
QuantumArchitect-MCP: Quantum Circuit Creation & Validation Platform
Main Gradio 6 application with MCP endpoints for HuggingFace Spaces.

Inspired by IBM Quantum Composer - Visual circuit builder with interactive gates.

This app exposes quantum circuit tools via MCP protocol for AI agents.

NOTE: This is the main entry point. All components are modularized in frontend/:
- frontend/mcp_tools.py: MCP protocol wrappers
- frontend/ui_handlers.py: Gradio UI event handlers
- frontend/circuits.py: Predefined test circuits
- frontend/helpers.py: Utility functions
- frontend/visualizations.py: SVG and Plotly visualizations
- frontend/styles.py: CSS styling
- frontend/constants.py: Gate library and templates
"""

import gradio as gr
import json

# Import all frontend components from modular structure
from frontend import (
    # Constants
    GATE_LIBRARY,
    GATE_CATEGORIES,
    TEMPLATE_PARAMS,
    # Styles
    IBM_COMPOSER_CSS,
    # Visualizations
    render_visual_circuit,
    render_qsphere_svg,
    # MCP Tools
    mcp_get_gate_info,
    mcp_get_algorithm_info,
    # UI Handlers
    ui_create_circuit,
    ui_validate_circuit,
    ui_simulate_circuit,
    ui_score_circuit,
    ui_simulate_visual,
    # Circuits
    PREDEFINED_CIRCUITS,
    load_test_circuit,
    # Helpers
    EXAMPLE_QASM,
    get_template_params,
    get_template_info,
    add_gate_to_json,
    clear_gates,
    make_gate_handler,
    clear_circuit_handler,
    undo_handler,
    # Tab modules
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


# =============================================================================
# BUILD GRADIO APP WITH MCP - IBM COMPOSER STYLE (COMPACT)
# =============================================================================

with gr.Blocks(
    title="QuantumArchitect-MCP",
) as demo:
    
    gr.Markdown("""
    <div style="text-align: center; padding: 10px 0;">
        <h1 style="
            font-size: 2em;
            background: linear-gradient(135deg, #4fc3f7 0%, #7c4dff 50%, #ff4081 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 6px;
        ">🔮 QuantumArchitect-MCP</h1>
        <p style="color: #8b949e; font-size: 0.95em; margin: 0;">
            Visual Quantum Circuit Builder • Inspired by IBM Quantum Composer
        </p>
        <div style="display: flex; justify-content: center; gap: 12px; margin-top: 8px;">
            <span style="
                background: #21262d;
                padding: 4px 10px;
                border-radius: 16px;
                font-size: 0.75em;
                color: #58a6ff;
                border: 1px solid #30363d;
            ">🎯 MCP Enabled</span>
            <span style="
                background: #21262d;
                padding: 4px 10px;
                border-radius: 16px;
                font-size: 0.75em;
                color: #3fb950;
                border: 1px solid #30363d;
            ">⚡ Qiskit Backend</span>
            <span style="
                background: #21262d;
                padding: 4px 10px;
                border-radius: 16px;
                font-size: 0.75em;
                color: #d29922;
                border: 1px solid #30363d;
            ">🔬 Real-time Sim</span>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        add_circuit_builder_tab()
        add_templates_tab()
        add_validate_tab()
        add_simulate_tab()
        add_score_tab()
        add_getting_started_tab()
        add_dirac_notation_study_tab()
        add_measurement_study_tab()
        add_grover_study_tab()
        add_vqe_study_tab()
        add_transpilation_study_tab()
        add_bell_state_study_tab()
        add_ghz_state_study_tab()
        add_qft_study_tab()
        add_reference_tab()
        add_mcp_endpoints_tab()
    
    # Quick Gate Reference (compact) - collapsed by default
    with gr.Accordion("🔧 Gate Reference", open=False):
        gr.Markdown(r"""
        | Gate | Qubits | Effect |
        |------|--------|--------|
        | **H** | 1 | Superposition |
        | **X,Y,Z** | 1 | Pauli rotations |
        | **CX** | 2 | CNOT |
        | **Rx,Ry,Rz** | 1 | Parameterized rotations |
        | **CCX** | 3 | Toffoli |
        """)


# Launch the Gradio app     
if __name__ == "__main__":  
    demo.launch(
        server_name="0.0.0.0",
        show_error=True,
        debug=True,
        share=False,
        mcp_server=True,
        css=IBM_COMPOSER_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="green",
            neutral_hue="slate",
        ),
    )