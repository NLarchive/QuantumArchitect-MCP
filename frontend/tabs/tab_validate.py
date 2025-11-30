"""
Validate Tab for QuantumArchitect-MCP
"""

import gradio as gr

# Import required components
from ..core import EXAMPLE_QASM
from ..ui import ui_validate_circuit


def add_validate_tab():
    """Add the Validate tab to the Gradio interface."""
    with gr.TabItem("✅ Validate"):
        gr.Markdown("### Validate quantum circuits")

        with gr.Row():
            with gr.Column():
                val_qasm = gr.Code(value=EXAMPLE_QASM, language="python", label="QASM", lines=10)
                val_hardware = gr.Dropdown(
                    choices=["", "ibm_brisbane", "ibm_sherbrooke", "rigetti_aspen", "ionq_harmony"],
                    value="",
                    label="Hardware (optional)"
                )
                validate_btn = gr.Button("✅ Validate", variant="primary", size="lg")

            with gr.Column():
                val_status = gr.Markdown(value="*Click Validate*")
                val_details = gr.Markdown(value="")

        validate_btn.click(ui_validate_circuit, inputs=[val_qasm, val_hardware], outputs=[val_status, val_details])