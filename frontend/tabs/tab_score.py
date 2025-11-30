"""
Score Tab for QuantumArchitect-MCP
"""

import gradio as gr

# Import required components
from ..core import EXAMPLE_QASM
from ..ui import ui_score_circuit


def add_score_tab():
    """Add the Score tab to the Gradio interface."""
    with gr.TabItem("📊 Score"):
        gr.Markdown("### Analyze circuit complexity")

        with gr.Row():
            with gr.Column():
                score_qasm = gr.Code(value=EXAMPLE_QASM, language="python", label="QASM", lines=10)
                score_hardware = gr.Dropdown(
                    choices=["", "ibm_brisbane", "rigetti_aspen", "ionq_harmony"],
                    value="ibm_brisbane",
                    label="Hardware"
                )
                score_btn = gr.Button("📊 Score", variant="primary", size="lg")

            with gr.Column():
                score_output = gr.Markdown(value="*Click Score*")

        score_btn.click(ui_score_circuit, inputs=[score_qasm, score_hardware], outputs=[score_output])