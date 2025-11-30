"""
Simulate Tab for QuantumArchitect-MCP
"""

import gradio as gr

# Import required components
from ..core import EXAMPLE_QASM
from ..ui import ui_simulate_circuit


def add_simulate_tab():
    """Add the Simulate tab to the Gradio interface."""
    with gr.TabItem("🎲 Simulate"):
        gr.Markdown("### Simulate circuits")

        with gr.Row():
            with gr.Column():
                sim_qasm = gr.Code(value=EXAMPLE_QASM, language="python", label="QASM", lines=10)
                sim_shots = gr.Slider(minimum=100, maximum=10000, value=1024, step=100, label="Shots")
                simulate_btn = gr.Button("🎲 Simulate", variant="primary", size="lg")

            with gr.Column():
                sim_counts = gr.Markdown(value="*Click Simulate*")
                sim_probs = gr.Markdown(value="")

        simulate_btn.click(ui_simulate_circuit, inputs=[sim_qasm, sim_shots], outputs=[sim_counts, sim_probs])