"""
Templates Tab for QuantumArchitect-MCP
"""

import gradio as gr

# Import required components
from ..core import (
    TEMPLATE_PARAMS,
    get_template_info,
    get_template_params,
)
from ..ui import ui_create_circuit


def add_templates_tab():
    """Add the Templates tab to the Gradio interface."""
    with gr.TabItem("🛠️ Templates"):
        gr.Markdown("### Create circuits from templates")

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### 📋 Template")
                    template_dropdown = gr.Dropdown(
                        choices=list(TEMPLATE_PARAMS.keys()),
                        value="bell_state",
                        label="Template"
                    )
                    template_info = gr.Markdown(value=get_template_info("bell_state"))

                with gr.Group():
                    gr.Markdown("#### 🔢 Config")
                    qubits_slider = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Qubits")
                    params_input = gr.Textbox(value="{}", label="Params (JSON)", lines=2)

                create_btn = gr.Button("🚀 Create", variant="primary", size="lg")

            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("#### 📤 Output")
                    qasm_output = gr.Code(language="python", label="QASM", lines=8)
                    viz_output = gr.Textbox(label="Diagram", lines=4)
                    summary_output = gr.JSON(label="Summary")

        template_dropdown.change(get_template_info, inputs=[template_dropdown], outputs=[template_info])
        template_dropdown.change(get_template_params, inputs=[template_dropdown], outputs=[params_input])
        create_btn.click(ui_create_circuit, inputs=[template_dropdown, qubits_slider, params_input], outputs=[qasm_output, viz_output, summary_output])