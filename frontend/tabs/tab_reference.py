"""
Reference Tab for QuantumArchitect-MCP
"""

import gradio as gr
import json

# Import required components
from ..ui import (
    mcp_get_gate_info,
    mcp_get_algorithm_info,
)


def add_reference_tab():
    """Add the Reference tab to the Gradio interface."""
    with gr.TabItem("📚 Reference", id="reference"):
        gr.Markdown("### Explore quantum gates and algorithms")

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 🔲 Gate Reference")
                    gr.Markdown("*Click a gate to learn more:*")
                    with gr.Row():
                        gate_btn_h = gr.Button("H", size="sm", elem_id="gate_btn_h")
                        gate_btn_x = gr.Button("X", size="sm", elem_id="gate_btn_x")
                        gate_btn_y = gr.Button("Y", size="sm", elem_id="gate_btn_y")
                        gate_btn_z = gr.Button("Z", size="sm", elem_id="gate_btn_z")
                        gate_btn_cx = gr.Button("CX", size="sm", elem_id="gate_btn_cx")
                        gate_btn_rx = gr.Button("RX", size="sm", elem_id="gate_btn_rx")

                    gate_input = gr.Textbox(
                        value="h",
                        label="Gate Name",
                        elem_id="gate_name_input",
                        placeholder="h, x, cx, rx, ...",
                        info="Enter gate name or click above"
                    )
                    gate_btn = gr.Button("🔍 Get Gate Info", variant="primary")
                    gate_output = gr.JSON(label="Gate Documentation", elem_id="gate_docs")

            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 🧮 Algorithm Reference")
                    gr.Markdown("*Click an algorithm to learn more:*")
                    with gr.Row():
                        algo_btn_bell = gr.Button("Bell State", size="sm", elem_id="algo_btn_bell_state")
                        algo_btn_qft = gr.Button("QFT", size="sm", elem_id="algo_btn_qft")
                        algo_btn_grover = gr.Button("Grover", size="sm", elem_id="algo_btn_grover")

                    algo_input = gr.Textbox(
                        value="bell_state",
                        label="Algorithm Name",
                        elem_id="algo_name_input",
                        placeholder="bell_state, qft, grover, ...",
                        info="Enter algorithm name or click above"
                    )
                    algo_btn = gr.Button("🔍 Get Algorithm Info", variant="primary")
                    algo_output = gr.JSON(label="Algorithm Documentation", elem_id="algo_docs")

        # Connect gate buttons to auto-fill input and fetch info
        def gate_click_handler(gate_name):
            info = json.loads(mcp_get_gate_info(gate_name))
            return gate_name, info

        gate_btn_h.click(lambda: gate_click_handler("h"), outputs=[gate_input, gate_output])
        gate_btn_x.click(lambda: gate_click_handler("x"), outputs=[gate_input, gate_output])
        gate_btn_y.click(lambda: gate_click_handler("y"), outputs=[gate_input, gate_output])
        gate_btn_z.click(lambda: gate_click_handler("z"), outputs=[gate_input, gate_output])
        gate_btn_cx.click(lambda: gate_click_handler("cx"), outputs=[gate_input, gate_output])
        gate_btn_rx.click(lambda: gate_click_handler("rx"), outputs=[gate_input, gate_output])

        # Connect algorithm buttons to auto-fill input and fetch info
        def algo_click_handler(algo_name):
            info = json.loads(mcp_get_algorithm_info(algo_name))
            return algo_name, info

        algo_btn_bell.click(lambda: algo_click_handler("bell_state"), outputs=[algo_input, algo_output])
        algo_btn_qft.click(lambda: algo_click_handler("qft"), outputs=[algo_input, algo_output])
        algo_btn_grover.click(lambda: algo_click_handler("grover"), outputs=[algo_input, algo_output])

        gate_btn.click(
            lambda g: json.loads(mcp_get_gate_info(g)),
            inputs=[gate_input],
            outputs=[gate_output],
        )
        algo_btn.click(
            lambda a: json.loads(mcp_get_algorithm_info(a)),
            inputs=[algo_input],
            outputs=[algo_output],
        )