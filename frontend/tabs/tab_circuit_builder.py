"""
Circuit Builder Tab for QuantumArchitect-MCP
"""

import gradio as gr

# Import required components
from ..ui import (
    render_visual_circuit,
    ui_simulate_visual,
)
from ..core import (
    make_gate_handler,
    clear_circuit_handler,
    undo_handler,
    load_test_circuit,
)


def add_circuit_builder_tab():
    """Add the Circuit Builder tab to the Gradio interface."""
    with gr.TabItem("⚛️ Circuit Builder", id="builder"):
        # Quick Test Circuits Panel
        with gr.Accordion("🧪 Quick Test Circuits", open=True):
            gr.Markdown("*Load predefined circuits to test all visualization features*", elem_classes=["category-header"])

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**🔵 Single Qubit**", elem_classes=["category-header"])
                    with gr.Row():
                        test_btn_h = gr.Button("H", size="sm", variant="secondary")
                        test_btn_x = gr.Button("X", size="sm", variant="secondary")
                        test_btn_y = gr.Button("Y", size="sm", variant="secondary")
                    with gr.Row():
                        test_btn_rx = gr.Button("RX", size="sm", variant="secondary")
                        test_btn_ry = gr.Button("RY", size="sm", variant="secondary")
                        test_btn_hzh = gr.Button("HZH", size="sm", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("**🔗 Entanglement**", elem_classes=["category-header"])
                    with gr.Row():
                        test_btn_bell = gr.Button("Bell", size="sm", variant="primary")
                        test_btn_bell_psi = gr.Button("Ψ+", size="sm", variant="primary")
                        test_btn_super2 = gr.Button("Super", size="sm", variant="secondary")
                    with gr.Row():
                        test_btn_ghz = gr.Button("GHZ", size="sm", variant="primary")
                        test_btn_swap = gr.Button("SWAP", size="sm", variant="secondary")
                        test_btn_toffoli = gr.Button("CCX", size="sm", variant="secondary")

            test_circuit_info = gr.Markdown(
                value="*Click a button to load a test circuit*",
                elem_id="test-circuit-info"
            )

        with gr.Row():
            # Left sidebar - Gate Palette (compact)
            with gr.Column(scale=1, min_width=150):
                gr.Markdown("### 🎨 Gates")

                with gr.Group():
                    gr.Markdown("**Pauli**", elem_classes=["category-header"])
                    with gr.Row():
                        btn_x = gr.Button("X", size="sm", elem_classes=["gate-pauli"])
                        btn_y = gr.Button("Y", size="sm", elem_classes=["gate-pauli"])
                        btn_z = gr.Button("Z", size="sm", elem_classes=["gate-pauli"])

                with gr.Group():
                    gr.Markdown("**Superpos.**", elem_classes=["category-header"])
                    with gr.Row():
                        btn_h = gr.Button("H", size="sm", elem_classes=["gate-hadamard"])
                        btn_s = gr.Button("S", size="sm", elem_classes=["gate-phase"])
                        btn_t = gr.Button("T", size="sm", elem_classes=["gate-phase"])

                with gr.Group():
                    gr.Markdown("**Rotation**", elem_classes=["category-header"])
                    with gr.Row():
                        btn_rx = gr.Button("Rx", size="sm", elem_classes=["gate-rotation"])
                        btn_ry = gr.Button("Ry", size="sm", elem_classes=["gate-rotation"])
                        btn_rz = gr.Button("Rz", size="sm", elem_classes=["gate-rotation"])

                with gr.Group():
                    gr.Markdown("**Multi-Q**", elem_classes=["category-header"])
                    with gr.Row():
                        btn_cx = gr.Button("CX", size="sm", elem_classes=["gate-multi"])
                        btn_cz = gr.Button("CZ", size="sm", elem_classes=["gate-multi"])
                        btn_swap = gr.Button("SW", size="sm", elem_classes=["gate-multi"])
                    with gr.Row():
                        btn_ccx = gr.Button("CCX", size="sm", elem_classes=["gate-multi3"])

                gr.Markdown("---")

                with gr.Group():
                    gr.Markdown("**🎯 Target**", elem_classes=["category-header"])
                    with gr.Row(elem_classes=["compact-row"]):
                        builder_qubit1 = gr.Number(value=0, label="Q1", minimum=0, maximum=9, step=1, scale=1, min_width=50, show_label=True, container=True)
                        builder_qubit2 = gr.Number(value=1, label="Q2", minimum=0, maximum=9, step=1, scale=1, min_width=50, show_label=True, container=True)
                    builder_param = gr.Number(value=3.14159, label="θ", step=0.01, scale=1, min_width=80, show_label=True, container=True)

            # Main canvas area
            with gr.Column(scale=3):
                with gr.Row():
                    builder_num_qubits = gr.Slider(
                        minimum=1, maximum=8, value=4, step=1,
                        label="Qubits", info="1-8"
                    )
                    builder_shots = gr.Slider(
                        minimum=100, maximum=8192, value=1024, step=100,
                        label="Shots", info="Measurements"
                    )

                gr.Markdown("### 🔲 Circuit")
                with gr.Group():
                    circuit_canvas = gr.HTML(
                        value=render_visual_circuit("[]", 4),
                        elem_id="circuit-canvas",
                        elem_classes=["circuit-canvas-tall"]
                    )

                with gr.Row():
                    run_sim_btn = gr.Button("▶️ Simulate", variant="primary", size="lg")
                    clear_circuit_btn = gr.Button("🗑️ Clear", variant="secondary")
                    undo_btn = gr.Button("↩️ Undo", variant="secondary")

                builder_gates_json = gr.Code(
                    value="[]",
                    language="json",
                    label="Gates JSON",
                    lines=1,
                    visible=True,
                    elem_classes=["compact-code"]
                )

        # Results section - compact 3-column grid layout
        gr.Markdown("### 📊 Simulation Results")

        with gr.Row(equal_height=True):
            # Column 1: Probabilities + Q-Sphere
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion("📊 Probabilities", open=True):
                    simulation_results = gr.HTML(
                        value="<div class='results-placeholder'>Run simulation to see probability distribution</div>",
                        elem_id="simulation-results",
                        elem_classes=["result-compact"]
                    )
                
                with gr.Accordion("🌐 Q-Sphere", open=True):
                    qsphere_display = gr.HTML(
                        value="<div class='results-placeholder'>Q-Sphere visualization</div>",
                        elem_id="qsphere-display",
                        elem_classes=["result-compact"]
                    )

            # Column 2: Bloch Sphere (single qubit)
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion("🔮 Bloch Sphere (1q)", open=True):
                    bloch_display = gr.Plot(
                        elem_id="bloch-display",
                        elem_classes=["result-compact"]
                    )

            # Column 3: Statevector + Raw JSON
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion("📐 Statevector", open=True):
                    statevector_display = gr.HTML(
                        value="<div class='results-placeholder'>Complex amplitudes</div>",
                        elem_id="statevector-display",
                        elem_classes=["result-compact"]
                    )
                
                with gr.Accordion("📈 Raw Data", open=False):
                    raw_results = gr.JSON(
                        elem_id="raw-results",
                        elem_classes=["result-compact"]
                    )

        # Gate button handlers
        btn_x.click(make_gate_handler("x"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_y.click(make_gate_handler("y"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_z.click(make_gate_handler("z"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_h.click(make_gate_handler("h"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_s.click(make_gate_handler("s"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_t.click(make_gate_handler("t"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_rx.click(make_gate_handler("rx"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_ry.click(make_gate_handler("ry"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_rz.click(make_gate_handler("rz"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_cx.click(make_gate_handler("cx"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_cz.click(make_gate_handler("cz"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_swap.click(make_gate_handler("swap"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])
        btn_ccx.click(make_gate_handler("ccx"), inputs=[builder_gates_json, builder_qubit1, builder_qubit2, builder_param, builder_num_qubits], outputs=[builder_gates_json, circuit_canvas])

        # Action buttons
        clear_circuit_btn.click(
            clear_circuit_handler,
            inputs=[builder_num_qubits],
            outputs=[builder_gates_json, circuit_canvas]
        )

        undo_btn.click(
            undo_handler,
            inputs=[builder_gates_json, builder_num_qubits],
            outputs=[builder_gates_json, circuit_canvas]
        )

        # Test circuit buttons
        test_btn_h.click(lambda: load_test_circuit("single_h"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_x.click(lambda: load_test_circuit("x_gate"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_y.click(lambda: load_test_circuit("y_gate"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_rx.click(lambda: load_test_circuit("rx_pi_2"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_ry.click(lambda: load_test_circuit("ry_pi_2"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_hzh.click(lambda: load_test_circuit("h_z_h"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_bell.click(lambda: load_test_circuit("bell_state"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_bell_psi.click(lambda: load_test_circuit("bell_psi_plus"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_super2.click(lambda: load_test_circuit("superposition_2"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_ghz.click(lambda: load_test_circuit("ghz_3"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_swap.click(lambda: load_test_circuit("swap_test"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])
        test_btn_toffoli.click(lambda: load_test_circuit("toffoli"), outputs=[builder_gates_json, builder_num_qubits, circuit_canvas, test_circuit_info])

        # Update canvas on qubit change
        builder_num_qubits.change(
            lambda gates, n: render_visual_circuit(gates, int(n)),
            inputs=[builder_gates_json, builder_num_qubits],
            outputs=[circuit_canvas]
        )

        # Simulate button
        run_sim_btn.click(
            ui_simulate_visual,
            inputs=[builder_gates_json, builder_num_qubits, builder_shots],
            outputs=[circuit_canvas, simulation_results, qsphere_display, bloch_display, statevector_display, raw_results]
        )