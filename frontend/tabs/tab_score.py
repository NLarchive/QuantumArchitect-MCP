"""
Path: QuantumArchitect-MCP/frontend/tabs/tab_score.py
Related: ui/ui_handlers.py (ui_score_circuit), core/circuits.py (PREDEFINED_CIRCUITS)
Purpose: Score tab with editable QASM input and predefined circuit selection
"""

import gradio as gr
import json

# Import required components
from ..core import EXAMPLE_QASM, PREDEFINED_CIRCUITS


def get_qasm_from_gates(gates_json: str, num_qubits: int) -> str:
    """Convert gates JSON to OpenQASM 2.0 code."""
    try:
        gates = json.loads(gates_json) if isinstance(gates_json, str) else gates_json
    except:
        return EXAMPLE_QASM
    
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{num_qubits}];",
        f"creg c[{num_qubits}];",
        ""
    ]
    
    for gate in gates:
        # Skip if gate is not a dict or name is not a string
        if not isinstance(gate, dict):
            continue
        name_raw = gate.get("name", "")
        if not isinstance(name_raw, str):
            continue
        name = name_raw.lower()
        qubits = gate.get("qubits", [])
        params = gate.get("params", [])
        
        if name in ["h", "x", "y", "z", "s", "t", "sdg", "tdg"]:
            lines.append(f"{name} q[{qubits[0]}];")
        elif name in ["rx", "ry", "rz"]:
            param = params[0] if params else 0
            lines.append(f"{name}({param}) q[{qubits[0]}];")
        elif name in ["cx", "cnot"]:
            lines.append(f"cx q[{qubits[0]}],q[{qubits[1]}];")
        elif name == "cz":
            lines.append(f"cz q[{qubits[0]}],q[{qubits[1]}];")
        elif name == "swap":
            lines.append(f"swap q[{qubits[0]}],q[{qubits[1]}];")
        elif name == "ccx":
            lines.append(f"ccx q[{qubits[0]}],q[{qubits[1]}],q[{qubits[2]}];")
    
    # Add measurements
    lines.append("")
    for i in range(num_qubits):
        lines.append(f"measure q[{i}] -> c[{i}];")
    
    return "\n".join(lines)


def load_circuit_for_scoring(circuit_id: str):
    """Load a predefined circuit and return its QASM code."""
    if circuit_id in PREDEFINED_CIRCUITS:
        circuit = PREDEFINED_CIRCUITS[circuit_id]
        qasm = get_qasm_from_gates(json.dumps(circuit["gates"]), circuit["qubits"])
        info = f"**{circuit['name']}**\n\n{circuit['description']}\n\n**Expected:** {circuit['expected']}"
        return qasm, info
    return EXAMPLE_QASM, "*Select a circuit or edit QASM directly*"


# Import UI handler
from ..ui import ui_score_circuit


def add_score_tab():
    """Add the Score tab to the Gradio interface."""
    with gr.TabItem("📊 Score"):
        gr.Markdown("### Analyze circuit complexity")
        gr.Markdown("*Select a predefined circuit or edit the QASM code directly*")

        with gr.Row():
            with gr.Column():
                # Circuit selection dropdown
                circuit_choices = [(f"{v['name']} ({v['qubits']}q)", k) for k, v in PREDEFINED_CIRCUITS.items()]
                circuit_selector = gr.Dropdown(
                    choices=circuit_choices,
                    value="bell_state",
                    label="📋 Load Predefined Circuit",
                    interactive=True
                )
                circuit_info = gr.Markdown(value="*Select a circuit to see details*")
                
                # Editable QASM input
                score_qasm = gr.Code(value=EXAMPLE_QASM, language="python", label="OpenQASM 2.0 Code", lines=12, interactive=True)
                
                score_hardware = gr.Dropdown(
                    choices=["", "ibm_brisbane", "ibm_sherbrooke", "rigetti_aspen", "google_sycamore", "ionq_harmony", "quantinuum_h1", "linear_5"],
                    value="ibm_brisbane",
                    label="🖥️ Hardware Target"
                )
                score_btn = gr.Button("📊 Score", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("#### 📈 Scoring Results")
                score_output = gr.Markdown(value="*Click Score to analyze the circuit*")

        # Event handlers
        circuit_selector.change(
            load_circuit_for_scoring, 
            inputs=[circuit_selector], 
            outputs=[score_qasm, circuit_info]
        )
        score_btn.click(ui_score_circuit, inputs=[score_qasm, score_hardware], outputs=[score_output])