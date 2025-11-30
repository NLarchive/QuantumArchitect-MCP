# =============================================================================
# HELPER FUNCTIONS - Utility functions for circuit building and templates
# =============================================================================
"""
Helper functions for the Gradio UI including gate manipulation,
template handling, and circuit construction utilities.
"""

import json

from .constants import GATE_LIBRARY, TEMPLATE_PARAMS


# Example QASM for demo
EXAMPLE_QASM = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""


def get_template_params(template: str) -> str:
    """Return example parameters for selected template."""
    return TEMPLATE_PARAMS.get(template, {}).get("example", "{}")


def get_template_info(template: str) -> str:
    """Return info about template parameters."""
    info = TEMPLATE_PARAMS.get(template, {})
    return f"**{template}**: {info.get('description', 'No info')}\n\nExample: `{info.get('example', '{}')}`"


def add_gate_to_json(current_json: str, gate_name: str, qubit: int, qubit2: int, param: float, num_qubits: int = 4) -> str:
    """Add a gate to the current JSON configuration.
    
    Args:
        current_json: Current gates JSON string
        gate_name: Name of the gate to add
        qubit: First qubit index
        qubit2: Second qubit index (for 2+ qubit gates)
        param: Parameter value (for parameterized gates)
        num_qubits: Total number of qubits in the circuit
    """
    try:
        gates = json.loads(current_json) if current_json and current_json != "{}" else []
        if not isinstance(gates, list):
            gates = []
    except json.JSONDecodeError:
        gates = []
    
    gate_info = GATE_LIBRARY.get(gate_name, {})
    new_gate = {"name": gate_name, "qubits": [qubit]}
    num_qubits_required = gate_info.get("qubits", 1)
    
    if num_qubits_required >= 2:
        new_gate["qubits"] = [qubit, qubit2]
    if num_qubits_required >= 3:
        # For 3-qubit gates, find a valid third qubit within circuit bounds
        used_qubits = {qubit, qubit2}
        third_qubit = -1
        for i in range(num_qubits):
            if i not in used_qubits:
                third_qubit = i
                break
        
        # Fallback if no third qubit found (circuit too small)
        if third_qubit == -1:
            third_qubit = (qubit2 + 1) % max(num_qubits, 3)
            while third_qubit in used_qubits and num_qubits > 2:
                third_qubit = (third_qubit + 1) % num_qubits
        
        new_gate["qubits"] = [qubit, qubit2, third_qubit]
    if gate_info.get("params"):
        new_gate["params"] = [param]
    
    gates.append(new_gate)
    return json.dumps(gates, indent=2)


def clear_gates() -> str:
    """Clear all gates."""
    return "[]"


def make_gate_handler(gate_name: str):
    """Create a gate handler function for a specific gate."""
    from ..ui.visualizations import render_visual_circuit
    
    def handler(gates_json, q1, q2, param, num_qubits):
        result = add_gate_to_json(gates_json, gate_name, int(q1), int(q2), param, int(num_qubits))
        svg = render_visual_circuit(result, int(num_qubits))
        return result, svg
    return handler


def clear_circuit_handler(num_qubits: int) -> tuple[str, str]:
    """Handle clearing the circuit."""
    from ..ui.visualizations import render_visual_circuit
    return "[]", render_visual_circuit("[]", int(num_qubits))


def undo_handler(gates_json: str, num_qubits: int) -> tuple[str, str]:
    """Handle undoing the last gate."""
    from ..ui.visualizations import render_visual_circuit
    try:
        gates = json.loads(gates_json)
        if gates:
            gates.pop()
        result = json.dumps(gates, indent=2)
        return result, render_visual_circuit(result, int(num_qubits))
    except (json.JSONDecodeError, ValueError, TypeError):
        return "[]", render_visual_circuit("[]", int(num_qubits))
