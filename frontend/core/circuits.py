# =============================================================================
# PREDEFINED TEST CIRCUITS - Important quantum circuits for learning
# =============================================================================
"""
Predefined quantum circuits for testing and learning purposes.
Includes basic gates, entanglement circuits, and multi-qubit operations.
"""

import json


PREDEFINED_CIRCUITS = {
    "single_h": {
        "name": "Single Hadamard",
        "description": "Creates superposition |+⟩ = (|0⟩+|1⟩)/√2",
        "qubits": 1,
        "gates": [{"name": "h", "qubits": [0]}],
        "expected": "50% |0⟩, 50% |1⟩",
        "bloch": "Equator (X-axis)",
        "category": "basic"
    },
    "x_gate": {
        "name": "X Gate (NOT)",
        "description": "Bit flip: |0⟩ → |1⟩",
        "qubits": 1,
        "gates": [{"name": "x", "qubits": [0]}],
        "expected": "100% |1⟩",
        "bloch": "South pole",
        "category": "basic"
    },
    "y_gate": {
        "name": "Y Gate",
        "description": "Bit + phase flip: |0⟩ → i|1⟩",
        "qubits": 1,
        "gates": [{"name": "y", "qubits": [0]}],
        "expected": "100% |1⟩ (with phase)",
        "bloch": "South pole (Y-rotated)",
        "category": "basic"
    },
    "z_gate": {
        "name": "Z Gate",
        "description": "Phase flip: |1⟩ → -|1⟩ (no effect on |0⟩)",
        "qubits": 1,
        "gates": [{"name": "z", "qubits": [0]}],
        "expected": "100% |0⟩ (phase only)",
        "bloch": "North pole (no visible change)",
        "category": "basic"
    },
    "h_z_h": {
        "name": "H-Z-H Sandwich",
        "description": "H·Z·H = X (equivalent to X gate)",
        "qubits": 1,
        "gates": [
            {"name": "h", "qubits": [0]},
            {"name": "z", "qubits": [0]},
            {"name": "h", "qubits": [0]}
        ],
        "expected": "100% |1⟩ (same as X gate)",
        "bloch": "South pole",
        "category": "identity"
    },
    "bell_state": {
        "name": "Bell State (Φ+)",
        "description": "Maximally entangled: (|00⟩+|11⟩)/√2",
        "qubits": 2,
        "gates": [
            {"name": "h", "qubits": [0]},
            {"name": "cx", "qubits": [0, 1]}
        ],
        "expected": "50% |00⟩, 50% |11⟩",
        "bloch": "N/A (2-qubit)",
        "category": "entanglement"
    },
    "bell_psi_plus": {
        "name": "Bell State (Ψ+)",
        "description": "Entangled: (|01⟩+|10⟩)/√2",
        "qubits": 2,
        "gates": [
            {"name": "x", "qubits": [0]},
            {"name": "h", "qubits": [0]},
            {"name": "cx", "qubits": [0, 1]}
        ],
        "expected": "50% |01⟩, 50% |10⟩",
        "bloch": "N/A (2-qubit)",
        "category": "entanglement"
    },
    "ghz_3": {
        "name": "GHZ State (3 qubits)",
        "description": "(|000⟩+|111⟩)/√2 - tripartite entanglement",
        "qubits": 3,
        "gates": [
            {"name": "h", "qubits": [0]},
            {"name": "cx", "qubits": [0, 1]},
            {"name": "cx", "qubits": [0, 2]}
        ],
        "expected": "50% |000⟩, 50% |111⟩",
        "bloch": "N/A (3-qubit)",
        "category": "entanglement"
    },
    "superposition_2": {
        "name": "Uniform Superposition (2q)",
        "description": "All states equal: (|00⟩+|01⟩+|10⟩+|11⟩)/2",
        "qubits": 2,
        "gates": [
            {"name": "h", "qubits": [0]},
            {"name": "h", "qubits": [1]}
        ],
        "expected": "25% each of |00⟩, |01⟩, |10⟩, |11⟩",
        "bloch": "N/A (2-qubit)",
        "category": "superposition"
    },
    "rx_pi_2": {
        "name": "RX(π/2) Rotation",
        "description": "Rotate around X-axis by 90°",
        "qubits": 1,
        "gates": [{"name": "rx", "qubits": [0], "params": [1.5708]}],
        "expected": "50% |0⟩, 50% |1⟩ (on Y-Z plane)",
        "bloch": "On Y-axis (positive)",
        "category": "rotation"
    },
    "ry_pi_2": {
        "name": "RY(π/2) Rotation",
        "description": "Rotate around Y-axis by 90°",
        "qubits": 1,
        "gates": [{"name": "ry", "qubits": [0], "params": [1.5708]}],
        "expected": "50% |0⟩, 50% |1⟩ (on X-Z plane)",
        "bloch": "On X-axis (positive) = |+⟩",
        "category": "rotation"
    },
    "rz_pi": {
        "name": "RZ(π) = Z Gate",
        "description": "Rotate around Z-axis by 180°",
        "qubits": 1,
        "gates": [{"name": "rz", "qubits": [0], "params": [3.14159]}],
        "expected": "100% |0⟩ (phase only)",
        "bloch": "North pole (no visible change)",
        "category": "rotation"
    },
    "swap_test": {
        "name": "SWAP Gate Test",
        "description": "Swap qubit states: |01⟩ → |10⟩",
        "qubits": 2,
        "gates": [
            {"name": "x", "qubits": [0]},
            {"name": "swap", "qubits": [0, 1]}
        ],
        "expected": "100% |10⟩ (swapped from |01⟩)",
        "bloch": "N/A (2-qubit)",
        "category": "multi-qubit"
    },
    "toffoli": {
        "name": "Toffoli (CCX) Gate",
        "description": "Controlled-controlled-X (AND gate)",
        "qubits": 3,
        "gates": [
            {"name": "x", "qubits": [0]},
            {"name": "x", "qubits": [1]},
            {"name": "ccx", "qubits": [0, 1, 2]}
        ],
        "expected": "100% |111⟩ (target flipped)",
        "bloch": "N/A (3-qubit)",
        "category": "multi-qubit"
    },
    "phase_kickback": {
        "name": "Phase Kickback Demo",
        "description": "Shows phase kickback: H·CX·H on control",
        "qubits": 2,
        "gates": [
            {"name": "x", "qubits": [1]},
            {"name": "h", "qubits": [0]},
            {"name": "cx", "qubits": [0, 1]},
            {"name": "h", "qubits": [0]}
        ],
        "expected": "100% |11⟩ (phase kickback flips control)",
        "bloch": "N/A (2-qubit)",
        "category": "advanced"
    }
}


def get_predefined_circuit(circuit_id: str) -> tuple[str, int, str, str]:
    """Get predefined circuit data.
    Returns: (gates_json, num_qubits, description, expected_output)
    """
    if circuit_id not in PREDEFINED_CIRCUITS:
        return "[]", 2, "Unknown circuit", ""
    
    circuit = PREDEFINED_CIRCUITS[circuit_id]
    return (
        json.dumps(circuit["gates"]),
        circuit["qubits"],
        f"**{circuit['name']}**\n\n{circuit['description']}\n\n**Expected:** {circuit['expected']}\n\n**Bloch:** {circuit['bloch']}",
        circuit["expected"]
    )


def render_predefined_circuit_card(circuit_id: str, circuit: dict) -> str:
    """Render a beautiful card for a predefined circuit."""
    category_colors = {
        "basic": "#4fc3f7",
        "entanglement": "#f06292",
        "superposition": "#81c784",
        "rotation": "#ffb74d",
        "multi-qubit": "#ce93d8",
        "identity": "#90a4ae",
        "advanced": "#ff8a65"
    }
    color = category_colors.get(circuit.get("category", "basic"), "#4fc3f7")
    
    return f'''
    <div style="
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid {color}40;
        border-radius: 8px;
        padding: 10px;
        margin: 4px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    " onmouseover="this.style.borderColor='{color}'" onmouseout="this.style.borderColor='{color}40'">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: {color}; font-weight: bold; font-size: 0.95em;">{circuit["name"]}</span>
            <span style="
                background: {color}20;
                color: {color};
                padding: 2px 6px;
                border-radius: 8px;
                font-size: 0.7em;
            ">{circuit["qubits"]}q</span>
        </div>
        <p style="color: #a0a0b0; font-size: 0.8em; margin: 4px 0 2px 0;">{circuit["description"]}</p>
        <div style="color: #6a6a7a; font-size: 0.7em;">Expected: {circuit["expected"]}</div>
    </div>
    '''


def load_test_circuit(circuit_id: str) -> tuple[str, int, str, str]:
    """Load a predefined test circuit and update all UI components."""
    from ..ui.visualizations import render_visual_circuit
    
    gates_json, num_qubits, info, expected = get_predefined_circuit(circuit_id)
    circuit_svg = render_visual_circuit(gates_json, num_qubits)
    return gates_json, num_qubits, circuit_svg, info
