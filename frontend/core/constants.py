
# Gate definitions for interactive builder - IBM Composer style categories
GATE_LIBRARY = {
    # Pauli Gates (Blue)
    "x": {"symbol": "X", "name": "Pauli-X (NOT)", "qubits": 1, "params": [], "formula": "|0⟩↔|1⟩", "category": "pauli", "color": "#1976D2"},
    "y": {"symbol": "Y", "name": "Pauli-Y", "qubits": 1, "params": [], "formula": "iσy rotation", "category": "pauli", "color": "#1976D2"},
    "z": {"symbol": "Z", "name": "Pauli-Z", "qubits": 1, "params": [], "formula": "|1⟩→-|1⟩", "category": "pauli", "color": "#1976D2"},
    # Hadamard & Phase (Teal)
    "h": {"symbol": "H", "name": "Hadamard", "qubits": 1, "params": [], "formula": "|+⟩ = (|0⟩+|1⟩)/√2", "category": "hadamard", "color": "#00897B"},
    "s": {"symbol": "S", "name": "S Phase", "qubits": 1, "params": [], "formula": "√Z gate", "category": "phase", "color": "#00897B"},
    "sdg": {"symbol": "S†", "name": "S-dagger", "qubits": 1, "params": [], "formula": "S† = √Z†", "category": "phase", "color": "#00897B"},
    "t": {"symbol": "T", "name": "T Phase", "qubits": 1, "params": [], "formula": "π/4 phase", "category": "phase", "color": "#00897B"},
    "tdg": {"symbol": "T†", "name": "T-dagger", "qubits": 1, "params": [], "formula": "T† = π/8†", "category": "phase", "color": "#00897B"},
    # Rotation Gates (Purple)
    "rx": {"symbol": "Rx", "name": "X Rotation", "qubits": 1, "params": ["θ"], "formula": "e^(-iθX/2)", "category": "rotation", "color": "#7B1FA2"},
    "ry": {"symbol": "Ry", "name": "Y Rotation", "qubits": 1, "params": ["θ"], "formula": "e^(-iθY/2)", "category": "rotation", "color": "#7B1FA2"},
    "rz": {"symbol": "Rz", "name": "Z Rotation", "qubits": 1, "params": ["θ"], "formula": "e^(-iθZ/2)", "category": "rotation", "color": "#7B1FA2"},
    "u": {"symbol": "U", "name": "Universal", "qubits": 1, "params": ["θ", "φ", "λ"], "formula": "U(θ,φ,λ)", "category": "rotation", "color": "#7B1FA2"},
    # Two-Qubit Gates (Orange)
    "cx": {"symbol": "CX", "name": "CNOT", "qubits": 2, "params": [], "formula": "|11⟩↔|10⟩", "category": "multi", "color": "#F57C00"},
    "cz": {"symbol": "CZ", "name": "Controlled-Z", "qubits": 2, "params": [], "formula": "|11⟩→-|11⟩", "category": "multi", "color": "#F57C00"},
    "cy": {"symbol": "CY", "name": "Controlled-Y", "qubits": 2, "params": [], "formula": "CY gate", "category": "multi", "color": "#F57C00"},
    "swap": {"symbol": "SWAP", "name": "SWAP", "qubits": 2, "params": [], "formula": "|01⟩↔|10⟩", "category": "multi", "color": "#F57C00"},
    "ch": {"symbol": "CH", "name": "Controlled-H", "qubits": 2, "params": [], "formula": "Controlled Hadamard", "category": "multi", "color": "#F57C00"},
    # Three-Qubit Gates (Red)
    "ccx": {"symbol": "CCX", "name": "Toffoli", "qubits": 3, "params": [], "formula": "AND gate", "category": "multi3", "color": "#D32F2F"},
    "cswap": {"symbol": "CSWAP", "name": "Fredkin", "qubits": 3, "params": [], "formula": "Controlled SWAP", "category": "multi3", "color": "#D32F2F"},
}

# Gate categories for the palette
GATE_CATEGORIES = {
    "pauli": {"name": "Pauli", "icon": "σ", "color": "#1976D2", "description": "Pauli X, Y, Z gates"},
    "hadamard": {"name": "Hadamard & Phase", "icon": "H", "color": "#00897B", "description": "Superposition and phase gates"},
    "rotation": {"name": "Rotation", "icon": "Rθ", "color": "#7B1FA2", "description": "Parameterized rotation gates"},
    "multi": {"name": "2-Qubit", "icon": "⊕", "color": "#F57C00", "description": "Two-qubit entangling gates"},
    "multi3": {"name": "3-Qubit", "icon": "⊗", "color": "#D32F2F", "description": "Three-qubit gates"},
}

# Parameter examples for each template
TEMPLATE_PARAMS = {
    "bell_state": {"example": "{}", "description": "No parameters needed"},
    "ghz_state": {"example": "{}", "description": "Uses num_qubits setting"},
    "w_state": {"example": "{}", "description": "Uses num_qubits setting"},
    "superposition": {"example": "{}", "description": "Uniform superposition on all qubits"},
    "qft": {"example": '{"with_swaps": true}', "description": "Quantum Fourier Transform"},
    "grover": {"example": '{"marked_states": [0], "iterations": 2}', "description": "Search iterations"},
    "vqe": {"example": '{"depth": 2, "entanglement": "linear"}', "description": "VQE ansatz depth"},
    "qaoa": {"example": '{"num_layers": 2}', "description": "QAOA optimization layers"},
}
