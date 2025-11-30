"""
Circuit Visualizers - Generate ASCII art and diagrams of quantum circuits.
"""

from typing import Any


def visualize_circuit_ascii(circuit_data: dict[str, Any]) -> str:
    """
    Generate ASCII art representation of a quantum circuit.
    
    Args:
        circuit_data: Circuit dictionary with gates and qubits
    
    Returns:
        ASCII art string
    """
    num_qubits = circuit_data.get("num_qubits", 0)
    gates = circuit_data.get("gates", [])
    
    if num_qubits == 0:
        return "Empty circuit"
    
    # Track wire positions for each qubit
    wires = [f"q[{i}]: " for i in range(num_qubits)]
    max_prefix = max(len(w) for w in wires)
    wires = [w.ljust(max_prefix) + "─" for w in wires]
    
    for gate in gates:
        name = gate.get("name", "?").upper()
        qubits = gate.get("qubits", [])
        params = gate.get("params", [])
        
        if not qubits:
            continue
        
        # Format gate name
        if params:
            param_str = ",".join(f"{p:.2f}" if isinstance(p, float) else str(p) for p in params[:2])
            gate_str = f"{name}({param_str})"
        else:
            gate_str = name
        
        if len(qubits) == 1:
            # Single qubit gate
            q = qubits[0]
            gate_box = f"┤{gate_str}├"
            wires[q] += gate_box + "─"
            # Pad other wires
            for i in range(num_qubits):
                if i != q:
                    wires[i] += "─" * (len(gate_box) + 1)
        
        elif len(qubits) == 2:
            # Two qubit gate (control-target style)
            q0, q1 = qubits[0], qubits[1]
            min_q, max_q = min(q0, q1), max(q0, q1)
            
            if name in ("CX", "CNOT"):
                # CNOT visualization
                control_q = q0
                target_q = q1
                
                for i in range(num_qubits):
                    if i == control_q:
                        wires[i] += "●──"
                    elif i == target_q:
                        wires[i] += "⊕──"
                    elif min_q < i < max_q:
                        wires[i] += "│──"
                    else:
                        wires[i] += "───"
            elif name in ("CZ",):
                for i in range(num_qubits):
                    if i == q0:
                        wires[i] += "●──"
                    elif i == q1:
                        wires[i] += "●──"
                    elif min_q < i < max_q:
                        wires[i] += "│──"
                    else:
                        wires[i] += "───"
            elif name == "SWAP":
                for i in range(num_qubits):
                    if i == q0 or i == q1:
                        wires[i] += "×──"
                    elif min_q < i < max_q:
                        wires[i] += "│──"
                    else:
                        wires[i] += "───"
            else:
                # Generic two-qubit gate
                gate_width = len(gate_str) + 2
                for i in range(num_qubits):
                    if i == q0:
                        wires[i] += "●" + "─" * (gate_width - 1)
                    elif i == q1:
                        wires[i] += f"┤{gate_str}├"
                    elif min_q < i < max_q:
                        wires[i] += "│" + "─" * (gate_width - 1)
                    else:
                        wires[i] += "─" * gate_width
        
        elif len(qubits) == 3:
            # Three qubit gate (Toffoli, Fredkin)
            min_q = min(qubits)
            max_q = max(qubits)
            
            for i in range(num_qubits):
                if i in qubits:
                    if i == qubits[-1]:
                        if name in ("CCX", "TOFFOLI"):
                            wires[i] += "⊕──"
                        else:
                            wires[i] += f"┤{name}├"
                    else:
                        wires[i] += "●──"
                elif min_q < i < max_q:
                    wires[i] += "│──"
                else:
                    wires[i] += "───"
    
    # Add final lines
    for i in range(num_qubits):
        wires[i] += "─"
    
    return "\n".join(wires)


def circuit_to_latex(circuit_data: dict[str, Any]) -> str:
    """
    Generate LaTeX/Qcircuit representation of a quantum circuit.
    
    Args:
        circuit_data: Circuit dictionary
    
    Returns:
        LaTeX code for Qcircuit
    """
    num_qubits = circuit_data.get("num_qubits", 0)
    gates = circuit_data.get("gates", [])
    
    latex = r"\begin{quantikz}" + "\n"
    
    # Build column-by-column
    columns: list[list[str]] = []
    current_col: list[str] = [r"\ket{0}"] * num_qubits
    columns.append(current_col.copy())
    
    for gate in gates:
        name = gate.get("name", "").upper()
        qubits = gate.get("qubits", [])
        
        col: list[str] = [r"\qw"] * num_qubits
        
        if len(qubits) == 1:
            q = qubits[0]
            if name == "H":
                col[q] = r"\gate{H}"
            elif name == "X":
                col[q] = r"\gate{X}"
            elif name == "Y":
                col[q] = r"\gate{Y}"
            elif name == "Z":
                col[q] = r"\gate{Z}"
            elif name in ("RX", "RY", "RZ"):
                col[q] = rf"\gate{{{name}}}"
            else:
                col[q] = rf"\gate{{{name}}}"
        
        elif len(qubits) == 2:
            q0, q1 = qubits
            diff = q1 - q0
            if name in ("CX", "CNOT"):
                col[q0] = rf"\ctrl{{{diff}}}"
                col[q1] = r"\targ{}"
            elif name == "CZ":
                col[q0] = rf"\ctrl{{{diff}}}"
                col[q1] = r"\gate{Z}"
            elif name == "SWAP":
                col[q0] = rf"\swap{{{diff}}}"
                col[q1] = r"\targX{}"
        
        columns.append(col)
    
    # Build rows
    for q in range(num_qubits):
        row = " & ".join(col[q] for col in columns)
        latex += row + r" \\" + "\n"
    
    latex += r"\end{quantikz}"
    return latex


def circuit_summary(circuit_data: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a summary of circuit properties.
    
    Args:
        circuit_data: Circuit dictionary
    
    Returns:
        Summary dictionary
    """
    gates = circuit_data.get("gates", [])
    num_qubits = circuit_data.get("num_qubits", 0)
    
    gate_counts: dict[str, int] = {}
    single_qubit_gates = 0
    two_qubit_gates = 0
    multi_qubit_gates = 0
    parameterized_gates = 0
    
    for gate in gates:
        name = gate.get("name", "unknown").lower()
        qubits = gate.get("qubits", [])
        params = gate.get("params", [])
        
        if name == "barrier":
            continue
        
        gate_counts[name] = gate_counts.get(name, 0) + 1
        
        if len(qubits) == 1:
            single_qubit_gates += 1
        elif len(qubits) == 2:
            two_qubit_gates += 1
        else:
            multi_qubit_gates += 1
        
        if params:
            parameterized_gates += 1
    
    # Estimate depth (simplified)
    qubit_depths = [0] * num_qubits
    for gate in gates:
        qubits = gate.get("qubits", [])
        if gate.get("name", "").lower() == "barrier":
            continue
        max_depth = max((qubit_depths[q] for q in qubits), default=0)
        for q in qubits:
            qubit_depths[q] = max_depth + 1
    
    depth = max(qubit_depths) if qubit_depths else 0
    
    return {
        "num_qubits": num_qubits,
        "num_classical_bits": circuit_data.get("num_classical_bits", 0),
        "depth": depth,
        "total_gates": len([g for g in gates if g.get("name", "").lower() != "barrier"]),
        "single_qubit_gates": single_qubit_gates,
        "two_qubit_gates": two_qubit_gates,
        "multi_qubit_gates": multi_qubit_gates,
        "parameterized_gates": parameterized_gates,
        "gate_breakdown": gate_counts,
    }
