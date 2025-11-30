# =============================================================================
# UI HANDLER FUNCTIONS - Human-Friendly Interface Wrappers
# =============================================================================
"""
UI handler functions that wrap MCP tools for Gradio interface.
These provide formatted output suitable for display to users.
"""

import json
from typing import Any

from .mcp_tools import (
    mcp_create_circuit,
    mcp_validate_circuit,
    mcp_simulate,
    mcp_score_circuit,
)
from .visualizations import (
    render_visual_circuit,
    render_qsphere_svg,
    render_probability_bars,
    render_statevector_amplitudes,
    plot_bloch_sphere_plotly,
    create_placeholder_plot,
)


def ui_create_circuit(template: str, qubits: int, params: str) -> tuple[str, str, str]:
    """UI wrapper for circuit creation."""
    try:
        result_json = mcp_create_circuit(template, qubits, params)
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        error_msg = f"JSON Parse Error: {e}"
        return error_msg, error_msg, json.dumps({"error": str(e)})
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg, json.dumps({"error": str(e)})
    
    if result.get("success"):
        qasm = result.get("qasm", "")
        visualization = result.get("visualization", "")
        summary = json.dumps(result.get("summary", {}), indent=2)
        return qasm, visualization, summary
    else:
        error = result.get("error", "Unknown error")
        error_msg = f"Error: {error}"
        return error_msg, error_msg, json.dumps({"error": error})


def ui_validate_circuit(qasm: str, hardware: str) -> tuple[str, str]:
    """UI wrapper for validation."""
    try:
        result_json = mcp_validate_circuit(qasm, hardware, True, True)
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        return "❌ Parse Error", f"Failed to parse response: {e}"
    except Exception as e:
        return "❌ Error", f"Validation error: {str(e)}"
    
    status = "✅ Valid" if result.get("valid") else "❌ Invalid"
    
    details = []
    if result.get("errors"):
        details.append("**Errors:**")
        details.extend([f"- {e}" for e in result["errors"]])
    if result.get("warnings"):
        details.append("**Warnings:**")
        details.extend([f"- {w}" for w in result["warnings"]])
    if result.get("circuit_info"):
        details.append("**Circuit Info:**")
        details.append(f"- Qubits: {result['circuit_info'].get('num_qubits', 'N/A')}")
        details.append(f"- Gates: {result['circuit_info'].get('num_gates', 'N/A')}")
        details.append(f"- Depth: {result['circuit_info'].get('depth', 'N/A')}")
    
    return status, "\n".join(details)


def ui_simulate_circuit(qasm: str, shots: int) -> tuple[str, str]:
    """UI wrapper for simulation."""
    try:
        result_json = mcp_simulate(qasm, shots, True, "")
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        error_msg = f"Parse Error: {e}"
        return error_msg, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg
    
    if result.get("success"):
        counts = result.get("counts", {})
        probs = result.get("probabilities", {})
        
        counts_str = "\n".join([f"|{k}⟩: {v}" for k, v in sorted(counts.items())])
        probs_str = "\n".join([f"|{k}⟩: {v:.4f}" for k, v in sorted(probs.items())])
        
        return f"**Counts ({shots} shots):**\n{counts_str}", f"**Probabilities:**\n{probs_str}"
    else:
        error_msg = f"Error: {result.get('error', 'Unknown')}"
        return error_msg, error_msg


def ui_score_circuit(qasm: str, hardware: str) -> str:
    """UI wrapper for scoring."""
    try:
        result_json = mcp_score_circuit(qasm, hardware)
        result = json.loads(result_json)
    except json.JSONDecodeError as e:
        return f"Parse Error: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

    if result.get("success"):
        lines = [f"**Overall Score:** {result.get('overall_score', 0):.2f}/1.00"]
        
        complexity = result.get("complexity", {})
        if complexity:
            lines.append("\n**Complexity Metrics:**")
            lines.append(f"- Gate Count: {complexity.get('gate_count', 'N/A')}")
            lines.append(f"- Depth: {complexity.get('depth', 'N/A')}")
            lines.append(f"- Two-Qubit Gates: {complexity.get('two_qubit_gates', 'N/A')}")
        
        fitness = result.get("hardware_fitness")
        if fitness:
            lines.append(f"\n**Hardware Fitness:** {fitness.get('overall_score', 0):.2f}")
        
        fidelity = result.get("estimated_fidelity")
        if fidelity:
            lines.append(f"\n**Estimated Fidelity:** {fidelity:.4f}")
        
        return "\n".join(lines)
    else:
        return f"Error: {result.get('error', 'Unknown')}"


def ui_simulate_visual(gates_json: str, num_qubits: int, shots: int) -> tuple[str, str, str, Any, str, str]:
    """Simulate circuit and return visual results with enhanced visualization.
    Returns: (circuit_svg, prob_bars, qsphere, bloch_figure, statevector_html, result_json)
    """
    # Build QASM from gates
    try:
        try:
            gates = json.loads(gates_json) if gates_json else []
        except json.JSONDecodeError as je:
            return (
                render_visual_circuit("[]", num_qubits),
                f"<p style='color: #ef5350;'>Invalid gates JSON: {je}</p>",
                "<p style='color:#ef5350;'>Q-sphere error</p>",
                create_placeholder_plot(f"JSON Error: {je}"),
                "<p style='color:#ef5350;'>Statevector error</p>",
                json.dumps({"error": f"Invalid JSON: {je}"})
            )
        if not gates:
            empty_probs = {"0" * num_qubits: 1.0}
            # Use placeholder for multi-qubit empty circuit
            if num_qubits == 1:
                bloch_fig = plot_bloch_sphere_plotly([1, 0])
            else:
                bloch_fig = create_placeholder_plot("Bloch Sphere is only available for single-qubit circuits.")
            return (
                render_visual_circuit("[]", num_qubits),
                "<p style='color: #78909c;'>Add gates to simulate</p>",
                render_qsphere_svg(empty_probs, num_qubits),
                bloch_fig,
                render_statevector_amplitudes({}, num_qubits),
                "{}"
            )

        # Create QASM
        qasm = f'''OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];
'''
        for gate in gates:
            name = gate.get("name", "")
            qubits = gate.get("qubits", [])
            params = gate.get("params", [])
            
            if params:
                param_str = ",".join(str(p) for p in params)
                qubit_str = ",".join(f"q[{q}]" for q in qubits)
                qasm += f"{name}({param_str}) {qubit_str};\n"
            else:
                qubit_str = ",".join(f"q[{q}]" for q in qubits)
                qasm += f"{name} {qubit_str};\n"
        qasm += "measure q -> c;\n"

        # Run simulation
        result_json = mcp_simulate(qasm, shots, True, "")
        result = json.loads(result_json)
        
        circuit_svg = render_visual_circuit(gates_json, num_qubits)
        
        # Handle simulation errors gracefully
        if not result.get("success"):
            error_msg = result.get("error", "Unknown simulation error")
            error_html = f"<p style='color: #ef5350;'>Error: {error_msg}</p>"
            return (
                circuit_svg,
                error_html,
                "<p style='color:#ef5350;'>Q-sphere error</p>",
                create_placeholder_plot(f"Simulation Error: {error_msg}"),
                error_html,
                result_json,
            )
        
        prob_bars = render_probability_bars(result)
        
        # Generate Q-sphere
        probs = result.get("probabilities", {})
        qsphere = render_qsphere_svg(probs, num_qubits)
        
        # Get statevector data for amplitudes display
        statevector = result.get("statevector", {})
        statevector_html = render_statevector_amplitudes(statevector, num_qubits)
        
        # Generate Bloch sphere for single qubit using Plotly
        if num_qubits == 1 and statevector:
            bloch = plot_bloch_sphere_plotly(statevector)
        else:
            bloch = create_placeholder_plot("Bloch Sphere is only available for single-qubit circuits.")
        
        return circuit_svg, prob_bars, qsphere, bloch, statevector_html, result_json
        
    except Exception as e:
        error_msg = str(e)
        return (
            render_visual_circuit("[]", num_qubits),
            f"<p style='color: #ef5350;'>Error: {error_msg}</p>",
            "<p style='color:#ef5350;'>Q-sphere error</p>",
            create_placeholder_plot(f"Error: {error_msg}"),
            "<p style='color:#ef5350;'>Statevector error</p>",
            json.dumps({"error": error_msg})
        )
