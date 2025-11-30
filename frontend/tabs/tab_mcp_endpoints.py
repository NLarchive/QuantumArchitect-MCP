"""
Path: QuantumArchitect-MCP/frontend/tabs/tab_mcp_endpoints.py
Related: ui/mcp_tools.py, src/mcp_server/endpoint_handlers.py
Purpose: MCP Endpoints documentation tab with health status monitoring
"""

import gradio as gr
import time


def check_mcp_health() -> str:
    """Check health status of MCP endpoints and return HTML status report."""
    start_time = time.time()
    
    # Define endpoint categories and their tools
    endpoint_categories = {
        "Circuit Creation": ["create_circuit", "parse_qasm", "build_circuit"],
        "Validation": ["validate_circuit", "check_hardware"],
        "Simulation": ["simulate", "get_statevector", "estimate_fidelity"],
        "Scoring": ["score_circuit", "compare_circuits"],
        "Documentation": ["get_gate_info", "get_algorithm_info", "list_hardware", "list_templates", "get_learning_path"]
    }
    
    # Check if core imports work
    status_checks = {}
    
    try:
        from src.mcp_server import context_provider
        status_checks["Context Provider"] = ("✅", "Loaded")
    except Exception as e:
        status_checks["Context Provider"] = ("❌", str(e)[:50])
    
    try:
        from src.mcp_server import endpoint_handlers
        status_checks["Endpoint Handlers"] = ("✅", "Loaded")
    except Exception as e:
        status_checks["Endpoint Handlers"] = ("❌", str(e)[:50])
    
    try:
        from src.core import circuit_parser
        status_checks["Circuit Parser"] = ("✅", "Loaded")
    except Exception as e:
        status_checks["Circuit Parser"] = ("❌", str(e)[:50])
    
    try:
        import qiskit
        status_checks["Qiskit Backend"] = ("✅", f"v{qiskit.__version__}")
    except Exception as e:
        status_checks["Qiskit Backend"] = ("⚠️", "Not installed")
    
    try:
        import numpy
        status_checks["NumPy"] = ("✅", f"v{numpy.__version__}")
    except Exception as e:
        status_checks["NumPy"] = ("❌", str(e)[:50])
    
    # Calculate overall health
    healthy_count = sum(1 for s, _ in status_checks.values() if s == "✅")
    total_count = len(status_checks)
    health_percent = (healthy_count / total_count) * 100
    
    elapsed = (time.time() - start_time) * 1000
    
    # Build HTML report
    html = f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 12px; border: 1px solid #30363d;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <h3 style="color: #4fc3f7; margin: 0;">🔧 MCP Server Health Status</h3>
            <div style="display: flex; gap: 15px; align-items: center;">
                <span style="color: #8b949e; font-size: 0.9em;">Response: {elapsed:.1f}ms</span>
                <span style="
                    background: {'#3fb950' if health_percent >= 80 else '#d29922' if health_percent >= 50 else '#f85149'};
                    color: #0d1117;
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 0.9em;
                ">{health_percent:.0f}% Healthy</span>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px;">
    """
    
    for name, (status, detail) in status_checks.items():
        color = "#3fb950" if status == "✅" else "#d29922" if status == "⚠️" else "#f85149"
        html += f"""
            <div style="background: #21262d; padding: 12px; border-radius: 8px; border-left: 3px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #c9d1d9; font-weight: 500;">{name}</span>
                    <span>{status}</span>
                </div>
                <div style="color: #8b949e; font-size: 0.8em; margin-top: 4px;">{detail}</div>
            </div>
        """
    
    html += """
        </div>
        
        <div style="margin-top: 15px;">
            <h4 style="color: #7c4dff; margin-bottom: 10px;">📡 Endpoint Categories</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
    """
    
    category_colors = {
        "Circuit Creation": "#4fc3f7",
        "Validation": "#3fb950",
        "Simulation": "#d29922",
        "Scoring": "#f06292",
        "Documentation": "#7c4dff"
    }
    
    for category, tools in endpoint_categories.items():
        color = category_colors.get(category, "#8b949e")
        html += f"""
            <div style="
                background: {color}20;
                border: 1px solid {color}60;
                padding: 8px 14px;
                border-radius: 20px;
                display: flex;
                align-items: center;
                gap: 6px;
            ">
                <span style="color: {color}; font-weight: 500;">{category}</span>
                <span style="
                    background: {color};
                    color: #0d1117;
                    padding: 2px 8px;
                    border-radius: 10px;
                    font-size: 0.75em;
                    font-weight: bold;
                ">{len(tools)}</span>
            </div>
        """
    
    html += """
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 10px; background: #21262d; border-radius: 8px;">
            <p style="color: #8b949e; margin: 0; font-size: 0.9em;">
                💡 <strong style="color: #c9d1d9;">Gradio 6 MCP Server:</strong> 
                When <code style="background: #30363d; padding: 2px 6px; border-radius: 4px;">mcp_server=True</code> is set in 
                <code style="background: #30363d; padding: 2px 6px; border-radius: 4px;">demo.launch()</code>, 
                Gradio automatically exposes all functions with <code style="background: #30363d; padding: 2px 6px; border-radius: 4px;">@gr.tool</code> 
                decorator as MCP endpoints.
            </p>
        </div>
    </div>
    """
    
    return html


def add_mcp_endpoints_tab():
    """Add the MCP Endpoints tab to the Gradio interface."""
    with gr.TabItem(" MCP Endpoints", id="mcp-endpoints"):

        # MCP Health Status Section
        with gr.Accordion(" MCP Server Health Status", open=True):
            gr.Markdown("Check the health and availability of MCP endpoints and core dependencies.")
            health_btn = gr.Button(" Check Health Status", variant="primary")
            health_output = gr.HTML(value="<p style='color: #8b949e;'>Click 'Check Health Status' to view server health.</p>")
            health_btn.click(fn=check_mcp_health, inputs=[], outputs=[health_output])

        gr.Markdown("""
        #  Model Context Protocol (MCP) Endpoints

        This application exposes **26 quantum circuit tools** via the MCP protocol for AI agents and automation.
        All endpoints accept JSON parameters and return JSON results.

        ---

        ## 🛠️ Circuit Creation Tools
        """)

        with gr.Accordion("📦 **create_circuit** - Create from templates", open=False):
            gr.Markdown("""
            Creates quantum circuits from predefined templates.

            **Parameters:**
            - `template_name` (str): Template name (bell_state, ghz_state, qft, grover, vqe, qaoa)
            - `num_qubits` (int): Number of qubits (1-20)
            - `parameters_json` (str): JSON parameters object

            **Returns:**
            - `qasm` (str): OpenQASM 2.0 code
            - `circuit_diagram` (str): Visual representation
            - `num_gates`, `depth`: Circuit metrics

            **Example:**
            ```json
            {
              "template_name": "bell_state",
              "num_qubits": 2,
              "parameters_json": "{}"
            }
            ```
            """)

        with gr.Accordion("📝 **parse_qasm** - Parse OpenQASM", open=False):
            gr.Markdown("""
            Parse and validate OpenQASM 2.0 or 3.0 code.

            **Parameters:**
            - `qasm_code` (str): OpenQASM source code
            - `qasm_version` (str): "2.0" or "3.0"

            **Returns:**
            - Parsed circuit structure with gates, qubits, measurements
            - Gate sequence and metadata

            **Example:**
            ```qasm
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q -> c;
            ```
            """)

        with gr.Accordion("🔨 **build_circuit** - Build custom", open=False):
            gr.Markdown("""
            Build a circuit from JSON gate specifications.

            **Parameters:**
            - `num_qubits` (int): Number of qubits
            - `gates_json` (str): JSON array of gates
            - `measurements_json` (str): JSON array of measurements

            **Returns:**
            - Full circuit representation
            - Compiled circuit with verification

            **Gate JSON Example:**
            ```json
            [
              {"name": "h", "qubits": [0]},
              {"name": "cx", "qubits": [0, 1]},
              {"name": "measure", "qubits": [0, 1]}
            ]
            ```
            """)

        gr.Markdown("""
        ---

        ## ✅ Validation & Checking Tools
        """)

        with gr.Accordion("🔍 **validate_circuit** - Full validation", open=False):
            gr.Markdown("""
            Comprehensive circuit validation including syntax, connectivity, and unitarity.

            **Parameters:**
            - `qasm_code` (str): OpenQASM code
            - `hardware_target` (str): Optional hardware name
            - `check_connectivity` (bool): Check hardware connectivity
            - `check_unitary` (bool): Verify unitarity

            **Returns:**
            - `valid` (bool): Overall validation result
            - `errors` (array): List of errors
            - `warnings` (array): Warnings
            - `circuit_info`: Qubit count, gate count, depth
            """)

        with gr.Accordion("🖥️ **check_hardware** - Hardware compatibility", open=False):
            gr.Markdown("""
            Check circuit compatibility with specific quantum hardware.

            **Parameters:**
            - `qasm_code` (str): OpenQASM code
            - `hardware_name` (str): Hardware profile (ibm_brisbane, rigetti_aspen, ionq_harmony, etc.)

            **Returns:**
            - Connectivity violations
            - Native gate mapping
            - Required SWAP insertion points
            - Estimated execution time

            **Supported Hardware:**
            - IBM: brisbane, sherbrooke
            - Rigetti: aspen-m
            - Google: sycamore
            - IonQ: harmony
            - Quantinuum: h1
            """)

        gr.Markdown("""
        ---

        ## 🎲 Simulation & Evaluation Tools
        """)

        with gr.Accordion("🎯 **simulate** - Simulate with measurements", open=False):
            gr.Markdown("""
            Run circuit simulation with measurement sampling.

            **Parameters:**
            - `qasm_code` (str): OpenQASM code
            - `shots` (int): Number of measurement shots (100-10000)
            - `include_statevector` (bool): Include full statevector
            - `noise_model` (str): Optional noise model

            **Returns:**
            - `counts` (dict): Measurement counts {bitstring: count}
            - `probabilities` (dict): Normalized probabilities
            - `statevector` (dict): Full quantum state
            - `execution_time` (float): Simulation time
            """)

        with gr.Accordion("📐 **get_statevector** - Extract statevector", open=False):
            gr.Markdown("""
            Get the full statevector without measurement.

            **Parameters:**
            - `qasm_code` (str): OpenQASM code

            **Returns:**
            - `statevector` (dict): Complex amplitudes for all basis states
            - Format: {basis_state: {real: ..., imag: ...}}
            """)

        with gr.Accordion("📊 **estimate_fidelity** - Hardware fidelity", open=False):
            gr.Markdown("""
            Estimate circuit fidelity on target hardware.

            **Parameters:**
            - `qasm_code` (str): OpenQASM code
            - `hardware_name` (str): Target hardware

            **Returns:**
            - `estimated_fidelity` (float): 0-1 fidelity estimate
            - `error_breakdown`: Gate errors, readout errors
            - `decoherence_impact`: T1/T2 effects
            """)

        gr.Markdown("""
        ---

        ## 📈 Scoring & Analysis Tools
        """)

        with gr.Accordion("⭐ **score_circuit** - Circuit scoring", open=False):
            gr.Markdown("""
            Score circuit across multiple metrics.

            **Parameters:**
            - `qasm_code` (str): OpenQASM code
            - `hardware_name` (str): Optional target hardware

            **Returns:**
            - `overall_score` (0-1): Composite score
            - `complexity`: Gate count, depth, qubit usage
            - `expressibility`: Entanglement measures
            - `hardware_fitness`: Compatibility metrics
            - `estimated_fidelity`: Expected success rate
            """)

        with gr.Accordion("⚖️ **compare_circuits** - Compare multiple", open=False):
            gr.Markdown("""
            Compare multiple circuits across metrics.

            **Parameters:**
            - `circuits_json` (str): Array of QASM strings or gate specs
            - `hardware_name` (str): Optional target hardware

            **Returns:**
            - Comparative metrics for each circuit
            - Ranked by quality
            - Recommendations for best circuit
            """)

        gr.Markdown("""
        ---

        ## 📚 Documentation & Reference Tools
        """)

        with gr.Accordion("🔲 **get_gate_info** - Gate documentation", open=False):
            gr.Markdown("""
            Get detailed information about a quantum gate.

            **Parameters:**
            - `gate_name` (str): Gate name (h, x, cx, rx, etc.)

            **Returns:**
            - `description`: What the gate does
            - `matrix`: Matrix representation
            - `parameters`: Parameter explanations
            - `use_cases`: Common applications
            - `hardware_native`: Which platforms support it natively
            """)

        with gr.Accordion("🧮 **get_algorithm_info** - Algorithm documentation", open=False):
            gr.Markdown("""
            Get details about quantum algorithms.

            **Parameters:**
            - `algorithm_name` (str): Algorithm name (bell_state, qft, grover, vqe, qaoa)

            **Returns:**
            - `description`: Algorithm explanation
            - `circuit_template`: Reference circuit
            - `use_cases`: Applications
            - `complexity`: Time/space complexity
            - `resource_requirements`: Qubit and gate counts
            """)

        with gr.Accordion("📋 **list_hardware** - Available hardware", open=False):
            gr.Markdown("""
            List all supported quantum hardware backends.

            **Returns:**
            - Array of hardware profiles with:
              - Name, qubit count
              - Native gate set
              - Coupling map (connectivity)
              - Error rates
              - T1/T2 times
            """)

        with gr.Accordion("📚 **list_templates** - Available templates", open=False):
            gr.Markdown("""
            List all predefined circuit templates.

            **Returns:**
            - Template names
            - Descriptions
            - Parameter requirements
            - Use cases
            """)

        with gr.Accordion("🎓 **get_learning_path** - Educational content", open=False):
            gr.Markdown("""
            Get curated learning paths for different skill levels.

            **Parameters:**
            - `level` (str): "beginner", "intermediate", "advanced"

            **Returns:**
            - Learning objectives
            - Recommended circuits
            - Exercise descriptions
            - Resource links
            """)

        gr.Markdown("""
        ---

        ## 🚀 Quick Integration Examples

        ### Python Example
        ```python
        import requests
        import json

        # MCP Server URL
        url = "http://127.0.0.1:7861/api"

        # Create a Bell state circuit
        response = requests.post(f"{url}/create_circuit", json={
            "template_name": "bell_state",
            "num_qubits": 2,
            "parameters_json": "{}"
        })

        circuit = response.json()
        # Use circuit["qasm"] for further processing
        ```

        ### Validation Example
        ```python
        # Validate circuit
        response = requests.post(f"{url}/validate_circuit", json={
            "qasm_code": "OPENQASM 2.0;...",
            "hardware_target": "ibm_brisbane",
            "check_connectivity": True
        })

        result = response.json()
        # Check result['valid'] and result['errors'] for validation results
        ```

        ---

        **Total Endpoints:** 26 tools available
        **Protocol:** Model Context Protocol (MCP)
        **Format:** JSON input/output
        **Server:** http://127.0.0.1:7861
        """)