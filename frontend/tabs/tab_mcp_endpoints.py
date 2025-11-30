"""
MCP Endpoints Tab for QuantumArchitect-MCP
"""

import gradio as gr


def add_mcp_endpoints_tab():
    """Add the MCP Endpoints tab to the Gradio interface."""
    with gr.TabItem("🔗 MCP Endpoints", id="mcp-endpoints"):
        gr.Markdown("""
        # 🔗 Model Context Protocol (MCP) Endpoints

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