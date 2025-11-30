---
title: QuantumArchitect MCP
emoji: ⚛️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
pinned: false
license: mit
short_description: Quantum Circuit Architect & MCP Server for AI Agents
tags:
  - building-mcp-track-enterprise
  - building-mcp-track-consumer
  - building-mcp-track-creative 
---

# QuantumArchitect-MCP 🔬⚛️

A Python-based MCP (Model Context Protocol) Server for Quantum Circuit creation, validation, and evaluation. This serves as a "Quantum Logic Engine" that AI Agents can call upon to validate, score, and execute quantum logic.

## 🚀 Features

- **Circuit Creation**: Generate Bell States, GHZ States, QFT, Grover's Algorithm, and VQE Ansatz circuits
- **Circuit Validation**: Syntax checking, connectivity validation for real hardware, unitarity verification
- **Circuit Evaluation**: Statevector simulation, noise estimation, resource estimation
- **Circuit Scoring**: Complexity metrics, expressibility scores, hardware fitness evaluation
- **MCP Endpoints**: Full MCP protocol support for AI Agent integration
- **Hardware Profiles**: Support for IBM, Rigetti, and other quantum hardware topologies

## 📦 Installation

### For Hugging Face Spaces
This project is designed to run directly on Hugging Face Spaces. Simply clone and deploy!

### Local Installation
```bash
pip install -r requirements.txt
python app.py
```

## 🚀 Quick Start

### 1. Start the Application
```bash
python app.py
```
The app will start at `http://127.0.0.1:7861`

### 2. Build Your First Circuit
1. Open the web interface in your browser
2. Go to the "Circuit Builder" tab
3. Click the "H" button to add a Hadamard gate
4. Click "Simulate" to see the results
5. View the Bloch sphere visualization of the qubit state

### 3. Try a Bell State
1. Go to the "Templates" tab
2. Select "Bell State" from the dropdown
3. Click "Load Template"
4. Click "Simulate" to see entangled output (50/50 probabilities)

### 4. Validate a Circuit
1. Go to the "Validate" tab
2. Paste or enter QASM code
3. Select target hardware (e.g., "ibm_eagle")
4. Click "Validate" to check syntax, connectivity, and unitarity

## 🔧 Project Structure

```
QuantumArchitect-MCP/
├── app.py                          # Main entry point (Gradio + MCP)
├── requirements.txt                # Dependencies
├── pyproject.toml                  # Project configuration
├── src/
│   ├── mcp_server/                 # MCP Protocol handling
│   │   ├── server.py               # MCP capabilities definition
│   │   ├── schemas.py              # JSON schemas for I/O
│   │   └── context_provider.py     # Resource providers
│   ├── core/                       # Quantum engine core
│   │   ├── circuit_parser.py       # QASM/circuit parsing
│   │   ├── dag_representation.py   # Internal DAG model
│   │   └── exceptions.py           # Custom exceptions
│   ├── plugins/                    # Modular components
│   │   ├── creation/               # Circuit generation
│   │   ├── validation/             # Circuit validation
│   │   ├── evaluation/             # Circuit evaluation
│   │   └── scoring/                # Circuit scoring
│   └── data/                       # Knowledge base
│       ├── hardware_profiles/      # Hardware topology configs
│       └── reference_circuits/     # Standard algorithm references
└── tests/                          # Test suite
```

## 🎯 MCP Endpoints

### Creation Tools
- `create_bell_state`: Generate a 2-qubit Bell state circuit
- `create_ghz_state`: Generate an N-qubit GHZ state
- `create_qft`: Generate Quantum Fourier Transform circuit
- `create_grover`: Generate Grover's search algorithm
- `create_vqe_ansatz`: Generate VQE variational ansatz

### Validation Tools
- `validate_syntax`: Check circuit syntax validity
- `validate_connectivity`: Verify hardware topology compatibility
- `validate_unitarity`: Check if circuit is properly unitary

### Evaluation Tools
- `simulate_statevector`: Get ideal simulation results
- `estimate_noise`: Estimate circuit noise accumulation
- `estimate_resources`: Calculate required shots and resources

### Scoring Tools
- `score_complexity`: Get circuit depth, gate count, width
- `score_expressibility`: Evaluate VQC expressibility (QML)
- `score_hardware_fitness`: Rate circuit for specific hardware

## 🖥️ Usage

### Web Interface
Access the Gradio UI at the deployed URL or `http://localhost:7860` for local runs.

### MCP Integration
Connect your AI Agent to the MCP endpoints:

```python
# Example: Claude Desktop configuration
{
    "mcpServers": {
        "quantum-architect": {
            "url": "https://your-space.hf.space/mcp"
        }
    }
}
```

## 📚 Learning Path Integration

This tool follows the "Zero to Hero" quantum computing curriculum:

1. **Level 0 (Beginner)**: Use creation templates (Bell, GHZ states)
2. **Level 1 (Practitioner)**: Validate circuits against real hardware
3. **Level 2 (Advanced)**: Evaluate noise and optimize for NISQ devices
4. **Level 3 (PhD/Hero)**: Score expressibility and develop new algorithms

## 🤖 AI Agent Integration

### Available MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `mcp_create_circuit` | Create from template | `template_name`, `num_qubits`, `parameters_json` |
| `mcp_parse_qasm` | Parse OpenQASM code | `qasm_code`, `qasm_version` |
| `mcp_build_circuit` | Build custom circuit | `num_qubits`, `gates_json`, `measurements_json` |
| `mcp_validate_circuit` | Validate circuit | `qasm_code`, `hardware_target`, `check_connectivity`, `check_unitary` |
| `mcp_check_hardware` | Check hardware compatibility | `qasm_code`, `hardware_name` |
| `mcp_simulate` | Simulate circuit | `qasm_code`, `shots`, `include_statevector`, `noise_model` |
| `mcp_get_statevector` | Get ideal statevector | `qasm_code` |
| `mcp_estimate_fidelity` | Estimate hardware fidelity | `qasm_code`, `hardware_name` |
| `mcp_score_circuit` | Score circuit | `qasm_code`, `hardware_name` |
| `mcp_compare_circuits` | Compare multiple circuits | `circuits_json`, `hardware_name` |
| `mcp_get_gate_info` | Gate documentation | `gate_name` |
| `mcp_get_algorithm_info` | Algorithm explanation | `algorithm_name` |
| `mcp_list_hardware` | List hardware profiles | - |
| `mcp_list_templates` | List circuit templates | - |
| `mcp_get_learning_path` | Get learning resources | `level` |

### Supported Hardware Profiles

- **IBM Eagle** (127 qubits, heavy-hex topology)
- **Rigetti Aspen** (80 qubits, octagonal topology)
- **IonQ Aria** (25 qubits, all-to-all connectivity)

### Circuit Templates

- `bell_state` - Maximally entangled 2-qubit state
- `ghz_state` - N-qubit GHZ entangled state
- `w_state` - N-qubit W state
- `superposition` - Uniform superposition
- `qft` - Quantum Fourier Transform
- `grover` - Grover's search algorithm
- `vqe` - VQE variational ansatz
- `qaoa` - QAOA optimization circuit

## 🧪 Running Tests

```bash
pytest tests/ -v
```

## 📄 License

MIT License - See LICENSE file for details.

## 🙏 Acknowledgments

Built with:
- [Gradio](https://gradio.app/) - UI and MCP integration
- [Qiskit](https://qiskit.org/) - Quantum computing framework
- [Pydantic](https://pydantic.dev/) - Data validation
