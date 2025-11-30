"""
Getting Started Tab for QuantumArchitect-MCP
"""

import gradio as gr


def add_getting_started_tab():
    """Add the Getting Started tab to the Gradio interface."""
    with gr.TabItem("🚀 Getting Started", id="getting-started"):
        gr.Markdown("""
        <div style="max-width: 900px; margin: 0 auto;">

        # 🚀 Getting Started with Quantum Circuits

        Welcome to **QuantumArchitect-MCP**! This guide will help you understand quantum computing
        basics and how to use this tool effectively.

        ---

        ## 🎯 Quick Start (5 Minutes)

        ### Step 1: Understand What a Qubit Is
        A **qubit** (quantum bit) is the basic unit of quantum information. Unlike classical bits (0 or 1),
        qubits can exist in **superposition** - being both 0 and 1 simultaneously until measured.

        ```
        Classical bit:  0 OR 1
        Qubit:          α|0⟩ + β|1⟩  (superposition of both states)
        ```

        ### Step 2: Learn the Basic Gates

        | Gate | What it Does | Analogy |
        |------|--------------|---------|
        | **H** (Hadamard) | Creates superposition | Flipping a coin in the air |
        | **X** (NOT) | Flips 0↔1 | Light switch |
        | **CX** (CNOT) | Controlled NOT | "If A is 1, flip B" |
        | **Measure** | Collapses to 0 or 1 | Catching the coin |

        ### Step 3: Build Your First Circuit

        1. Go to the **⚛️ Circuit Builder** tab
        2. Set **Qubit 1** to `0` in the left panel
        3. Click the **H** button to add a Hadamard gate
        4. Click **▶️ Simulate** to see the results

        **Expected Result:** You'll see 50% probability for |0⟩ and 50% for |1⟩ (superposition!)

        ---

        ## 📚 Skill Levels

        </div>
        """)

        with gr.Tabs():
            with gr.TabItem("🌱 Beginner"):
                gr.Markdown("""
                ### Beginner Concepts

                **What You'll Learn:**
                - Qubits and superposition
                - Basic single-qubit gates (H, X, Y, Z)
                - Measurement
                - Simple 2-qubit entanglement (Bell State)

                **Recommended First Circuits:**

                | Circuit | Description | Try This |
                |---------|-------------|----------|
                | **Single Hadamard** | H gate on qubit 0 | Shows 50/50 superposition |
                | **Bell State** | H on q0, then CX on q0→q1 | Creates entanglement |
                | **NOT Gate** | X gate on qubit 0 | Flips |0⟩ to |1⟩ |

                **Key Concepts:**

                🔹 **Superposition**: A qubit can be in multiple states at once
                🔹 **Measurement**: Observing a qubit forces it to choose 0 or 1
                🔹 **Entanglement**: Two qubits become correlated (Bell State)

                **Practice Exercise:**
                1. Go to Circuit Builder
                2. Add H gate to qubit 0
                3. Add CX gate with control=0, target=1
                4. Simulate and observe that |00⟩ and |11⟩ each have 50% probability
                """)

            with gr.TabItem("🔬 Intermediate"):
                gr.Markdown("""
                ### Intermediate Concepts

                **What You'll Learn:**
                - Phase gates (S, T, Z)
                - Rotation gates (Rx, Ry, Rz)
                - Multi-qubit circuits
                - GHZ and W states
                - Quantum Fourier Transform basics

                **Phase Gates Explained:**

                | Gate | Matrix | Effect |
                |------|--------|--------|
                | **Z** | diag(1, -1) | Flips phase of |1⟩ |
                | **S** | diag(1, i) | 90° phase rotation |
                | **T** | diag(1, e^(iπ/4)) | 45° phase rotation |

                **Rotation Gates:**
                - **Rx(θ)**: Rotates around X-axis by angle θ
                - **Ry(θ)**: Rotates around Y-axis by angle θ
                - **Rz(θ)**: Rotates around Z-axis by angle θ

                **Try These Circuits:**

                1. **GHZ State** (3 qubits): H(0), CX(0,1), CX(1,2)
                   - Creates |000⟩ + |111⟩ superposition

                2. **Phase Kickback**: H(0), H(1), CZ(0,1), H(0), H(1)
                   - Demonstrates phase relationships

                3. **Rotation Sequence**: Rx(π/2), Ry(π/2), Rz(π/2)
                   - Explore the Bloch sphere
                """)

            with gr.TabItem("🎓 Advanced"):
                gr.Markdown("""
                ### Advanced Concepts

                **What You'll Learn:**
                - Quantum Fourier Transform (QFT)
                - Grover's Search Algorithm
                - Variational Quantum Eigensolver (VQE)
                - QAOA for optimization
                - Error mitigation strategies
                - Hardware-aware circuit design

                **Quantum Fourier Transform:**
                ```
                QFT transforms computational basis states to frequency domain:
                |j⟩ → (1/√N) Σₖ e^(2πijk/N) |k⟩
                ```

                **Grover's Algorithm:**
                - Searches unsorted database in O(√N) time
                - Uses oracle + diffusion operator
                - Optimal iterations: ≈ π/4 × √N

                **VQE (Variational Quantum Eigensolver):**
                - Hybrid classical-quantum algorithm
                - Finds ground state energy of molecules
                - Uses parameterized circuits (ansatz)

                **Hardware Considerations:**
                - Gate fidelity and error rates
                - Qubit connectivity constraints
                - T1/T2 coherence times
                - Circuit depth limitations

                **Use the Templates tab** to generate these circuits automatically!
                """)

            with gr.TabItem("🔧 Professional"):
                gr.Markdown("""
                ### Professional & Research Topics

                **Topics Covered:**
                - Custom gate decomposition
                - Noise modeling and simulation
                - Quantum error correction codes
                - Transpilation strategies
                - Hardware backend optimization

                **Circuit Optimization Techniques:**

                | Technique | Description | Benefit |
                |-----------|-------------|---------|
                | Gate cancellation | Remove adjacent inverse gates | Reduces depth |
                | Commutation | Reorder commuting gates | Better scheduling |
                | Decomposition | Break complex gates into native set | Hardware compatibility |
                | Routing | Add SWAPs for connectivity | Executable circuits |

                **Using the MCP API:**

                This app exposes all functionality via MCP (Model Context Protocol) endpoints.
                AI agents can use these tools programmatically:

                ```python
                # Example MCP tool calls
                mcp_create_circuit("bell_state", 2, "{}")
                mcp_validate_circuit(qasm, "ibm_brisbane", True, True)
                mcp_simulate(qasm, 1024, True, "depolarizing")
                mcp_score_circuit(qasm, "ibm_brisbane")
                ```

                **Research Applications:**
                - Quantum chemistry simulations
                - Optimization problems (MaxCut, TSP)
                - Quantum machine learning
                - Cryptographic protocols
                """)