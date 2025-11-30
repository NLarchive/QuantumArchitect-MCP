"""
Circuit Transpilation Study Tab for QuantumArchitect-MCP

This module provides an interactive educational tab explaining circuit transpilation:
converting ideal circuits to hardware-executable form with native gates, 
connectivity constraints, and optimization.

Professional-level content with beginner-friendly explanations.
"""

import gradio as gr
import numpy as np
from typing import Tuple, Dict, List


def create_coupling_map_html(topology: str) -> str:
    """Create HTML visualization of hardware topology."""
    
    topologies = {
        "linear_5": {
            "qubits": 5,
            "edges": [(0,1), (1,2), (2,3), (3,4)],
            "description": "Linear chain - typical of early devices"
        },
        "ring_5": {
            "qubits": 5, 
            "edges": [(0,1), (1,2), (2,3), (3,4), (4,0)],
            "description": "Ring topology - better connectivity than linear"
        },
        "heavy_hex_7": {
            "qubits": 7,
            "edges": [(0,1), (1,2), (2,3), (3,4), (1,5), (3,6)],
            "description": "Heavy-hex (IBM) - optimized for error correction"
        },
        "grid_4x4": {
            "qubits": 16,
            "edges": [(i, i+1) for i in range(15) if i % 4 != 3] + 
                     [(i, i+4) for i in range(12)],
            "description": "Square grid - common for 2D architectures"
        },
        "all_to_all": {
            "qubits": 5,
            "edges": [(i, j) for i in range(5) for j in range(i+1, 5)],
            "description": "Full connectivity - trapped ions, ideal case"
        }
    }
    
    topo = topologies.get(topology, topologies["linear_5"])
    
    # Create ASCII representation
    if topology == "linear_5":
        ascii_art = """
        ●───●───●───●───●
        0   1   2   3   4
        """
    elif topology == "ring_5":
        ascii_art = """
        ●───●
       /     \\
      4       1
       \\     /
        ●───●───●
            2   3
        """
    elif topology == "heavy_hex_7":
        ascii_art = """
        0─●
          |
        1─●─5─●
          |
        2─●
          |
        3─●─6─●
          |
        4─●
        """
    elif topology == "grid_4x4":
        ascii_art = """
        0──1──2──3
        |  |  |  |
        4──5──6──7
        |  |  |  |
        8──9──10─11
        |  |  |  |
        12─13─14─15
        """
    else:  # all-to-all
        ascii_art = """
           0
          /|\\
         / | \\
        1──●──4
        |\\ | /|
        | \\|/ |
        2──●──3
        """
    
    html = f"""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 10px;">🔗 {topology.replace('_', ' ').title()}</h4>
        <pre style="color: #3fb950; font-family: monospace; font-size: 1.1em; margin: 15px 0;">
{ascii_art}
        </pre>
        <div style="background: #21262d; padding: 10px; border-radius: 6px;">
            <p style="color: #c9d1d9; margin: 0;">
                <strong>Qubits:</strong> {topo['qubits']} | 
                <strong>Connections:</strong> {len(topo['edges'])}
            </p>
            <p style="color: #8b949e; margin: 5px 0 0 0;">{topo['description']}</p>
        </div>
    </div>
    """
    return html


def show_swap_insertion_example() -> str:
    """Show example of SWAP insertion for non-adjacent qubits."""
    html = """
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">🔄 SWAP Insertion Example</h4>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <h5 style="color: #ff4081; margin-bottom: 10px;">❌ Original Circuit</h5>
                <pre style="color: #c9d1d9; margin: 0;">
q0 ─────●─────
        │
q1 ─────┼─────
        │
q2 ─────X─────

CNOT(q0, q2) - NOT ALLOWED!
(q0 and q2 not connected)
                </pre>
            </div>
            
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <h5 style="color: #3fb950; margin-bottom: 10px;">✓ After SWAP Insertion</h5>
                <pre style="color: #c9d1d9; margin: 0;">
q0 ─────●─────────────
        │
q1 ──X──X──X──●───X───
     │     │  │   │
q2 ──X─────X──X───X───
    └─SWAP─┘     └─SWAP─┘
     (move)      (restore)
                </pre>
            </div>
        </div>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-top: 15px;">
            <h5 style="color: #d29922; margin-bottom: 10px;">📊 Cost Analysis</h5>
            <table style="width: 100%; color: #c9d1d9; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #30363d;">
                    <td style="padding: 8px;">Original gates</td>
                    <td style="padding: 8px; text-align: right;">1 CNOT</td>
                </tr>
                <tr style="border-bottom: 1px solid #30363d;">
                    <td style="padding: 8px;">After transpilation</td>
                    <td style="padding: 8px; text-align: right;">7 CNOTs (1 + 2×3 for SWAPs)</td>
                </tr>
                <tr>
                    <td style="padding: 8px;">Overhead</td>
                    <td style="padding: 8px; text-align: right; color: #ff4081;">7× more gates!</td>
                </tr>
            </table>
        </div>
        
        <div style="background: #21262d; padding: 10px 15px; border-radius: 6px; margin-top: 15px; border-left: 4px solid #7c4dff;">
            <p style="color: #7c4dff; margin: 0;">
                💡 <strong>Key insight:</strong> SWAP gates are expensive (3 CNOTs each).
                Good qubit routing minimizes the number of SWAPs needed.
            </p>
        </div>
    </div>
    """
    return html


def show_gate_decomposition_example() -> str:
    """Show example of gate decomposition to native gates."""
    html = """
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">🔧 Gate Decomposition Examples</h4>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #3fb950; margin-bottom: 10px;">Toffoli (CCX) → Native Gates</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <p style="color: #8b949e; margin-bottom: 10px;">
                    The Toffoli gate decomposes into 6 CNOTs + single-qubit gates:
                </p>
                <pre style="color: #c9d1d9; margin: 0; font-size: 0.9em;">
Original:     ●──    Decomposed:  ─────────●───────────────●──T──●─────
              │                            │               │     │
              ●──    ───────────  ─────●───X───●───────●───X─────X──T†─
              │                        │       │       │
              X──                   H──X──T†───X──T────X──────────────H
                </pre>
                <p style="color: #ff4081; margin-top: 10px; margin-bottom: 0;">
                    6 CNOTs + 7 single-qubit gates!
                </p>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #7c4dff; margin-bottom: 10px;">SWAP → 3 CNOTs</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <pre style="color: #c9d1d9; margin: 0;">
SWAP = ─X─      ─●───X───●─
       │   =    │   │   │
      ─X─      ─X───●───X─
                </pre>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #d29922; margin-bottom: 10px;">Rz(θ) → IBM Native</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <p style="color: #c9d1d9; margin: 0;">
                    IBM native gate set: {√X, Rz, CNOT}<br><br>
                    All single-qubit gates decompose to: Rz(α) √X Rz(β) √X Rz(γ)
                </p>
            </div>
        </div>
        
        <div>
            <h5 style="color: #ff4081; margin-bottom: 10px;">Hadamard → Native</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <pre style="color: #c9d1d9; margin: 0;">
H = Rz(π/2) √X Rz(π/2)  (up to global phase)
                </pre>
            </div>
        </div>
    </div>
    """
    return html


def show_optimization_techniques() -> str:
    """Show circuit optimization techniques."""
    html = """
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">⚡ Optimization Techniques</h4>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #3fb950; margin-bottom: 10px;">1. Gate Cancellation</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <pre style="color: #c9d1d9; margin: 0;">
Before: ─H─H─X─X─Z─Z─   →   After: ─────────────
        (6 gates)                   (0 gates)

H² = X² = Z² = I
                </pre>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #7c4dff; margin-bottom: 10px;">2. Gate Fusion</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <pre style="color: #c9d1d9; margin: 0;">
Before: ─Rz(θ₁)─Rz(θ₂)─   →   After: ─Rz(θ₁+θ₂)─
        (2 gates)                    (1 gate)

Adjacent rotations about same axis combine
                </pre>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #d29922; margin-bottom: 10px;">3. Commutation</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <pre style="color: #c9d1d9; margin: 0;">
Before: ─Rz(θ)─●─   →   After: ─●─Rz(θ)─
              │              │
             ─X─            ─X─

Rz commutes through CNOT control
(allows more cancellation opportunities)
                </pre>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #ff4081; margin-bottom: 10px;">4. KAK Decomposition (for 2-qubit blocks)</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <p style="color: #c9d1d9; margin: 0;">
                    Any 2-qubit unitary can be decomposed into at most 3 CNOTs + single-qubit gates.<br><br>
                    Optimal decomposition uses Cartan/KAK decomposition.
                </p>
            </div>
        </div>
        
        <div>
            <h5 style="color: #58a6ff; margin-bottom: 10px;">5. Template Matching</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <pre style="color: #c9d1d9; margin: 0;">
Before: ─●───●───●─   →   After: ─●─
         │   │   │              │
        ─X───X───X─            ─X─

Three CNOTs on same qubits = one CNOT
                </pre>
            </div>
        </div>
    </div>
    """
    return html


def compare_transpilation_levels() -> str:
    """Compare different optimization levels."""
    html = """
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">📊 Transpilation Levels Comparison</h4>
        
        <table style="width: 100%; color: #c9d1d9; border-collapse: collapse; margin-bottom: 15px;">
            <tr style="background: #21262d; border-bottom: 1px solid #30363d;">
                <th style="padding: 12px; text-align: left;">Level</th>
                <th style="padding: 12px; text-align: left;">Optimizations</th>
                <th style="padding: 12px; text-align: right;">Time</th>
                <th style="padding: 12px; text-align: right;">Quality</th>
            </tr>
            <tr style="border-bottom: 1px solid #30363d;">
                <td style="padding: 12px; color: #8b949e;">0</td>
                <td style="padding: 12px;">Basis gates only, no optimization</td>
                <td style="padding: 12px; text-align: right; color: #3fb950;">⚡ Fastest</td>
                <td style="padding: 12px; text-align: right; color: #ff4081;">Poor</td>
            </tr>
            <tr style="border-bottom: 1px solid #30363d;">
                <td style="padding: 12px; color: #d29922;">1</td>
                <td style="padding: 12px;">Light: commutation, cancellation</td>
                <td style="padding: 12px; text-align: right; color: #3fb950;">Fast</td>
                <td style="padding: 12px; text-align: right; color: #d29922;">Medium</td>
            </tr>
            <tr style="border-bottom: 1px solid #30363d;">
                <td style="padding: 12px; color: #3fb950;">2</td>
                <td style="padding: 12px;">Medium: + noise-aware routing</td>
                <td style="padding: 12px; text-align: right; color: #d29922;">Medium</td>
                <td style="padding: 12px; text-align: right; color: #3fb950;">Good</td>
            </tr>
            <tr>
                <td style="padding: 12px; color: #7c4dff;">3</td>
                <td style="padding: 12px;">Heavy: all optimizations, synthesis</td>
                <td style="padding: 12px; text-align: right; color: #ff4081;">Slowest</td>
                <td style="padding: 12px; text-align: right; color: #3fb950;">Best</td>
            </tr>
        </table>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <h5 style="color: #3fb950; margin-bottom: 10px;">When to use Level 0-1</h5>
                <ul style="color: #c9d1d9; margin: 0; padding-left: 20px;">
                    <li>Development/debugging</li>
                    <li>Simulator runs</li>
                    <li>Time-critical applications</li>
                </ul>
            </div>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <h5 style="color: #7c4dff; margin-bottom: 10px;">When to use Level 2-3</h5>
                <ul style="color: #c9d1d9; margin: 0; padding-left: 20px;">
                    <li>Real hardware execution</li>
                    <li>When fidelity is critical</li>
                    <li>Deep circuits</li>
                </ul>
            </div>
        </div>
    </div>
    """
    return html


def add_transpilation_study_tab():
    """Add the Transpilation study tab to the Gradio interface."""
    
    with gr.Tab("🔧 Transpilation"):
        gr.Markdown("""
        <div style="text-align: center; padding: 15px 0;">
            <h2 style="color: #4fc3f7; margin-bottom: 8px;">Circuit Transpilation</h2>
            <p style="color: #8b949e;">Bridging Ideal Circuits and Real Hardware</p>
        </div>
        """)
        
        # Introduction
        with gr.Accordion("📚 What is Transpilation?", open=True):
            gr.Markdown(r"""
            ## From Ideal to Executable
            
            **Transpilation** converts an ideal quantum circuit into one that can actually run on 
            a specific quantum processor. This involves:
            
            1. **Basis Translation:** Convert all gates to hardware-native gates
            2. **Qubit Mapping:** Assign logical qubits to physical qubits
            3. **Routing:** Insert SWAP gates for non-adjacent interactions
            4. **Optimization:** Reduce gate count and depth
            
            ## Why Is It Necessary?
            
            | Ideal Circuit | Real Hardware |
            |--------------|---------------|
            | Any gate | Limited native gate set |
            | Any-to-any connectivity | Sparse connectivity |
            | Perfect gates | Noisy gates with errors |
            | Unlimited qubits | Fixed number of qubits |
            
            ## The Transpilation Pipeline
            
            ```
            Input          Stage 1        Stage 2        Stage 3        Output
            ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
            │ Ideal  │ → │ Basis  │ → │ Layout │ → │Routing │ → │Optimized│
            │Circuit │    │ Gates  │    │+ Map   │    │+ SWAP  │    │ Native │
            └────────┘    └────────┘    └────────┘    └────────┘    └────────┘
            ```
            
            ## Impact on Circuit Quality
            
            - **Depth increase:** Typically 2-10× due to routing and decomposition
            - **Gate count increase:** Similar increase
            - **Error accumulation:** More gates = more errors
            
            > ⚠️ **Critical:** A well-transpiled circuit can have 2-5× better fidelity than a naive one!
            """)
        
        # Hardware Topology
        with gr.Accordion("🔗 Hardware Connectivity", open=True):
            gr.Markdown("""
            ## Qubit Coupling Maps
            
            Real quantum processors have limited qubit connectivity. Two-qubit gates
            can only be performed between physically connected qubits.
            """)
            
            with gr.Row():
                topology_dropdown = gr.Dropdown(
                    choices=["linear_5", "ring_5", "heavy_hex_7", "grid_4x4", "all_to_all"],
                    value="linear_5",
                    label="Select Topology"
                )
                topo_btn = gr.Button("🔗 Show Topology", variant="primary")
            
            topo_output = gr.HTML()
            
            topo_btn.click(
                fn=create_coupling_map_html,
                inputs=[topology_dropdown],
                outputs=[topo_output]
            )
            
            gr.Markdown("""
            ### Connectivity Impact
            
            | Topology | Avg. Distance | SWAP Overhead | Use Case |
            |----------|---------------|---------------|----------|
            | Linear | O(n) | High | Simple devices |
            | Ring | O(n) | High | Slightly better |
            | Grid | O(√n) | Medium | 2D superconducting |
            | Heavy-hex | O(√n) | Medium | IBM devices |
            | All-to-all | O(1) | None | Trapped ions |
            """)
        
        # SWAP Insertion
        with gr.Accordion("🔄 SWAP Insertion", open=False):
            gr.Markdown("""
            ## Routing Non-Adjacent Gates
            
            When a two-qubit gate needs to act on non-connected qubits,
            we must insert SWAP gates to move qubit states around.
            """)
            
            swap_btn = gr.Button("🔄 Show SWAP Example", variant="secondary")
            swap_output = gr.HTML()
            
            swap_btn.click(fn=show_swap_insertion_example, inputs=[], outputs=[swap_output])
            
            gr.Markdown(r"""
            ### SWAP Routing Algorithms
            
            | Algorithm | Quality | Speed | Description |
            |-----------|---------|-------|-------------|
            | **Basic** | Low | Fast | Greedy nearest-neighbor |
            | **Stochastic** | Medium | Medium | Random search + heuristics |
            | **SABRE** | High | Medium | Look-ahead bidirectional |
            | **Optimal (A*)** | Best | Slow | Exhaustive search |
            
            ### Reducing SWAP Count
            
            1. **Initial layout optimization:** Place interacting qubits close together
            2. **Gate reordering:** Commute gates to cluster interactions
            3. **Bridge gates:** Use intermediate qubits for 2-hop CNOTs
            """)
        
        # Gate Decomposition
        with gr.Accordion("🔧 Gate Decomposition", open=False):
            gr.Markdown("""
            ## Native Gate Sets
            
            Different hardware platforms have different native gates:
            
            | Platform | Native 1Q Gates | Native 2Q Gate |
            |----------|----------------|----------------|
            | IBM | √X, Rz | ECR or CX |
            | Rigetti | Rx, Rz | CZ |
            | IonQ | Rxy, Rz | XX (Mølmer-Sørensen) |
            | Google | √X, Rz | √iSWAP |
            """)
            
            decomp_btn = gr.Button("🔧 Show Decomposition Examples", variant="secondary")
            decomp_output = gr.HTML()
            
            decomp_btn.click(fn=show_gate_decomposition_example, inputs=[], outputs=[decomp_output])
        
        # Optimization
        with gr.Accordion("⚡ Circuit Optimization", open=False):
            gr.Markdown("""
            ## Reducing Gate Count and Depth
            
            After routing and decomposition, circuits are often bloated.
            Optimization passes reduce this overhead.
            """)
            
            opt_btn = gr.Button("⚡ Show Optimization Techniques", variant="secondary")
            opt_output = gr.HTML()
            
            opt_btn.click(fn=show_optimization_techniques, inputs=[], outputs=[opt_output])
        
        # Optimization Levels
        with gr.Accordion("📊 Optimization Levels", open=False):
            gr.Markdown("""
            ## Choosing the Right Level
            
            Most transpilers offer multiple optimization levels trading off
            compilation time vs circuit quality.
            """)
            
            levels_btn = gr.Button("📊 Compare Levels", variant="secondary")
            levels_output = gr.HTML()
            
            levels_btn.click(fn=compare_transpilation_levels, inputs=[], outputs=[levels_output])
        
        # Best Practices
        with gr.Accordion("📋 Best Practices", open=False):
            gr.Markdown(r"""
            ## Transpilation Tips
            
            ### 1. Know Your Hardware
            - Check native gate set and connectivity
            - Understand error rates per qubit/gate
            - Know T1/T2 coherence times
            
            ### 2. Design Hardware-Aware Circuits
            - Use native gates when possible
            - Minimize long-range interactions
            - Consider hardware topology in algorithm design
            
            ### 3. Optimize Iteratively
            ```
            1. Start with optimization level 1
            2. Check depth and gate count
            3. If too deep, try level 2-3
            4. Compare multiple initial layouts
            5. Profile on simulator before hardware
            ```
            
            ### 4. Layout Strategies
            
            | Strategy | When to Use |
            |----------|-------------|
            | Trivial | Quick tests, simulators |
            | Dense | Circuits with many interactions |
            | Noise-aware | Production runs |
            | Custom | Domain-specific knowledge |
            
            ### 5. Verify Transpilation
            
            - Check that output circuit is equivalent to input
            - Verify gate count is reasonable
            - Simulate both versions to compare results
            
            ## Common Pitfalls
            
            ❌ **Ignoring connectivity:** Leads to massive SWAP overhead
            
            ❌ **Over-optimization:** Can take hours for marginal gains
            
            ❌ **Wrong native gates:** Leads to unnecessary decomposition
            
            ❌ **Ignoring noise:** Optimal depth ≠ optimal fidelity
            """)
        
        # Quick Reference
        with gr.Accordion("📋 Quick Reference", open=False):
            gr.Markdown(r"""
            ## Key Concepts
            
            | Term | Definition |
            |------|------------|
            | **Basis gates** | Native gates the hardware can execute |
            | **Coupling map** | Graph of allowed two-qubit interactions |
            | **Layout** | Mapping of logical to physical qubits |
            | **Routing** | Inserting SWAPs for connectivity |
            | **Synthesis** | Optimal decomposition of unitaries |
            
            ## Typical Gate Costs
            
            | Gate | CNOT Equivalent | Typical Error |
            |------|-----------------|---------------|
            | Single-qubit | ~0 | 0.01-0.1% |
            | CNOT/CZ | 1 | 0.3-2% |
            | SWAP | 3 | 1-5% |
            | Toffoli | 6 | 2-10% |
            | 4-qubit gate | 15+ | High |
            
            ## Depth Formulas
            
            For circuits with d ideal 2-qubit gates on n qubits:
            
            - **Linear topology:** Depth ≈ d × O(n)
            - **Grid topology:** Depth ≈ d × O(√n)
            - **Full connectivity:** Depth ≈ d × O(1)
            
            ## Tools
            
            - **Qiskit:** `transpile()` with optimization_level
            - **Cirq:** `optimize_for_target_gateset()`
            - **tket:** Multiple optimization passes
            - **BQSKit:** Optimal synthesis
            """)
