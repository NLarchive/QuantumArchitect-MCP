"""
Grover's Algorithm Study Tab for QuantumArchitect-MCP

This module provides an interactive educational tab explaining Grover's quantum search
algorithm including oracle construction, amplitude amplification, and the diffusion operator.

Professional-level content with beginner-friendly explanations.
"""

import gradio as gr
import numpy as np
from typing import Tuple, Dict, List


def create_initial_state(n_qubits: int) -> np.ndarray:
    """Create the initial uniform superposition state."""
    N = 2**n_qubits
    return np.ones(N, dtype=complex) / np.sqrt(N)


def create_oracle(n_qubits: int, marked_states: List[int]) -> np.ndarray:
    """Create the oracle matrix that marks target states."""
    N = 2**n_qubits
    oracle = np.eye(N, dtype=complex)
    for marked in marked_states:
        if 0 <= marked < N:
            oracle[marked, marked] = -1
    return oracle


def create_diffusion_operator(n_qubits: int) -> np.ndarray:
    """Create the diffusion (Grover) operator: 2|s⟩⟨s| - I."""
    N = 2**n_qubits
    # |s⟩ is the uniform superposition
    s = np.ones(N, dtype=complex) / np.sqrt(N)
    # 2|s⟩⟨s| - I
    diffusion = 2 * np.outer(s, s.conj()) - np.eye(N)
    return diffusion


def run_grover_iteration(statevector: np.ndarray, oracle: np.ndarray, 
                          diffusion: np.ndarray) -> np.ndarray:
    """Apply one Grover iteration: Oracle followed by Diffusion."""
    # Apply oracle
    state_after_oracle = oracle @ statevector
    # Apply diffusion
    state_after_diffusion = diffusion @ state_after_oracle
    return state_after_diffusion


def get_optimal_iterations(n_qubits: int, num_marked: int = 1) -> int:
    """Calculate the optimal number of Grover iterations."""
    N = 2**n_qubits
    if num_marked >= N:
        return 0
    theta = np.arcsin(np.sqrt(num_marked / N))
    optimal = int(np.round(np.pi / (4 * theta) - 0.5))
    return max(1, optimal)


def format_probability_html(statevector: np.ndarray, marked_states: List[int], 
                           n_qubits: int, iteration: int) -> str:
    """Create HTML visualization of current state probabilities."""
    probabilities = np.abs(statevector)**2
    
    html = f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h4 style="color: #4fc3f7; margin-bottom: 12px;">📊 State Probabilities (Iteration {iteration})</h4>
    """
    
    # Sort by probability for display
    sorted_indices = np.argsort(probabilities)[::-1]
    
    # Show top states
    shown = 0
    for i in sorted_indices:
        if shown >= 8 and probabilities[i] < 0.01:
            break
        
        bitstring = format(i, f'0{n_qubits}b')
        prob = probabilities[i]
        is_marked = i in marked_states
        
        color = '#3fb950' if is_marked else '#4fc3f7'
        marker = '🎯' if is_marked else ''
        width = max(prob * 100, 1)
        
        html += f"""
        <div style="margin: 4px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="color: #c9d1d9; font-family: monospace; width: 60px;">|{bitstring}⟩{marker}</span>
                <div style="flex: 1; background: #21262d; border-radius: 4px; height: 20px; position: relative;">
                    <div style="width: {width}%; background: linear-gradient(90deg, {color}, {color}88); 
                                height: 100%; border-radius: 4px;"></div>
                </div>
                <span style="color: #8b949e; width: 60px; text-align: right;">{prob*100:.1f}%</span>
            </div>
        </div>
        """
        shown += 1
    
    if shown < len(statevector):
        remaining = len(statevector) - shown
        html += f"""
        <p style="color: #8b949e; font-size: 0.9em; margin-top: 8px;">
            ...and {remaining} more states with low probability
        </p>
        """
    
    html += "</div>"
    return html


def run_grover_demo(n_qubits: int, marked_state_str: str, num_iterations: int) -> Tuple[str, str, str]:
    """Run Grover's algorithm demo."""
    # Parse marked state
    try:
        marked_state = int(marked_state_str, 2) if marked_state_str else 0
    except:
        marked_state = 0
    
    N = 2**n_qubits
    marked_states = [marked_state % N]
    
    # Initialize
    state = create_initial_state(n_qubits)
    oracle = create_oracle(n_qubits, marked_states)
    diffusion = create_diffusion_operator(n_qubits)
    optimal = get_optimal_iterations(n_qubits, len(marked_states))
    
    # Collect iteration history
    history = [(0, np.abs(state)**2)]
    
    for i in range(num_iterations):
        state = run_grover_iteration(state, oracle, diffusion)
        history.append((i + 1, np.abs(state)**2))
    
    # Initial state visualization
    initial_html = format_probability_html(
        create_initial_state(n_qubits), marked_states, n_qubits, 0
    )
    
    # Final state visualization
    final_html = format_probability_html(state, marked_states, n_qubits, num_iterations)
    
    # Analysis
    marked_prob = sum(np.abs(state[m])**2 for m in marked_states)
    
    analysis_html = f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px;">
        <h4 style="color: #7c4dff; margin-bottom: 12px;">📈 Analysis</h4>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 15px;">
            <div style="background: #21262d; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e; font-size: 0.9em;">Search Space Size</div>
                <div style="color: #4fc3f7; font-size: 1.5em;">N = {N}</div>
            </div>
            <div style="background: #21262d; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e; font-size: 0.9em;">Target State</div>
                <div style="color: #3fb950; font-size: 1.5em;">|{format(marked_state % N, f'0{n_qubits}b')}⟩</div>
            </div>
            <div style="background: #21262d; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e; font-size: 0.9em;">Optimal Iterations</div>
                <div style="color: #d29922; font-size: 1.5em;">≈ {optimal}</div>
            </div>
            <div style="background: #21262d; padding: 12px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e; font-size: 0.9em;">Target Probability</div>
                <div style="color: {'#3fb950' if marked_prob > 0.5 else '#ff4081'}; font-size: 1.5em;">{marked_prob*100:.1f}%</div>
            </div>
        </div>
        
        <div style="background: #21262d; padding: 12px; border-radius: 6px;">
            <p style="color: #c9d1d9; margin: 0;">
                <strong>Speedup:</strong> Classical search requires O(N) = O({N}) queries on average.
                Grover achieves O(√N) = O({int(np.sqrt(N))}) queries.
                <br>
                <strong>Quadratic speedup:</strong> {N // max(1, int(np.sqrt(N)))}× faster than classical!
            </p>
        </div>
    </div>
    """

    return initial_html, final_html, analysis_html


def generate_grover_qasm(n_qubits: int, marked_state_str: str, num_iterations: int) -> tuple:
    """Generate OpenQASM 2.0 code for Grover's algorithm."""
    try:
        marked_state = int(marked_state_str, 2) if marked_state_str else 0
    except:
        marked_state = 0
    
    N = 2**n_qubits
    marked_state = marked_state % N
    
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n_qubits}];",
        f"creg c[{n_qubits}];",
        "",
        "// Initialize uniform superposition"
    ]
    
    for i in range(n_qubits):
        lines.append(f"h q[{i}];")
    
    for iteration in range(num_iterations):
        lines.append(f"\n// --- Grover Iteration {iteration + 1} ---")
        lines.append(f"// Oracle: mark state |{format(marked_state, f'0{n_qubits}b')}⟩")
        
        # Oracle: flip bits where marked_state has 0, apply multi-controlled Z, flip back
        for i in range(n_qubits):
            if not (marked_state >> i) & 1:
                lines.append(f"x q[{i}];")
        
        # Multi-controlled Z (simplified for 2-3 qubits)
        if n_qubits == 2:
            lines.append("cz q[0],q[1];")
        elif n_qubits >= 3:
            lines.append(f"h q[{n_qubits-1}];")
            if n_qubits == 3:
                lines.append("ccx q[0],q[1],q[2];")
            else:
                lines.append(f"// MCZ on {n_qubits} qubits (simplified)")
            lines.append(f"h q[{n_qubits-1}];")
        
        for i in range(n_qubits):
            if not (marked_state >> i) & 1:
                lines.append(f"x q[{i}];")
        
        lines.append("\n// Diffusion operator")
        for i in range(n_qubits):
            lines.append(f"h q[{i}];")
        for i in range(n_qubits):
            lines.append(f"x q[{i}];")
        
        if n_qubits == 2:
            lines.append("cz q[0],q[1];")
        elif n_qubits >= 3:
            lines.append(f"h q[{n_qubits-1}];")
            if n_qubits == 3:
                lines.append("ccx q[0],q[1],q[2];")
            else:
                lines.append(f"// MCZ on {n_qubits} qubits")
            lines.append(f"h q[{n_qubits-1}];")
        
        for i in range(n_qubits):
            lines.append(f"x q[{i}];")
        for i in range(n_qubits):
            lines.append(f"h q[{i}];")
    
    lines.append("\n// Measurement")
    for i in range(n_qubits):
        lines.append(f"measure q[{i}] -> c[{i}];")
    
    qasm = "\n".join(lines)
    
    # Generate ASCII diagram
    diagram_lines = []
    for i in range(n_qubits):
        line = f"q{i}: ─[H]─"
        for _ in range(num_iterations):
            line += "┤Oracle├─┤Diffusion├─"
        line += "─[M]─"
        diagram_lines.append(line)
    diagram = "\n".join(diagram_lines)
    
    return qasm, diagram


def visualize_amplitude_evolution(n_qubits: int,marked_state_str: str, max_iterations: int) -> str:
    """Create visualization showing amplitude evolution over iterations."""
    try:
        marked_state = int(marked_state_str, 2) if marked_state_str else 0
    except:
        marked_state = 0
    
    N = 2**n_qubits
    marked_states = [marked_state % N]
    
    # Initialize
    state = create_initial_state(n_qubits)
    oracle = create_oracle(n_qubits, marked_states)
    diffusion = create_diffusion_operator(n_qubits)
    
    # Track probabilities
    marked_probs = [np.abs(state[marked_states[0]])**2]
    unmarked_probs = [1 - marked_probs[0]]
    
    for i in range(max_iterations):
        state = run_grover_iteration(state, oracle, diffusion)
        marked_probs.append(np.abs(state[marked_states[0]])**2)
        unmarked_probs.append(1 - marked_probs[-1])
    
    # Create ASCII-style evolution chart
    html = f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 12px;">📈 Amplitude Evolution</h4>
        <div style="font-family: monospace; font-size: 0.9em;">
    """
    
    for i, (mp, up) in enumerate(zip(marked_probs, unmarked_probs)):
        marked_bar = '█' * int(mp * 40)
        unmarked_bar = '░' * int(up * 40)
        
        html += f"""
        <div style="display: flex; gap: 10px; margin: 3px 0;">
            <span style="color: #8b949e; width: 30px;">t={i}</span>
            <span style="color: #3fb950;">{marked_bar}</span>
            <span style="color: #30363d;">{unmarked_bar}</span>
            <span style="color: #8b949e; width: 50px;">{mp*100:.0f}%</span>
        </div>
        """
    
    html += """
        </div>
        <div style="margin-top: 10px;">
            <span style="color: #3fb950;">█ Target state</span>
            <span style="color: #8b949e; margin-left: 20px;">░ Other states</span>
        </div>
    </div>
    """
    
    return html


def explain_oracle(n_qubits: int, marked_state_str: str) -> str:
    """Explain how the oracle works."""
    try:
        marked_state = int(marked_state_str, 2) if marked_state_str else 0
    except:
        marked_state = 0
    
    N = 2**n_qubits
    marked_state = marked_state % N
    
    html = f"""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">🔮 The Oracle Operation</h4>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <p style="color: #c9d1d9; margin: 0;">
                The oracle $U_f$ is a <strong>black box</strong> that "knows" the solution.
                It flips the phase of the target state while leaving others unchanged.
            </p>
        </div>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <h5 style="color: #3fb950; margin-bottom: 10px;">Mathematical Definition</h5>
            <p style="color: #c9d1d9; font-family: monospace;">
                U<sub>f</sub>|x⟩ = (-1)<sup>f(x)</sup>|x⟩
            </p>
            <p style="color: #8b949e; margin-top: 10px;">
                where f(x) = 1 if x is the target, 0 otherwise
            </p>
        </div>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px;">
            <h5 style="color: #d29922; margin-bottom: 10px;">For target |{format(marked_state, f'0{n_qubits}b')}⟩</h5>
            <table style="width: 100%; color: #c9d1d9; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #30363d;">
                    <th style="text-align: left; padding: 8px;">Input State</th>
                    <th style="text-align: center; padding: 8px;">f(x)</th>
                    <th style="text-align: left; padding: 8px;">Output State</th>
                </tr>
    """
    
    # Show a few example transformations
    for i in range(min(N, 4)):
        bitstring = format(i, f'0{n_qubits}b')
        is_target = i == marked_state
        f_x = 1 if is_target else 0
        sign = "-" if is_target else "+"
        
        row_color = "color: #3fb950;" if is_target else ""
        html += f"""
                <tr style="border-bottom: 1px solid #30363d; {row_color}">
                    <td style="padding: 8px; font-family: monospace;">|{bitstring}⟩</td>
                    <td style="text-align: center; padding: 8px;">{f_x}</td>
                    <td style="padding: 8px; font-family: monospace;">{sign}|{bitstring}⟩</td>
                </tr>
        """
    
    if N > 4:
        html += f"""
                <tr>
                    <td colspan="3" style="padding: 8px; color: #8b949e;">...{N-4} more states unchanged...</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <div style="background: #21262d; padding: 10px 15px; border-radius: 6px; margin-top: 15px; border-left: 4px solid #7c4dff;">
            <p style="color: #7c4dff; margin: 0;">
                💡 <strong>Key insight:</strong> The phase flip is invisible if we measure immediately!
                We need the diffusion operator to convert phase differences into amplitude differences.
            </p>
        </div>
    </div>
    """
    
    return html


def explain_diffusion() -> str:
    """Explain the diffusion operator."""
    html = """
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">🌊 The Diffusion Operator</h4>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <p style="color: #c9d1d9; margin: 0;">
                Also called the <strong>Grover diffusion operator</strong> or <strong>inversion about the mean</strong>.
                It amplifies states with higher-than-average amplitude and suppresses states with lower-than-average amplitude.
            </p>
        </div>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <h5 style="color: #3fb950; margin-bottom: 10px;">Mathematical Definition</h5>
            <p style="color: #c9d1d9; font-family: monospace; font-size: 1.1em;">
                D = 2|s⟩⟨s| - I
            </p>
            <p style="color: #8b949e; margin-top: 10px;">
                where |s⟩ = H<sup>⊗n</sup>|0⟩<sup>⊗n</sup> is the uniform superposition
            </p>
        </div>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <h5 style="color: #d29922; margin-bottom: 10px;">Inversion About the Mean</h5>
            <p style="color: #c9d1d9;">
                For each amplitude α<sub>i</sub>, the new amplitude is:
            </p>
            <p style="color: #4fc3f7; font-family: monospace; font-size: 1.1em; margin: 10px 0;">
                α'<sub>i</sub> = 2⟨α⟩ - α<sub>i</sub>
            </p>
            <p style="color: #8b949e;">
                where ⟨α⟩ is the average of all amplitudes
            </p>
        </div>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px;">
            <h5 style="color: #ff4081; margin-bottom: 10px;">Geometric Visualization</h5>
            <pre style="color: #c9d1d9; margin: 0;">
                    mean
        ─────────────┼─────────────
                     │
        Before:  ▼   │      ▽ ▽ ▽ ▽
        (target ▼ has negative amplitude from oracle)
        
        After:   △   │  ▽ ▽ ▽ ▽
        (target △ reflected to positive, others diminished)
            </pre>
        </div>
        
        <div style="background: #21262d; padding: 10px 15px; border-radius: 6px; margin-top: 15px; border-left: 4px solid #7c4dff;">
            <p style="color: #7c4dff; margin: 0;">
                💡 <strong>Key insight:</strong> The oracle makes the target amplitude negative (below mean).
                The diffusion operator then reflects it above the mean, making it larger than the others!
            </p>
        </div>
    </div>
    """
    
    return html


def add_grover_study_tab():
    """Add the Grover's Algorithm study tab to the Gradio interface."""
    
    with gr.Tab("🔍 Grover"):
        gr.Markdown("""
        <div style="text-align: center; padding: 15px 0;">
            <h2 style="color: #4fc3f7; margin-bottom: 8px;">Grover's Search Algorithm</h2>
            <p style="color: #8b949e;">Quantum Search with Quadratic Speedup</p>
        </div>
        """)
        
        # Introduction
        with gr.Accordion("📚 Introduction to Grover's Algorithm", open=True):
            gr.Markdown(r"""
            ## The Unstructured Search Problem
            
            **Problem:** Find a marked item in an unsorted database of N items.
            
            | Algorithm | Queries Required | Complexity |
            |-----------|-----------------|------------|
            | **Classical** | N/2 average, N worst case | O(N) |
            | **Grover's** | ~π√N/4 | O(√N) |
            
            ### Quadratic Speedup
            
            For a database of 1 million items:
            - **Classical:** ~500,000 queries on average
            - **Grover's:** ~785 queries!
            
            This is a **quadratic speedup** - not exponential, but still significant!
            
            ## Algorithm Overview
            
            1. **Initialize:** Create uniform superposition of all N states
            2. **Repeat** ~√N times:
               - Apply **Oracle** (marks the solution with phase flip)
               - Apply **Diffusion** (amplitude amplification)
            3. **Measure:** High probability of getting the marked state
            
            ## Circuit Structure
            
            ```
            |0⟩ ─[H]─┬──────────────────────────┬─[H]─[X]─■─[X]─[H]─ ... ─[M]
                     │                          │         │
            |0⟩ ─[H]─┼──────────────────────────┼─[H]─[X]─■─[X]─[H]─ ... ─[M]
                     │       ORACLE             │         │
            |0⟩ ─[H]─┼──[ marks solution ]──────┼─[H]─[X]─■─[X]─[H]─ ... ─[M]
                     │                          │    DIFFUSION
            |1⟩ ─[H]─┴──────────────────────────┴─────────────────── ... ───
                    (auxiliary qubit for phase kickback)
            ```
            """)
        
        # Interactive Demo
        with gr.Accordion("🎮 Interactive Grover Simulator", open=True):
            gr.Markdown("""
            ## Try Grover's Algorithm
            
            Set up a search problem and watch the algorithm find the solution!
            """)
            
            with gr.Row():
                n_qubits_slider = gr.Slider(
                    minimum=2,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of Qubits (Search space = 2^n)"
                )
                marked_input = gr.Textbox(
                    value="101",
                    label="Target State (binary)",
                    placeholder="e.g., 101 for |101⟩"
                )
                iterations_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="Number of Iterations"
                )
            
            run_btn = gr.Button("🔍 Run Grover's Algorithm", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    initial_output = gr.HTML(label="Initial State")
                with gr.Column():
                    final_output = gr.HTML(label="Final State")
            
            analysis_output = gr.HTML(label="Analysis")
            
            run_btn.click(
                fn=run_grover_demo,
                inputs=[n_qubits_slider, marked_input, iterations_slider],
                outputs=[initial_output, final_output, analysis_output]
            )

        # QASM and Diagram Output
        with gr.Accordion("📝 Circuit Output (QASM & Diagram)", open=True):
            gr.Markdown("Generate OpenQASM 2.0 code and circuit diagram for the current Grover configuration.")
            
            with gr.Row():
                qasm_qubits = gr.Slider(minimum=2, maximum=5, value=3, step=1, label="Qubits")
                qasm_target = gr.Textbox(value="101", label="Target State (binary)")
                qasm_iters = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Iterations")
            
            generate_qasm_btn = gr.Button("📝 Generate QASM", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    grover_qasm_output = gr.Code(language="python", label="OpenQASM 2.0", lines=15)
                with gr.Column():
                    grover_diagram_output = gr.Textbox(label="Circuit Diagram", lines=8)
            
            generate_qasm_btn.click(
                fn=generate_grover_qasm,
                inputs=[qasm_qubits, qasm_target, qasm_iters],
                outputs=[grover_qasm_output, grover_diagram_output]
            )

        # Amplitude Evolution
        with gr.Accordion("📈 Amplitude Evolution Visualization", open=True):
            gr.Markdown("""
            Watch how the probability of finding the target state grows with each iteration.
            """)
            
            with gr.Row():
                evo_qubits = gr.Slider(minimum=2, maximum=4, value=3, step=1, label="Qubits")
                evo_marked = gr.Textbox(value="101", label="Target State")
                evo_iterations = gr.Slider(minimum=1, maximum=15, value=8, step=1, label="Max Iterations")
            
            evo_btn = gr.Button("📈 Show Evolution", variant="secondary")
            evo_output = gr.HTML()
            
            evo_btn.click(
                fn=visualize_amplitude_evolution,
                inputs=[evo_qubits, evo_marked, evo_iterations],
                outputs=[evo_output]
            )
        
        # Oracle Deep Dive
        with gr.Accordion("🔮 Understanding the Oracle", open=False):
            gr.Markdown("""
            The oracle is a key component that "knows" the solution. Let's understand how it works.
            """)
            
            with gr.Row():
                oracle_qubits = gr.Slider(minimum=2, maximum=4, value=3, step=1, label="Qubits")
                oracle_target = gr.Textbox(value="101", label="Target State")
                oracle_btn = gr.Button("🔮 Explain Oracle", variant="secondary")
            
            oracle_output = gr.HTML()
            
            oracle_btn.click(
                fn=explain_oracle,
                inputs=[oracle_qubits, oracle_target],
                outputs=[oracle_output]
            )
        
        # Diffusion Operator
        with gr.Accordion("🌊 The Diffusion Operator", open=False):
            diffusion_btn = gr.Button("🌊 Explain Diffusion Operator", variant="secondary")
            diffusion_output = gr.HTML()
            
            diffusion_btn.click(
                fn=explain_diffusion,
                inputs=[],
                outputs=[diffusion_output]
            )
        
        # Mathematical Details
        with gr.Accordion("📐 Mathematical Analysis", open=False):
            gr.Markdown(r"""
            ## Geometric Interpretation
            
            Grover's algorithm can be understood as rotations in a 2D plane spanned by:
            - $|\omega\rangle$ = superposition of marked states
            - $|s'\rangle$ = superposition of unmarked states
            
            ### Initial State
            
            $$|s\rangle = \sin\theta|\omega\rangle + \cos\theta|s'\rangle$$
            
            where $\sin\theta = \sqrt{M/N}$ (M = number of marked states, N = total states)
            
            ### Each Iteration
            
            Each Grover iteration rotates the state by angle $2\theta$ toward $|\omega\rangle$:
            
            $$|s_k\rangle = \sin((2k+1)\theta)|\omega\rangle + \cos((2k+1)\theta)|s'\rangle$$
            
            ### Optimal Number of Iterations
            
            Maximum probability when $(2k+1)\theta \approx \pi/2$:
            
            $$k_{opt} \approx \frac{\pi}{4}\sqrt{\frac{N}{M}}$$
            
            ## Success Probability
            
            After $k$ iterations:
            
            $$P(\text{success}) = \sin^2((2k+1)\theta)$$
            
            At optimal iterations:
            
            $$P(\text{success}) \geq 1 - \frac{M}{N}$$
            
            ## Why Only Quadratic Speedup?
            
            1. **No faster-than-light signaling:** Grover's is proven optimal for black-box search
            2. **Information theoretic limit:** Need to query the oracle at least $\Omega(\sqrt{N})$ times
            3. **Still impressive:** For $N = 10^{12}$, reduces from $10^{12}$ to $10^6$ queries!
            """)
        
        # Applications
        with gr.Accordion("🚀 Applications", open=False):
            gr.Markdown("""
            ## Practical Applications of Grover's Algorithm
            
            ### 1. Database Search
            - Searching unsorted databases
            - Finding records matching specific criteria
            
            ### 2. Cryptography
            - **Symmetric key search:** Reduces AES-256 security to AES-128 equivalent
            - **Hash collision finding:** Quadratic speedup for finding collisions
            
            ### 3. Optimization (via Amplitude Amplification)
            - **SAT solving:** Amplify solutions to Boolean satisfiability
            - **Constraint satisfaction:** Find valid configurations faster
            
            ### 4. Machine Learning
            - **Feature selection:** Search for optimal feature subsets
            - **Hyperparameter optimization:** Explore configuration spaces
            
            ### 5. Subroutine in Other Algorithms
            - **Quantum counting:** Count solutions using Grover + QFT
            - **Quantum walk algorithms:** Grover as a component
            - **Variational algorithms:** Amplitude amplification for solution finding
            
            ## Limitations
            
            | Challenge | Impact |
            |-----------|--------|
            | Requires known N | Must know search space size |
            | Single solution assumed | Multiple solutions reduce optimal iterations |
            | Unstructured only | No speedup for sorted/structured data |
            | Query complexity | Oracle must be efficiently implementable |
            """)
        
        # Quick Reference
        with gr.Accordion("📋 Quick Reference", open=False):
            gr.Markdown(r"""
            ## Key Formulas
            
            | Concept | Formula |
            |---------|---------|
            | Initial state | $\vert s\rangle = \frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}\vert x\rangle$ |
            | Oracle | $U_f\vert x\rangle = (-1)^{f(x)}\vert x\rangle$ |
            | Diffusion | $D = 2\vert s\rangle\langle s\vert - I$ |
            | Optimal iterations | $k \approx \frac{\pi}{4}\sqrt{N/M}$ |
            | Success probability | $P = \sin^2((2k+1)\theta)$ |
            | Complexity | $O(\sqrt{N/M})$ |
            
            ## Algorithm Steps
            
            1. **Prepare** $|0\rangle^{\otimes n}$
            2. **Apply** $H^{\otimes n}$ to create uniform superposition
            3. **Repeat** $k$ times:
               - Apply oracle $U_f$
               - Apply diffusion $D = H^{\otimes n}(2|0\rangle\langle 0| - I)H^{\otimes n}$
            4. **Measure** in computational basis
            
            ## Common Configurations
            
            | Qubits | N | Optimal k | P(success) |
            |--------|---|-----------|------------|
            | 2 | 4 | 1 | ~100% |
            | 3 | 8 | 2 | ~94.5% |
            | 4 | 16 | 3 | ~96.1% |
            | 5 | 32 | 4 | ~99.6% |
            | 10 | 1024 | 25 | >99% |
            """)
