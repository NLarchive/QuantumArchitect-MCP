"""
Quantum Measurement Study Tab for QuantumArchitect-MCP

This module provides an interactive educational tab explaining quantum measurement,
Born rule, wavefunction collapse, and different measurement bases.

Professional-level content with beginner-friendly explanations.
"""

import gradio as gr
import numpy as np
import random
from typing import Tuple, Dict, List


def get_measurement_probability(statevector: np.ndarray, basis_state: int) -> float:
    """Calculate measurement probability for a specific basis state."""
    return np.abs(statevector[basis_state])**2


def perform_measurement(statevector: np.ndarray, num_shots: int = 1000) -> Dict[str, int]:
    """Simulate measurement of a quantum state."""
    probabilities = np.abs(statevector)**2
    num_qubits = int(np.log2(len(statevector)))
    
    results = {}
    for _ in range(num_shots):
        outcome = np.random.choice(len(statevector), p=probabilities)
        bitstring = format(outcome, f'0{num_qubits}b')
        results[bitstring] = results.get(bitstring, 0) + 1
    
    return results


def single_measurement(statevector: np.ndarray) -> Tuple[str, np.ndarray]:
    """Perform a single measurement and return outcome and collapsed state."""
    probabilities = np.abs(statevector)**2
    num_qubits = int(np.log2(len(statevector)))
    
    outcome = np.random.choice(len(statevector), p=probabilities)
    bitstring = format(outcome, f'0{num_qubits}b')
    
    # Collapsed state is pure basis state
    collapsed = np.zeros_like(statevector)
    collapsed[outcome] = 1.0
    
    return bitstring, collapsed


def apply_hadamard(statevector: np.ndarray) -> np.ndarray:
    """Apply Hadamard gate to transform between Z and X basis."""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return H @ statevector


def apply_s_dagger_h(statevector: np.ndarray) -> np.ndarray:
    """Apply S†H to transform from Y basis to Z basis."""
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    S_dag = np.array([[1, 0], [0, -1j]])
    return H @ S_dag @ statevector


def format_complex(z: complex, precision: int = 4) -> str:
    """Format a complex number for display."""
    real, imag = z.real, z.imag
    if abs(imag) < 1e-10:
        return f"{real:.{precision}f}"
    elif abs(real) < 1e-10:
        return f"{imag:.{precision}f}i"
    else:
        sign = "+" if imag >= 0 else "-"
        return f"{real:.{precision}f} {sign} {abs(imag):.{precision}f}i"


def format_statevector_html(statevector: np.ndarray) -> str:
    """Format statevector as HTML for display."""
    num_qubits = int(np.log2(len(statevector)))
    terms = []
    
    for i, amp in enumerate(statevector):
        if abs(amp) > 1e-10:
            bitstring = format(i, f'0{num_qubits}b')
            amp_str = format_complex(amp, 3)
            terms.append(f"({amp_str})|{bitstring}⟩")
    
    if not terms:
        return "|0⟩"
    
    return " + ".join(terms)


def create_probability_bars_html(statevector: np.ndarray, basis_name: str = "Z") -> str:
    """Create HTML visualization of measurement probabilities."""
    probabilities = np.abs(statevector)**2
    num_qubits = int(np.log2(len(statevector)))
    
    html = f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; margin: 10px 0;">
        <h4 style="color: #4fc3f7; margin-bottom: 12px;">📊 Measurement Probabilities ({basis_name}-basis)</h4>
    """
    
    colors = ['#4fc3f7', '#7c4dff', '#ff4081', '#3fb950', '#d29922', '#58a6ff', '#bc8cff', '#ff7b72']
    
    for i, prob in enumerate(probabilities):
        if prob > 1e-10:
            bitstring = format(i, f'0{num_qubits}b')
            width = max(prob * 100, 2)
            color = colors[i % len(colors)]
            
            html += f"""
            <div style="margin: 6px 0;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: #c9d1d9; font-family: monospace; width: 50px;">|{bitstring}⟩</span>
                    <div style="flex: 1; background: #21262d; border-radius: 4px; height: 24px; position: relative;">
                        <div style="width: {width}%; background: linear-gradient(90deg, {color}, {color}88); 
                                    height: 100%; border-radius: 4px; transition: width 0.3s;"></div>
                    </div>
                    <span style="color: #8b949e; width: 70px; text-align: right;">{prob*100:.1f}%</span>
                </div>
            </div>
            """
    
    html += "</div>"
    return html


def simulate_measurement_demo(state_choice: str, num_shots: int) -> Tuple[str, str, str]:
    """Run measurement simulation for demo."""
    # Create the statevector based on choice
    states = {
        "|0⟩": np.array([1, 0], dtype=complex),
        "|1⟩": np.array([0, 1], dtype=complex),
        "|+⟩ = (|0⟩+|1⟩)/√2": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "|-⟩ = (|0⟩-|1⟩)/√2": np.array([1, -1], dtype=complex) / np.sqrt(2),
        "|i⟩ = (|0⟩+i|1⟩)/√2": np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "Custom: 0.6|0⟩ + 0.8|1⟩": np.array([0.6, 0.8], dtype=complex),
        "Custom: √(0.25)|0⟩ + √(0.75)|1⟩": np.array([np.sqrt(0.25), np.sqrt(0.75)], dtype=complex),
    }
    
    statevector = states.get(state_choice, np.array([1, 0], dtype=complex))
    
    # Calculate probabilities
    prob_0 = get_measurement_probability(statevector, 0)
    prob_1 = get_measurement_probability(statevector, 1)
    
    # Run measurements
    results = perform_measurement(statevector, num_shots)
    
    # Create visualization
    prob_html = create_probability_bars_html(statevector)
    
    # Create results summary
    results_html = f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px;">
        <h4 style="color: #3fb950; margin-bottom: 10px;">🎲 Measurement Results ({num_shots} shots)</h4>
        <div style="display: flex; gap: 20px; flex-wrap: wrap;">
    """
    
    for bitstring, count in sorted(results.items()):
        freq = count / num_shots * 100
        results_html += f"""
            <div style="background: #21262d; padding: 10px 15px; border-radius: 6px; text-align: center;">
                <div style="color: #4fc3f7; font-family: monospace; font-size: 1.2em;">|{bitstring}⟩</div>
                <div style="color: #c9d1d9; margin-top: 5px;">{count} ({freq:.1f}%)</div>
            </div>
        """
    
    results_html += """
        </div>
        <p style="color: #8b949e; margin-top: 10px; font-size: 0.9em;">
            ℹ️ Each measurement is probabilistic. Run again to see statistical variation!
        </p>
    </div>
    """
    
    # Theory comparison
    theory_html = f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px;">
        <h4 style="color: #7c4dff; margin-bottom: 10px;">📐 Theoretical vs Observed</h4>
        <table style="width: 100%; color: #c9d1d9; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #30363d;">
                <th style="text-align: left; padding: 8px;">Outcome</th>
                <th style="text-align: right; padding: 8px;">Theory (|α|²)</th>
                <th style="text-align: right; padding: 8px;">Observed</th>
                <th style="text-align: right; padding: 8px;">Deviation</th>
            </tr>
            <tr>
                <td style="padding: 8px; font-family: monospace;">|0⟩</td>
                <td style="text-align: right; padding: 8px;">{prob_0*100:.2f}%</td>
                <td style="text-align: right; padding: 8px;">{results.get('0', 0)/num_shots*100:.2f}%</td>
                <td style="text-align: right; padding: 8px;">{abs(prob_0 - results.get('0', 0)/num_shots)*100:.2f}%</td>
            </tr>
            <tr>
                <td style="padding: 8px; font-family: monospace;">|1⟩</td>
                <td style="text-align: right; padding: 8px;">{prob_1*100:.2f}%</td>
                <td style="text-align: right; padding: 8px;">{results.get('1', 0)/num_shots*100:.2f}%</td>
                <td style="text-align: right; padding: 8px;">{abs(prob_1 - results.get('1', 0)/num_shots)*100:.2f}%</td>
            </tr>
        </table>
        <p style="color: #8b949e; margin-top: 10px; font-size: 0.9em;">
            📈 Deviation decreases with more shots (Law of Large Numbers)
        </p>
    </div>
    """
    
    return prob_html, results_html, theory_html


def demonstrate_collapse(state_choice: str) -> Tuple[str, str, str]:
    """Demonstrate wavefunction collapse with single measurement."""
    states = {
        "|+⟩ = (|0⟩+|1⟩)/√2": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "|-⟩ = (|0⟩-|1⟩)/√2": np.array([1, -1], dtype=complex) / np.sqrt(2),
        "|i⟩ = (|0⟩+i|1⟩)/√2": np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "0.6|0⟩ + 0.8|1⟩": np.array([0.6, 0.8], dtype=complex),
    }
    
    statevector = states.get(state_choice, np.array([1, 1], dtype=complex) / np.sqrt(2))
    
    # Before state
    before_html = f"""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 10px;">📊 BEFORE Measurement</h4>
        <div style="background: #21262d; padding: 15px; border-radius: 6px; font-family: monospace; color: #c9d1d9; font-size: 1.1em;">
            |ψ⟩ = {format_statevector_html(statevector)}
        </div>
        <div style="margin-top: 15px;">
            <p style="color: #8b949e;">Probabilities:</p>
            <ul style="color: #c9d1d9;">
                <li>P(|0⟩) = |α|² = {np.abs(statevector[0])**2:.4f} = {np.abs(statevector[0])**2*100:.1f}%</li>
                <li>P(|1⟩) = |β|² = {np.abs(statevector[1])**2:.4f} = {np.abs(statevector[1])**2*100:.1f}%</li>
            </ul>
        </div>
        <p style="color: #3fb950; margin-top: 10px;">
            ✨ The qubit exists in a <strong>superposition</strong> of both states!
        </p>
    </div>
    """
    
    # Perform measurement
    outcome, collapsed = single_measurement(statevector)
    
    # After state
    after_html = f"""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px; border-left: 4px solid #ff4081;">
        <h4 style="color: #ff4081; margin-bottom: 10px;">⚡ AFTER Measurement</h4>
        <div style="background: #21262d; padding: 15px; border-radius: 6px; font-family: monospace; font-size: 1.1em;">
            <span style="color: #d29922;">Measured outcome:</span> <span style="color: #3fb950; font-size: 1.3em;">|{outcome}⟩</span>
        </div>
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-top: 10px; font-family: monospace; color: #c9d1d9; font-size: 1.1em;">
            |ψ'⟩ = |{outcome}⟩
        </div>
        <div style="margin-top: 15px;">
            <p style="color: #8b949e;">New probabilities:</p>
            <ul style="color: #c9d1d9;">
                <li>P(|0⟩) = {np.abs(collapsed[0])**2:.0f} = {np.abs(collapsed[0])**2*100:.0f}%</li>
                <li>P(|1⟩) = {np.abs(collapsed[1])**2:.0f} = {np.abs(collapsed[1])**2*100:.0f}%</li>
            </ul>
        </div>
        <p style="color: #ff4081; margin-top: 10px;">
            💥 The superposition has <strong>collapsed</strong> to a definite state!
        </p>
    </div>
    """
    
    # Explanation
    explain_html = f"""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px; border-left: 4px solid #7c4dff;">
        <h4 style="color: #7c4dff; margin-bottom: 10px;">🔬 What Just Happened?</h4>
        <ol style="color: #c9d1d9; line-height: 1.8;">
            <li><strong>Superposition existed:</strong> Before measurement, the qubit was in a superposition with amplitudes for both |0⟩ and |1⟩.</li>
            <li><strong>Measurement occurred:</strong> When we measured, nature "chose" outcome |{outcome}⟩ with probability {np.abs(statevector[int(outcome)])**2*100:.1f}%.</li>
            <li><strong>State collapsed:</strong> The wavefunction instantly became |{outcome}⟩ - no more superposition!</li>
            <li><strong>Irreversible:</strong> The original superposition information is lost. We cannot recover {format_statevector_html(statevector)}.</li>
        </ol>
        <div style="background: #21262d; padding: 10px; border-radius: 4px; margin-top: 10px;">
            <p style="color: #d29922; margin: 0;">
                ⚠️ <strong>Key insight:</strong> Measurement extracts classical information but destroys quantum information (superposition and phase).
            </p>
        </div>
    </div>
    """
    
    return before_html, after_html, explain_html


def demonstrate_basis_measurement(state_choice: str, basis_choice: str) -> str:
    """Show measurement in different bases."""
    # Define states
    states = {
        "|0⟩": np.array([1, 0], dtype=complex),
        "|1⟩": np.array([0, 1], dtype=complex),
        "|+⟩": np.array([1, 1], dtype=complex) / np.sqrt(2),
        "|-⟩": np.array([1, -1], dtype=complex) / np.sqrt(2),
        "|i⟩": np.array([1, 1j], dtype=complex) / np.sqrt(2),
        "|-i⟩": np.array([1, -1j], dtype=complex) / np.sqrt(2),
    }
    
    statevector = states.get(state_choice, np.array([1, 0], dtype=complex))
    
    # Transform to measurement basis
    if basis_choice == "Z-basis (computational)":
        transformed = statevector.copy()
        basis_states = ["|0⟩", "|1⟩"]
        basis_desc = "Standard computational basis - measures if qubit is 0 or 1"
    elif basis_choice == "X-basis (Hadamard)":
        transformed = apply_hadamard(statevector)
        basis_states = ["|+⟩", "|-⟩"]
        basis_desc = "Hadamard basis - measures if qubit is |+⟩ or |-⟩"
    else:  # Y-basis
        transformed = apply_s_dagger_h(statevector)
        basis_states = ["|i⟩", "|-i⟩"]
        basis_desc = "Circular basis - measures if qubit is |i⟩ or |-i⟩"
    
    prob_0 = np.abs(transformed[0])**2
    prob_1 = np.abs(transformed[1])**2
    
    html = f"""
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">📐 Measuring {state_choice} in {basis_choice}</h4>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <p style="color: #8b949e; margin: 0;">{basis_desc}</p>
        </div>
        
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 15px;">
            <div style="flex: 1; min-width: 200px; background: #21262d; padding: 15px; border-radius: 6px; text-align: center;">
                <div style="color: #4fc3f7; font-size: 0.9em;">Probability of {basis_states[0]}</div>
                <div style="color: #3fb950; font-size: 2em; margin: 10px 0;">{prob_0*100:.1f}%</div>
                <div style="background: #30363d; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="width: {prob_0*100}%; background: #3fb950; height: 100%;"></div>
                </div>
            </div>
            <div style="flex: 1; min-width: 200px; background: #21262d; padding: 15px; border-radius: 6px; text-align: center;">
                <div style="color: #4fc3f7; font-size: 0.9em;">Probability of {basis_states[1]}</div>
                <div style="color: #ff4081; font-size: 2em; margin: 10px 0;">{prob_1*100:.1f}%</div>
                <div style="background: #30363d; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="width: {prob_1*100}%; background: #ff4081; height: 100%;"></div>
                </div>
            </div>
        </div>
    """
    
    # Special cases explanation
    if prob_0 > 0.99 or prob_1 > 0.99:
        html += f"""
        <div style="background: #21262d; padding: 10px 15px; border-radius: 6px; border-left: 4px solid #3fb950;">
            <p style="color: #3fb950; margin: 0;">
                ✅ <strong>Deterministic outcome!</strong> This state is an eigenstate of the measurement operator.
            </p>
        </div>
        """
    elif abs(prob_0 - 0.5) < 0.01:
        html += f"""
        <div style="background: #21262d; padding: 10px 15px; border-radius: 6px; border-left: 4px solid #d29922;">
            <p style="color: #d29922; margin: 0;">
                ⚖️ <strong>Maximum uncertainty!</strong> Equal superposition in this basis - perfectly random outcome.
            </p>
        </div>
        """
    
    html += "</div>"
    return html


def add_measurement_study_tab():
    """Add the Quantum Measurement study tab to the Gradio interface."""
    
    with gr.Tab("📏 Measurement"):
        gr.Markdown("""
        <div style="text-align: center; padding: 15px 0;">
            <h2 style="color: #4fc3f7; margin-bottom: 8px;">Quantum Measurement</h2>
            <p style="color: #8b949e;">The Born Rule, Wavefunction Collapse, and Measurement Bases</p>
        </div>
        """)
        
        # Introduction Section
        with gr.Accordion("📚 Introduction to Quantum Measurement", open=True):
            gr.Markdown(r"""
            ## The Measurement Problem
            
            Quantum measurement is fundamentally different from classical measurement:
            
            | Classical Measurement | Quantum Measurement |
            |----------------------|---------------------|
            | Reveals pre-existing value | Creates the measured value |
            | Non-invasive (ideally) | Fundamentally disturbs the system |
            | Deterministic | Probabilistic |
            | Repeatable | Irreversible (collapses state) |
            
            ## The Born Rule
            
            When we measure a quantum state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$:
            
            $$P(\text{outcome } |0\rangle) = |\langle 0|\psi\rangle|^2 = |\alpha|^2$$
            $$P(\text{outcome } |1\rangle) = |\langle 1|\psi\rangle|^2 = |\beta|^2$$
            
            **Key Properties:**
            - Probabilities sum to 1: $|\alpha|^2 + |\beta|^2 = 1$ (normalization)
            - Only the **magnitude** of amplitudes matter for probability
            - **Phase** affects interference, not individual measurement probabilities
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### Example: |+⟩ State
                    
                    $|+\\rangle = \\frac{1}{\\sqrt{2}}(|0\\rangle + |1\\rangle)$
                    
                    - $\\alpha = \\frac{1}{\\sqrt{2}}$, so $|\\alpha|^2 = 0.5$
                    - $\\beta = \\frac{1}{\\sqrt{2}}$, so $|\\beta|^2 = 0.5$
                    - **50% chance of |0⟩, 50% chance of |1⟩**
                    """)
                
                with gr.Column():
                    gr.Markdown("""
                    ### Example: Biased State
                    
                    $|\\psi\\rangle = \\sqrt{0.25}|0\\rangle + \\sqrt{0.75}|1\\rangle$
                    
                    - $\\alpha = 0.5$, so $|\\alpha|^2 = 0.25$
                    - $\\beta \\approx 0.866$, so $|\\beta|^2 = 0.75$
                    - **25% chance of |0⟩, 75% chance of |1⟩**
                    """)
        
        # Interactive Measurement Demo
        with gr.Accordion("🎲 Interactive Measurement Simulator", open=True):
            gr.Markdown("""
            ## Try It Yourself!
            
            Select a quantum state and run measurements to see the Born rule in action.
            """)
            
            with gr.Row():
                state_dropdown = gr.Dropdown(
                    choices=[
                        "|0⟩",
                        "|1⟩", 
                        "|+⟩ = (|0⟩+|1⟩)/√2",
                        "|-⟩ = (|0⟩-|1⟩)/√2",
                        "|i⟩ = (|0⟩+i|1⟩)/√2",
                        "Custom: 0.6|0⟩ + 0.8|1⟩",
                        "Custom: √(0.25)|0⟩ + √(0.75)|1⟩"
                    ],
                    value="|+⟩ = (|0⟩+|1⟩)/√2",
                    label="Select Quantum State"
                )
                shots_slider = gr.Slider(
                    minimum=10,
                    maximum=10000,
                    value=1000,
                    step=10,
                    label="Number of Shots (Measurements)"
                )
                measure_btn = gr.Button("🎲 Run Measurements", variant="primary")
            
            with gr.Row():
                prob_output = gr.HTML(label="Theoretical Probabilities")
            
            with gr.Row():
                results_output = gr.HTML(label="Measurement Results")
            
            with gr.Row():
                theory_output = gr.HTML(label="Theory vs Observation")
            
            measure_btn.click(
                fn=simulate_measurement_demo,
                inputs=[state_dropdown, shots_slider],
                outputs=[prob_output, results_output, theory_output]
            )
        
        # Wavefunction Collapse Demo
        with gr.Accordion("💥 Wavefunction Collapse Demonstration", open=True):
            gr.Markdown("""
            ## State Collapse
            
            Before measurement, a qubit can be in superposition. After measurement, it **collapses** to a definite state.
            
            Click to see a single measurement and observe the collapse!
            """)
            
            with gr.Row():
                collapse_state = gr.Dropdown(
                    choices=[
                        "|+⟩ = (|0⟩+|1⟩)/√2",
                        "|-⟩ = (|0⟩-|1⟩)/√2",
                        "|i⟩ = (|0⟩+i|1⟩)/√2",
                        "0.6|0⟩ + 0.8|1⟩"
                    ],
                    value="|+⟩ = (|0⟩+|1⟩)/√2",
                    label="State to Measure"
                )
                collapse_btn = gr.Button("⚡ Perform Single Measurement", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    before_html = gr.HTML(label="Before Measurement")
                with gr.Column():
                    after_html = gr.HTML(label="After Measurement")
            
            with gr.Row():
                explain_html = gr.HTML(label="Explanation")
            
            collapse_btn.click(
                fn=demonstrate_collapse,
                inputs=[collapse_state],
                outputs=[before_html, after_html, explain_html]
            )
        
        # Measurement Bases
        with gr.Accordion("📐 Measurement Bases", open=True):
            gr.Markdown(r"""
            ## Different Ways to Measure
            
            Quantum states can be measured in different **bases**. The choice of basis determines what question we ask the qubit.
            
            | Basis | Eigenstates | What We Measure |
            |-------|-------------|-----------------|
            | **Z (Computational)** | $\vert 0\rangle$, $\vert 1\rangle$ | "Is the qubit 0 or 1?" |
            | **X (Hadamard)** | $\vert +\rangle$, $\vert -\rangle$ | "Is the qubit |+⟩ or |-⟩?" |
            | **Y (Circular)** | $\vert i\rangle$, $\vert -i\rangle$ | "Is the qubit |i⟩ or |-i⟩?" |
            
            ### Important Principle: Complementarity
            
            If a state is definite in one basis, it's uncertain in others:
            - $|0\rangle$ measured in Z-basis → **always** gives 0
            - $|0\rangle$ measured in X-basis → **50/50** between |+⟩ and |-⟩
            
            This is the quantum analog of the Heisenberg uncertainty principle!
            """)
            
            with gr.Row():
                basis_state = gr.Dropdown(
                    choices=["|0⟩", "|1⟩", "|+⟩", "|-⟩", "|i⟩", "|-i⟩"],
                    value="|0⟩",
                    label="State to Measure"
                )
                basis_choice = gr.Dropdown(
                    choices=[
                        "Z-basis (computational)",
                        "X-basis (Hadamard)", 
                        "Y-basis (circular)"
                    ],
                    value="Z-basis (computational)",
                    label="Measurement Basis"
                )
                basis_btn = gr.Button("📐 Analyze Measurement", variant="primary")
            
            basis_output = gr.HTML(label="Measurement Analysis")
            
            basis_btn.click(
                fn=demonstrate_basis_measurement,
                inputs=[basis_state, basis_choice],
                outputs=[basis_output]
            )
        
        # Advanced Topics
        with gr.Accordion("🔬 Advanced: POVM and Generalized Measurements", open=False):
            gr.Markdown(r"""
            ## Beyond Projective Measurement
            
            Standard (projective) measurements are described by:
            - **Projectors** $P_i$ that satisfy $\sum_i P_i = I$
            - Each $P_i$ corresponds to a possible outcome
            
            ### Positive Operator-Valued Measures (POVM)
            
            For more general measurements, we use POVMs:
            - **POVM elements** $E_i$ satisfy $\sum_i E_i = I$ and $E_i \geq 0$
            - Probability of outcome $i$: $P(i) = \langle\psi|E_i|\psi\rangle$
            
            ### Why POVMs?
            
            1. **More outcomes than dimensions**: POVMs can have more outcomes than the Hilbert space dimension
            2. **Optimal state discrimination**: Sometimes better for distinguishing non-orthogonal states
            3. **Weak measurements**: Allow partial information extraction with less disturbance
            
            ### Example: Unambiguous State Discrimination
            
            To distinguish non-orthogonal states $|0\rangle$ and $|+\rangle$ without error:
            - Sometimes get "I don't know" (inconclusive)
            - But never misidentify one for the other
            
            This is impossible with projective measurement but achievable with POVM!
            """)
        
        # Mathematical Formalism
        with gr.Accordion("📖 Mathematical Formalism", open=False):
            gr.Markdown(r"""
            ## Measurement Postulate (von Neumann)
            
            For an observable $A = \sum_a a |a\rangle\langle a|$ with eigenstates $|a\rangle$:
            
            1. **Possible outcomes**: Eigenvalues $a$ of $A$
            2. **Probability**: $P(a) = |\langle a|\psi\rangle|^2$
            3. **Post-measurement state**: $\frac{|a\rangle\langle a|\psi\rangle}{\sqrt{P(a)}}$
            
            ## Expectation Value
            
            The average value over many measurements:
            $$\langle A \rangle = \langle\psi|A|\psi\rangle = \sum_a a \cdot P(a)$$
            
            ## Measurement Operators
            
            For computational basis measurement on qubit $k$ in an $n$-qubit system:
            
            $$M_0 = I^{\otimes (k-1)} \otimes |0\rangle\langle 0| \otimes I^{\otimes (n-k)}$$
            $$M_1 = I^{\otimes (k-1)} \otimes |1\rangle\langle 1| \otimes I^{\otimes (n-k)}$$
            
            ## Partial Measurement
            
            When measuring only some qubits of an entangled state:
            - The unmeasured qubits are **projected** based on the outcome
            - This is how entanglement is "consumed" in quantum protocols
            
            ### Example: Bell State Measurement
            
            Measuring first qubit of $|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$:
            
            - If outcome is 0 → second qubit is $|0\rangle$
            - If outcome is 1 → second qubit is $|1\rangle$
            
            The qubits are **perfectly correlated** despite the randomness!
            """)
        
        # Quick Reference
        with gr.Accordion("📋 Quick Reference", open=False):
            gr.Markdown(r"""
            ## Key Formulas
            
            | Concept | Formula |
            |---------|---------|
            | Born Rule | $P(a) = \vert\langle a\vert\psi\rangle\vert^2$ |
            | Normalization | $\sum_a P(a) = 1$ |
            | Expectation | $\langle A \rangle = \langle\psi\vert A\vert\psi\rangle$ |
            | Variance | $(\Delta A)^2 = \langle A^2 \rangle - \langle A \rangle^2$ |
            | Z-basis states | $\vert 0\rangle = \begin{pmatrix}1\\0\end{pmatrix}$, $\vert 1\rangle = \begin{pmatrix}0\\1\end{pmatrix}$ |
            | X-basis states | $\vert +\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}$, $\vert -\rangle = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}$ |
            
            ## Common Measurement Scenarios
            
            | State | Z-basis | X-basis |
            |-------|---------|---------|
            | $\vert 0\rangle$ | 100% $\vert 0\rangle$ | 50/50 $\vert \pm\rangle$ |
            | $\vert 1\rangle$ | 100% $\vert 1\rangle$ | 50/50 $\vert \pm\rangle$ |
            | $\vert +\rangle$ | 50/50 $\vert 0/1\rangle$ | 100% $\vert +\rangle$ |
            | $\vert -\rangle$ | 50/50 $\vert 0/1\rangle$ | 100% $\vert -\rangle$ |
            
            ## Key Takeaways
            
            1. **Measurement is probabilistic** - governed by Born rule
            2. **Measurement collapses state** - superposition is destroyed  
            3. **Choice of basis matters** - complementary observables
            4. **Information is extracted at a cost** - quantum info lost
            """)
