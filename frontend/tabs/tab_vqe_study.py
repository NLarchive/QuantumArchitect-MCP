"""
VQE (Variational Quantum Eigensolver) Study Tab for QuantumArchitect-MCP

This module provides an interactive educational tab explaining VQE,
a cornerstone NISQ algorithm for quantum chemistry and optimization.

Professional-level content with beginner-friendly explanations.
"""

import gradio as gr
import numpy as np
from typing import Tuple, Dict, List


def create_simple_ansatz(num_qubits: int, num_layers: int, params: List[float]) -> str:
    """Create a simple hardware-efficient ansatz QASM."""
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{num_qubits}];"
    ]
    
    param_idx = 0
    for layer in range(num_layers):
        # Rotation layer
        for q in range(num_qubits):
            if param_idx < len(params):
                theta = params[param_idx]
                param_idx += 1
            else:
                theta = 0.0
            lines.append(f"ry({theta:.4f}) q[{q}];")
        
        # Entanglement layer
        for q in range(num_qubits - 1):
            lines.append(f"cx q[{q}],q[{q+1}];")
    
    return "\n".join(lines)


def simulate_vqe_optimization(num_iterations: int = 20) -> Tuple[str, str]:
    """Simulate VQE optimization trajectory."""
    # Simulate a simple optimization curve
    np.random.seed(42)
    
    # Target energy (e.g., ground state of H2)
    target_energy = -1.137
    
    # Initial random energy
    initial_energy = np.random.uniform(-0.5, 0.5)
    
    # Simulated optimization trajectory
    energies = [initial_energy]
    params_history = []
    
    current_energy = initial_energy
    for i in range(num_iterations):
        # Simulate gradient descent with noise
        step_size = 0.1 * (0.9 ** i)  # Decreasing step size
        noise = np.random.normal(0, 0.02)
        current_energy = (current_energy + (target_energy - current_energy) * step_size 
                         + noise * (1 - i/num_iterations))
        energies.append(current_energy)
    
    # Create ASCII plot
    min_e, max_e = min(energies), max(energies)
    height = 12
    width = 50
    
    plot_html = """
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; font-family: monospace;">
        <h4 style="color: #4fc3f7; margin-bottom: 10px;">📉 VQE Optimization Progress</h4>
        <pre style="color: #c9d1d9; line-height: 1.2;">
"""
    
    # Scale energies to plot height
    for row in range(height):
        line = ""
        threshold = max_e - (row / (height - 1)) * (max_e - min_e)
        
        for col in range(min(len(energies), width)):
            if col == 0:
                line += "│"
            elif abs(energies[col] - threshold) < (max_e - min_e) / (height * 2):
                line += "●"
            elif energies[col] < threshold:
                line += " "
            else:
                line += " "
        
        # Y-axis label
        if row == 0:
            line = f"{max_e:6.2f} " + line
        elif row == height - 1:
            line = f"{min_e:6.2f} " + line
        else:
            line = "       " + line
        
        plot_html += line + "\n"
    
    plot_html += "       └" + "─" * 20 + " iterations\n"
    plot_html += f"""
        </pre>
        <div style="margin-top: 10px; color: #8b949e;">
            Initial: {initial_energy:.4f} Ha | Final: {energies[-1]:.4f} Ha | Target: {target_energy:.4f} Ha
        </div>
    </div>
    """
    
    # Analysis
    analysis_html = f"""
    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px;">
        <h4 style="color: #3fb950; margin-bottom: 10px;">📊 Optimization Analysis</h4>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
            <div style="background: #21262d; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e;">Energy Improvement</div>
                <div style="color: #3fb950; font-size: 1.3em;">{abs(energies[-1] - initial_energy):.4f} Ha</div>
            </div>
            <div style="background: #21262d; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e;">Chemical Accuracy</div>
                <div style="color: {'#3fb950' if abs(energies[-1] - target_energy) < 0.0016 else '#ff4081'}; font-size: 1.3em;">
                    {'✓ Achieved' if abs(energies[-1] - target_energy) < 0.0016 else '✗ Not yet'}
                </div>
            </div>
            <div style="background: #21262d; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e;">Error from Target</div>
                <div style="color: #4fc3f7; font-size: 1.3em;">{abs(energies[-1] - target_energy):.4f} Ha</div>
            </div>
            <div style="background: #21262d; padding: 10px; border-radius: 6px; text-align: center;">
                <div style="color: #8b949e;">Iterations Used</div>
                <div style="color: #d29922; font-size: 1.3em;">{num_iterations}</div>
            </div>
        </div>
        <p style="color: #8b949e; margin-top: 10px; font-size: 0.9em;">
            Chemical accuracy threshold: ±1.6 mHa (±0.0016 Ha) ≈ 1 kcal/mol
        </p>
    </div>
    """
    
    return plot_html, analysis_html


def explain_ansatz_types() -> str:
    """Generate HTML explaining different ansatz types."""
    html = """
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">🔧 Ansatz Types</h4>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #3fb950; margin-bottom: 10px;">1. Hardware-Efficient Ansatz (HEA)</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <pre style="color: #c9d1d9; margin: 0; font-size: 0.9em;">
Layer 1:    Ry(θ₁)─●───Ry(θ₃)─●───
                   │         │
            Ry(θ₂)─X───Ry(θ₄)─X───
                </pre>
                <p style="color: #8b949e; margin-top: 10px; margin-bottom: 0;">
                    <strong>Pros:</strong> Native to hardware, low depth<br>
                    <strong>Cons:</strong> May have barren plateaus, not problem-specific
                </p>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #7c4dff; margin-bottom: 10px;">2. Unitary Coupled Cluster (UCC)</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <p style="color: #c9d1d9; margin: 0;">
                    Chemically-inspired ansatz: e<sup>(T - T†)</sup> |HF⟩<br><br>
                    T = cluster operator (singles, doubles, ...)
                </p>
                <p style="color: #8b949e; margin-top: 10px; margin-bottom: 0;">
                    <strong>Pros:</strong> Problem-motivated, systematic improvement<br>
                    <strong>Cons:</strong> Deep circuits, expensive Trotterization
                </p>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h5 style="color: #ff4081; margin-bottom: 10px;">3. ADAPT-VQE (Adaptive)</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <p style="color: #c9d1d9; margin: 0;">
                    Grows ansatz iteratively by selecting operators with largest gradient.<br>
                    Balances accuracy vs circuit depth.
                </p>
                <p style="color: #8b949e; margin-top: 10px; margin-bottom: 0;">
                    <strong>Pros:</strong> Compact circuits, avoids barren plateaus<br>
                    <strong>Cons:</strong> Many measurement rounds, gradient overhead
                </p>
            </div>
        </div>
        
        <div>
            <h5 style="color: #d29922; margin-bottom: 10px;">4. Symmetry-Preserving Ansatz</h5>
            <div style="background: #21262d; padding: 15px; border-radius: 6px;">
                <p style="color: #c9d1d9; margin: 0;">
                    Encodes physical symmetries (particle number, spin) into circuit structure.
                </p>
                <p style="color: #8b949e; margin-top: 10px; margin-bottom: 0;">
                    <strong>Pros:</strong> Smaller search space, physical solutions<br>
                    <strong>Cons:</strong> Requires symmetry knowledge, complex gates
                </p>
            </div>
        </div>
    </div>
    """
    return html


def explain_measurement_strategies() -> str:
    """Explain Hamiltonian measurement strategies."""
    html = """
    <div style="background: #1a1a2e; padding: 20px; border-radius: 8px;">
        <h4 style="color: #4fc3f7; margin-bottom: 15px;">📐 Measuring the Hamiltonian</h4>
        
        <div style="background: #21262d; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
            <p style="color: #c9d1d9; margin: 0;">
                The molecular Hamiltonian is a sum of Pauli terms:<br><br>
                H = Σᵢ cᵢ Pᵢ &nbsp;&nbsp;(e.g., H = 0.5 ZZ + 0.3 XI + 0.2 IZ + ...)
            </p>
        </div>
        
        <h5 style="color: #3fb950; margin-bottom: 10px;">Measurement Strategies</h5>
        
        <table style="width: 100%; color: #c9d1d9; border-collapse: collapse; margin-bottom: 15px;">
            <tr style="border-bottom: 1px solid #30363d; background: #21262d;">
                <th style="padding: 10px; text-align: left;">Strategy</th>
                <th style="padding: 10px; text-align: left;">Measurements</th>
                <th style="padding: 10px; text-align: left;">Variance</th>
            </tr>
            <tr style="border-bottom: 1px solid #30363d;">
                <td style="padding: 10px;">Term-by-term</td>
                <td style="padding: 10px;">O(N⁴) for molecules</td>
                <td style="padding: 10px;">σ²/M per term</td>
            </tr>
            <tr style="border-bottom: 1px solid #30363d;">
                <td style="padding: 10px;">Qubit-wise commuting groups</td>
                <td style="padding: 10px;">O(N³) groups</td>
                <td style="padding: 10px;">Reduced</td>
            </tr>
            <tr style="border-bottom: 1px solid #30363d;">
                <td style="padding: 10px;">General commuting groups</td>
                <td style="padding: 10px;">O(N²) groups</td>
                <td style="padding: 10px;">Further reduced</td>
            </tr>
            <tr>
                <td style="padding: 10px;">Classical shadows</td>
                <td style="padding: 10px;">O(log N)</td>
                <td style="padding: 10px;">Optimal (with overhead)</td>
            </tr>
        </table>
        
        <div style="background: #21262d; padding: 10px 15px; border-radius: 6px; border-left: 4px solid #7c4dff;">
            <p style="color: #7c4dff; margin: 0;">
                💡 <strong>Key insight:</strong> Measurement cost often dominates VQE runtime!
                For large molecules, millions of shots may be needed.
            </p>
        </div>
    </div>
    """
    return html


def add_vqe_study_tab():
    """Add the VQE study tab to the Gradio interface."""
    
    with gr.Tab("⚛️ VQE"):
        gr.Markdown("""
        <div style="text-align: center; padding: 15px 0;">
            <h2 style="color: #4fc3f7; margin-bottom: 8px;">Variational Quantum Eigensolver (VQE)</h2>
            <p style="color: #8b949e;">The Cornerstone NISQ Algorithm for Quantum Chemistry</p>
        </div>
        """)
        
        # Introduction
        with gr.Accordion("📚 Introduction to VQE", open=True):
            gr.Markdown(r"""
            ## What is VQE?
            
            The **Variational Quantum Eigensolver** is a hybrid quantum-classical algorithm that finds
            the ground state energy of a Hamiltonian by leveraging the **variational principle**:
            
            $$E_0 \leq \langle\psi(\vec{\theta})| H |\psi(\vec{\theta})\rangle$$
            
            For any trial state $|\psi(\vec{\theta})\rangle$, the expectation value of $H$ is an **upper bound**
            on the true ground state energy $E_0$.
            
            ## Algorithm Overview
            
            ```
            ┌─────────────────────────────────────────────────────────────────┐
            │                    VQE OPTIMIZATION LOOP                         │
            │                                                                  │
            │  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
            │  │ Classical │    │   Quantum    │    │     Classical      │     │
            │  │ Optimizer │───▶│   Computer   │───▶│  Post-processing   │     │
            │  │  (θ→θ')   │    │  |ψ(θ)⟩      │    │  ⟨ψ(θ)|H|ψ(θ)⟩     │     │
            │  └─────▲─────┘    └──────────────┘    └─────────┬──────────┘     │
            │        │                                        │                │
            │        └────────────────────────────────────────┘                │
            │                      E(θ) feedback                               │
            └─────────────────────────────────────────────────────────────────┘
            ```
            
            ## Why VQE for NISQ?
            
            | Property | Benefit for NISQ |
            |----------|------------------|
            | **Shallow circuits** | Fits within coherence times |
            | **Error resilience** | Energy is variational (noise raises estimate) |
            | **Hybrid architecture** | Classical optimizer handles complexity |
            | **Flexible ansatz** | Can tailor to hardware constraints |
            
            ## Key Applications
            
            - **Quantum Chemistry:** Molecular ground states, reaction energies
            - **Materials Science:** Band structure, phase transitions  
            - **Optimization:** QUBO problems via Ising encoding
            - **Machine Learning:** Quantum kernel methods, QNNs
            """)
        
        # Interactive VQE Demo
        with gr.Accordion("🎮 VQE Optimization Demo", open=True):
            gr.Markdown("""
            Watch a simulated VQE optimization for finding the ground state of H₂ molecule.
            """)
            
            with gr.Row():
                iterations_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=5,
                    label="Optimization Iterations"
                )
                run_vqe_btn = gr.Button("▶️ Run VQE Simulation", variant="primary")
            
            with gr.Row():
                plot_output = gr.HTML(label="Optimization Progress")
            
            with gr.Row():
                analysis_output = gr.HTML(label="Analysis")
            
            run_vqe_btn.click(
                fn=simulate_vqe_optimization,
                inputs=[iterations_slider],
                outputs=[plot_output, analysis_output]
            )
        
        # Ansatz Section
        with gr.Accordion("🔧 Ansatz Design", open=False):
            gr.Markdown("""
            The **ansatz** (or variational form) defines the parameterized circuit that prepares trial states.
            Choosing the right ansatz is crucial for VQE success.
            """)
            
            ansatz_btn = gr.Button("📋 Show Ansatz Types", variant="secondary")
            ansatz_output = gr.HTML()
            
            ansatz_btn.click(fn=explain_ansatz_types, inputs=[], outputs=[ansatz_output])
        
        # Cost Function & Measurement
        with gr.Accordion("📊 Energy Estimation", open=False):
            gr.Markdown(r"""
            ## The Cost Function
            
            In VQE, we minimize the energy expectation value:
            
            $$E(\vec{\theta}) = \langle\psi(\vec{\theta})| H |\psi(\vec{\theta})\rangle$$
            
            This requires measuring the Hamiltonian H on the prepared state.
            
            ## Measurement Challenge
            
            Quantum computers can only measure in the computational (Z) basis.
            For other Pauli terms, we need basis rotations:
            
            | Pauli | Measurement Basis Change |
            |-------|--------------------------|
            | Z | None (native) |
            | X | Apply H before measurement |
            | Y | Apply S†H before measurement |
            """)
            
            measure_btn = gr.Button("📐 Measurement Strategies", variant="secondary")
            measure_output = gr.HTML()
            
            measure_btn.click(fn=explain_measurement_strategies, inputs=[], outputs=[measure_output])
        
        # Classical Optimizer
        with gr.Accordion("🔄 Classical Optimization", open=False):
            gr.Markdown(r"""
            ## Optimizer Choices
            
            The classical optimizer updates parameters $\vec{\theta}$ to minimize energy.
            
            | Optimizer | Type | Pros | Cons |
            |-----------|------|------|------|
            | **COBYLA** | Gradient-free | Noise-robust | Slow convergence |
            | **SPSA** | Stochastic gradient | Only 2 evaluations/step | High variance |
            | **Adam** | Gradient descent | Fast, adaptive | Needs gradients |
            | **Natural Gradient** | Geometric | Efficient in parameter space | Expensive |
            | **BFGS/L-BFGS** | Quasi-Newton | Fast convergence | Noise-sensitive |
            
            ## Gradient Estimation
            
            **Parameter-shift rule** for computing exact gradients:
            
            $$\frac{\partial E}{\partial \theta_i} = \frac{E(\theta_i + \frac{\pi}{2}) - E(\theta_i - \frac{\pi}{2})}{2}$$
            
            This requires 2 circuit evaluations per parameter!
            
            ## Noise Considerations
            
            - **Statistical noise:** Finite shot counts → variance in energy estimate
            - **Device noise:** Gate errors, decoherence → biased estimates
            - **Mitigation:** Error mitigation, regularization, symmetry verification
            """)
        
        # Barren Plateaus
        with gr.Accordion("⚠️ Barren Plateaus", open=False):
            gr.Markdown(r"""
            ## The Barren Plateau Problem
            
            A critical challenge in VQE is the **barren plateau** phenomenon:
            
            > As circuit depth or width increases, gradients vanish exponentially,
            > making optimization impossible.
            
            $$\text{Var}\left[\frac{\partial E}{\partial \theta_i}\right] \sim O(2^{-n})$$
            
            ## Causes
            
            1. **Hardware-efficient ansätze:** Random rotations lead to Haar-random states
            2. **Global cost functions:** Measuring all qubits creates trainability issues
            3. **Deep circuits:** Each layer compounds the expressibility problem
            4. **Noise:** Device noise can induce or worsen barren plateaus
            
            ## Mitigation Strategies
            
            | Strategy | How it helps |
            |----------|--------------|
            | **Layer-wise training** | Train layers sequentially |
            | **Local cost functions** | Measure fewer qubits at once |
            | **Problem-inspired ansätze** | Reduce to relevant subspace |
            | **ADAPT-VQE** | Grow ansatz adaptively |
            | **Warm-starting** | Initialize from classical solution |
            | **Overparameterization** | Larger parameter space, but use carefully |
            
            ## Detection
            
            Before running full VQE, check gradient variance:
            - If $\text{Var}[\nabla E] < 10^{-6}$: likely in barren plateau
            - Consider simplifying ansatz or using different strategy
            """)
        
        # Real-World Example
        with gr.Accordion("🧪 Example: H₂ Molecule", open=False):
            gr.Markdown(r"""
            ## Hydrogen Molecule (H₂)
            
            The simplest molecule, but foundational for understanding VQE.
            
            ### Problem Setup
            
            - **Hamiltonian:** After Jordan-Wigner transformation:
            
            $$H = g_0 I + g_1 Z_0 + g_2 Z_1 + g_3 Z_0 Z_1 + g_4 X_0 X_1 + g_5 Y_0 Y_1$$
            
            - **Qubits needed:** 2 (after symmetry reduction)
            - **Parameters:** Coefficients $g_i$ depend on bond distance
            
            ### Typical Ansatz (UCCSD-inspired)
            
            ```
            |0⟩ ─[X]─[Ry(θ)]─●─────
                             │
            |0⟩ ─────────────X─────
            ```
            
            ### Results at Equilibrium (0.735 Å)
            
            | Method | Energy (Ha) | Error |
            |--------|-------------|-------|
            | Exact | -1.1373 | 0 |
            | VQE (ideal) | -1.1373 | < 0.001 |
            | VQE (noisy) | -1.12 to -1.13 | 0.01-0.02 |
            | Hartree-Fock | -1.117 | 0.020 |
            
            ### Key Observations
            
            1. VQE achieves **chemical accuracy** (±1.6 mHa) on ideal simulators
            2. Noise raises energy estimate (variational property preserved)
            3. Simple 2-qubit circuit sufficient for this molecule
            """)
        
        # Advanced Topics
        with gr.Accordion("🔬 Advanced Topics", open=False):
            gr.Markdown(r"""
            ## Extensions of VQE
            
            ### Excited States
            
            - **VQD (Variational Quantum Deflation):** Penalize overlap with ground state
            - **SSVQE (Subspace-Search VQE):** Optimize multiple states simultaneously
            - **Equation-of-motion methods:** Excitations from ground state
            
            ### Time Evolution
            
            - **VQE + Time Evolution:** Combine with real-time dynamics
            - **Variational simulation:** Parameterized circuits for dynamics
            
            ### Error Mitigation
            
            | Technique | Overhead | Noise Types |
            |-----------|----------|-------------|
            | **Zero-noise extrapolation** | 3-5× | Gate errors |
            | **Probabilistic error cancellation** | Exponential | All |
            | **Symmetry verification** | 2× | Bit flips |
            | **Measurement error mitigation** | O(2^n) calibration | Readout |
            
            ### Quantum Resources
            
            - **Circuit depth:** Typically O(n) to O(n²) for chemistry
            - **Measurement shots:** 10⁶ - 10⁹ for large molecules
            - **Classical optimization:** Can dominate wall-clock time
            
            ## When to Use VQE
            
            ✅ **Good for:**
            - Small-to-medium molecules (< 50 qubits)
            - When circuit depth is limited
            - When approximate solutions are acceptable
            
            ❌ **Challenges:**
            - Very large systems (measurement cost)
            - Strongly correlated systems (ansatz limitations)
            - High precision requirements (noise accumulation)
            """)
        
        # Quick Reference
        with gr.Accordion("📋 Quick Reference", open=False):
            gr.Markdown(r"""
            ## VQE Checklist
            
            1. **Encode problem as Hamiltonian**
               - Fermion → qubit mapping (Jordan-Wigner, Bravyi-Kitaev)
               - Identify symmetries for qubit reduction
            
            2. **Choose ansatz**
               - Consider hardware constraints
               - Balance expressibility vs trainability
            
            3. **Select optimizer**
               - Gradient-free for noisy environments
               - Gradient-based for simulators
            
            4. **Plan measurements**
               - Group commuting terms
               - Estimate shot budget
            
            5. **Initialize parameters**
               - Warm-start from classical solution if available
               - Small random values otherwise
            
            6. **Implement error mitigation**
               - Symmetry verification
               - Zero-noise extrapolation
            
            ## Key Equations
            
            | Concept | Formula |
            |---------|---------|
            | Variational principle | $E_0 \leq \langle\psi(\theta)\vert H\vert\psi(\theta)\rangle$ |
            | Energy estimate | $\hat{E} = \sum_i c_i \langle P_i \rangle$ |
            | Parameter shift | $\partial_\theta E = \frac{E(\theta+\frac{\pi}{2}) - E(\theta-\frac{\pi}{2})}{2}$ |
            | Shot variance | $\text{Var}[\hat{E}] \sim \sum_i c_i^2 / N_{\text{shots}}$ |
            
            ## Common Pitfalls
            
            - ❌ Using too deep ansätze (barren plateaus)
            - ❌ Insufficient shots (high variance)
            - ❌ Not checking for local minima
            - ❌ Ignoring symmetries (wasting qubits)
            - ❌ Not validating on classical simulators first
            """)
