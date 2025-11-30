"""
Dirac Notation & Quantum States Study Tab

A comprehensive educational module covering:
- Dirac (bra-ket) notation fundamentals
- Quantum state representations
- Common quantum states with visualizations
- Gate operations as matrix transformations
- Interactive state vector calculations

Target audience: Both newcomers needing foundations and professionals needing refresher.
"""

import gradio as gr
import numpy as np


def create_state_vector_html(state_name: str, vector: list, description: str) -> str:
    """Create HTML representation of a quantum state with its vector."""
    vector_str = "<br>".join([f"  {v}" for v in vector])
    return f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #4fc3f7;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
    ">
        <div style="color: #4fc3f7; font-size: 1.3em; margin-bottom: 8px;">
            {state_name}
        </div>
        <div style="color: #e0e0e0; font-size: 1.1em; margin-bottom: 12px;">
            = <span style="color: #7c4dff;">[</span><br>{vector_str}<br><span style="color: #7c4dff;">]</span>
        </div>
        <div style="color: #9e9e9e; font-size: 0.9em; font-style: italic;">
            {description}
        </div>
    </div>
    """


def create_gate_matrix_html(gate_name: str, matrix: list[list], description: str) -> str:
    """Create HTML representation of a gate matrix."""
    rows = []
    for row in matrix:
        row_str = " ".join([f"<span style='min-width: 60px; display: inline-block; text-align: center;'>{v}</span>" for v in row])
        rows.append(f"<div style='margin: 4px 0;'>│ {row_str} │</div>")
    matrix_html = "\n".join(rows)
    
    return f"""
    <div style="
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        border: 1px solid #ff4081;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        font-family: 'Courier New', monospace;
    ">
        <div style="color: #ff4081; font-size: 1.2em; margin-bottom: 8px;">
            {gate_name}
        </div>
        <div style="color: #e0e0e0; font-size: 1em;">
            {matrix_html}
        </div>
        <div style="color: #9e9e9e; font-size: 0.85em; font-style: italic; margin-top: 8px;">
            {description}
        </div>
    </div>
    """


def calculate_gate_action(gate_name: str, input_state: str) -> tuple[str, str]:
    """Calculate the result of applying a gate to an input state."""
    # Define states
    states = {
        "|0⟩": np.array([1, 0], dtype=complex),
        "|1⟩": np.array([0, 1], dtype=complex),
        "|+⟩": np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
        "|-⟩": np.array([1/np.sqrt(2), -1/np.sqrt(2)], dtype=complex),
        "|i⟩": np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex),
        "|-i⟩": np.array([1/np.sqrt(2), -1j/np.sqrt(2)], dtype=complex),
    }
    
    # Define gates
    gates = {
        "I (Identity)": np.array([[1, 0], [0, 1]], dtype=complex),
        "X (NOT)": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        "H (Hadamard)": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
        "S (Phase)": np.array([[1, 0], [0, 1j]], dtype=complex),
        "T (π/8)": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
    }
    
    if input_state not in states or gate_name not in gates:
        return "Invalid input", ""
    
    input_vec = states[input_state]
    gate_mat = gates[gate_name]
    output_vec = gate_mat @ input_vec
    
    # Format input vector
    def format_complex(c):
        if np.abs(c.imag) < 1e-10:
            if np.abs(c.real - 1) < 1e-10:
                return "1"
            elif np.abs(c.real + 1) < 1e-10:
                return "-1"
            elif np.abs(c.real) < 1e-10:
                return "0"
            elif np.abs(c.real - 1/np.sqrt(2)) < 1e-10:
                return "1/√2"
            elif np.abs(c.real + 1/np.sqrt(2)) < 1e-10:
                return "-1/√2"
            else:
                return f"{c.real:.4f}"
        elif np.abs(c.real) < 1e-10:
            if np.abs(c.imag - 1) < 1e-10:
                return "i"
            elif np.abs(c.imag + 1) < 1e-10:
                return "-i"
            elif np.abs(c.imag - 1/np.sqrt(2)) < 1e-10:
                return "i/√2"
            elif np.abs(c.imag + 1/np.sqrt(2)) < 1e-10:
                return "-i/√2"
            else:
                return f"{c.imag:.4f}i"
        else:
            return f"{c.real:.4f} + {c.imag:.4f}i"
    
    # Try to identify output state
    output_state_name = "Unknown state"
    for name, vec in states.items():
        if np.allclose(output_vec, vec) or np.allclose(output_vec, -vec) or np.allclose(output_vec, 1j*vec) or np.allclose(output_vec, -1j*vec):
            # Check for phase factor
            if np.allclose(output_vec, vec):
                output_state_name = name
            elif np.allclose(output_vec, -vec):
                output_state_name = f"-{name}"
            elif np.allclose(output_vec, 1j*vec):
                output_state_name = f"i{name}"
            elif np.allclose(output_vec, -1j*vec):
                output_state_name = f"-i{name}"
            break
    
    # Create calculation display
    calculation = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #3fb950;
        border-radius: 12px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        color: #e0e0e0;
    ">
        <h3 style="color: #3fb950; margin-top: 0;">Calculation: {gate_name} × {input_state}</h3>
        
        <div style="display: flex; align-items: center; gap: 20px; flex-wrap: wrap; justify-content: center;">
            <div style="text-align: center;">
                <div style="color: #ff4081; margin-bottom: 4px;">Gate Matrix</div>
                <div style="background: #0d1117; padding: 10px; border-radius: 8px;">
                    [{format_complex(gate_mat[0,0])}, {format_complex(gate_mat[0,1])}]<br>
                    [{format_complex(gate_mat[1,0])}, {format_complex(gate_mat[1,1])}]
                </div>
            </div>
            
            <div style="font-size: 1.5em; color: #7c4dff;">×</div>
            
            <div style="text-align: center;">
                <div style="color: #4fc3f7; margin-bottom: 4px;">Input {input_state}</div>
                <div style="background: #0d1117; padding: 10px; border-radius: 8px;">
                    [{format_complex(input_vec[0])}]<br>
                    [{format_complex(input_vec[1])}]
                </div>
            </div>
            
            <div style="font-size: 1.5em; color: #7c4dff;">=</div>
            
            <div style="text-align: center;">
                <div style="color: #3fb950; margin-bottom: 4px;">Output</div>
                <div style="background: #0d1117; padding: 10px; border-radius: 8px; border: 1px solid #3fb950;">
                    [{format_complex(output_vec[0])}]<br>
                    [{format_complex(output_vec[1])}]
                </div>
            </div>
        </div>
        
        <div style="margin-top: 16px; text-align: center;">
            <span style="color: #9e9e9e;">Result: </span>
            <span style="color: #3fb950; font-size: 1.3em;">{output_state_name}</span>
        </div>
        
        <div style="margin-top: 12px; color: #9e9e9e; font-size: 0.9em;">
            Probabilities: |0⟩ = {np.abs(output_vec[0])**2:.4f}, |1⟩ = {np.abs(output_vec[1])**2:.4f}
        </div>
    </div>
    """
    
    return calculation, output_state_name


def add_dirac_notation_study_tab():
    """Create the Dirac Notation & Quantum States study tab."""
    
    with gr.Tab("📐 Dirac Notation"):
        gr.Markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="
                font-size: 2em;
                background: linear-gradient(135deg, #4fc3f7 0%, #7c4dff 50%, #ff4081 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 8px;
            ">📐 Dirac Notation & Quantum States</h1>
            <p style="color: #8b949e; font-size: 1.1em;">
                The mathematical language of quantum mechanics
            </p>
        </div>
        """)
        
        # Introduction Section
        with gr.Accordion("📚 Introduction to Dirac Notation", open=True):
            gr.Markdown(r"""
            ## What is Dirac Notation?
            
            **Dirac notation** (also called **bra-ket notation**) is the standard mathematical notation 
            used in quantum mechanics and quantum computing. Invented by physicist Paul Dirac, it provides 
            a clean and powerful way to represent quantum states and operations.
            
            ### The Basics
            
            | Symbol | Name | Meaning |
            |--------|------|---------|
            | $\vert\psi\rangle$ | **Ket** | A quantum state (column vector) |
            | $\langle\psi\vert$ | **Bra** | The conjugate transpose of a ket (row vector) |
            | $\langle\phi\vert\psi\rangle$ | **Bracket** (Inner Product) | Overlap between two states |
            | $\vert\phi\rangle\langle\psi\vert$ | **Outer Product** | Operator (matrix) |
            
            ### Why Use Dirac Notation?
            
            1. **Abstraction**: Focus on physics without getting lost in matrix details
            2. **Clarity**: Operations like inner products are immediately visible
            3. **Universality**: Standard across quantum mechanics, QFT, and quantum computing
            4. **Elegance**: Complex operations can be written concisely
            
            ---
            
            ### The Computational Basis
            
            In quantum computing, we work with **qubits**. The two fundamental basis states are:
            
            $$\vert 0 \rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \quad \text{and} \quad \vert 1 \rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$
            
            Any single-qubit state can be written as a **superposition**:
            
            $$\vert \psi \rangle = \alpha \vert 0 \rangle + \beta \vert 1 \rangle = \begin{pmatrix} \alpha \\ \beta \end{pmatrix}$$
            
            where $\alpha$ and $\beta$ are **complex amplitudes** satisfying $|\alpha|^2 + |\beta|^2 = 1$ (normalization).
            """)
        
        # Common States Section
        with gr.Accordion("🌟 Common Quantum States", open=True):
            gr.Markdown("""
            ## Important Single-Qubit States
            
            These states appear constantly in quantum computing. Memorize them!
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML(create_state_vector_html(
                        "|0⟩ (Computational Zero)",
                        ["1", "0"],
                        "Ground state. Always measured as 0."
                    ))
                    gr.HTML(create_state_vector_html(
                        "|1⟩ (Computational One)",
                        ["0", "1"],
                        "Excited state. Always measured as 1."
                    ))
                
                with gr.Column():
                    gr.HTML(create_state_vector_html(
                        "|+⟩ (Plus State)",
                        ["1/√2", "1/√2"],
                        "Equal superposition. 50% chance of 0 or 1. Created by H|0⟩."
                    ))
                    gr.HTML(create_state_vector_html(
                        "|-⟩ (Minus State)",
                        ["1/√2", "-1/√2"],
                        "Equal superposition with phase. Created by H|1⟩."
                    ))
            
            with gr.Row():
                with gr.Column():
                    gr.HTML(create_state_vector_html(
                        "|i⟩ (Y+ State)",
                        ["1/√2", "i/√2"],
                        "Circular polarization. On +Y axis of Bloch sphere."
                    ))
                
                with gr.Column():
                    gr.HTML(create_state_vector_html(
                        "|-i⟩ (Y- State)",
                        ["1/√2", "-i/√2"],
                        "Opposite circular polarization. On -Y axis of Bloch sphere."
                    ))
            
            gr.Markdown(r"""
            ### Relationship Between States
            
            | Basis | States | Relationship |
            |-------|--------|--------------|
            | **Z (Computational)** | $\vert 0\rangle$, $\vert 1\rangle$ | Eigenstates of Z gate |
            | **X (Hadamard)** | $\vert +\rangle$, $\vert -\rangle$ | Eigenstates of X gate; $\vert\pm\rangle = \frac{1}{\sqrt{2}}(\vert 0\rangle \pm \vert 1\rangle)$ |
            | **Y (Circular)** | $\vert i\rangle$, $\vert{-i}\rangle$ | Eigenstates of Y gate; $\vert\pm i\rangle = \frac{1}{\sqrt{2}}(\vert 0\rangle \pm i\vert 1\rangle)$ |
            
            > **Key Insight**: The Hadamard gate $H$ converts between Z and X bases: $H\vert 0\rangle = \vert +\rangle$ and $H\vert 1\rangle = \vert -\rangle$
            """)
        
        # Gates as Matrices Section
        with gr.Accordion("⚙️ Gates as Matrix Operations", open=True):
            gr.Markdown("""
            ## Quantum Gates = Unitary Matrices
            
            Every quantum gate is a **unitary matrix** $U$ satisfying $U^†U = I$ (preserves normalization).
            Applying a gate to a state is matrix-vector multiplication: $\vert\psi'\rangle = U\vert\psi\rangle$
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.HTML(create_gate_matrix_html(
                        "I (Identity)",
                        [["1", "0"], ["0", "1"]],
                        "Does nothing. |ψ⟩ → |ψ⟩"
                    ))
                    gr.HTML(create_gate_matrix_html(
                        "X (Pauli-X / NOT)",
                        [["0", "1"], ["1", "0"]],
                        "Bit flip. |0⟩ ↔ |1⟩. Rotation by π around X-axis."
                    ))
                    gr.HTML(create_gate_matrix_html(
                        "Y (Pauli-Y)",
                        [["0", "-i"], ["i", "0"]],
                        "Bit + phase flip. Rotation by π around Y-axis."
                    ))
                
                with gr.Column():
                    gr.HTML(create_gate_matrix_html(
                        "Z (Pauli-Z)",
                        [["1", "0"], ["0", "-1"]],
                        "Phase flip. |1⟩ → -|1⟩. Rotation by π around Z-axis."
                    ))
                    gr.HTML(create_gate_matrix_html(
                        "H (Hadamard)",
                        [["1/√2", "1/√2"], ["1/√2", "-1/√2"]],
                        "Creates superposition. Rotation by π around (X+Z)/√2."
                    ))
                    gr.HTML(create_gate_matrix_html(
                        "S (Phase Gate)",
                        [["1", "0"], ["0", "i"]],
                        "π/2 phase on |1⟩. S² = Z."
                    ))
            
            gr.Markdown(r"""
            ### Key Relationships
            
            - **Pauli Gates**: $X^2 = Y^2 = Z^2 = I$ (self-inverse)
            - **Hadamard**: $H^2 = I$ (self-inverse), $HXH = Z$, $HZH = X$
            - **Phase Gates**: $S^2 = Z$, $T^2 = S$, $T^4 = Z$
            - **Composition**: $XYZ = iI$ (up to global phase)
            """)
        
        # Interactive Calculator
        with gr.Accordion("🧮 Interactive Gate Calculator", open=True):
            gr.Markdown("""
            ## Try It Yourself!
            
            Select a gate and an input state to see the matrix multiplication in action.
            """)
            
            with gr.Row():
                gate_select = gr.Dropdown(
                    choices=["I (Identity)", "X (NOT)", "Y", "Z", "H (Hadamard)", "S (Phase)", "T (π/8)"],
                    value="H (Hadamard)",
                    label="Select Gate"
                )
                state_select = gr.Dropdown(
                    choices=["|0⟩", "|1⟩", "|+⟩", "|-⟩", "|i⟩", "|-i⟩"],
                    value="|0⟩",
                    label="Select Input State"
                )
                calc_btn = gr.Button("Calculate", variant="primary")
            
            calculation_output = gr.HTML(label="Calculation Result")
            result_state = gr.Textbox(label="Output State", interactive=False)
            
            calc_btn.click(
                fn=calculate_gate_action,
                inputs=[gate_select, state_select],
                outputs=[calculation_output, result_state]
            )
            
            # Auto-calculate on selection change
            gate_select.change(
                fn=calculate_gate_action,
                inputs=[gate_select, state_select],
                outputs=[calculation_output, result_state]
            )
            state_select.change(
                fn=calculate_gate_action,
                inputs=[gate_select, state_select],
                outputs=[calculation_output, result_state]
            )
        
        # Inner Products Section
        with gr.Accordion("📏 Inner Products & Measurement", open=False):
            gr.Markdown(r"""
            ## The Inner Product (Bracket)
            
            The **inner product** $\langle\phi|\psi\rangle$ measures the "overlap" between two quantum states.
            
            $$\langle\phi|\psi\rangle = \begin{pmatrix} \phi_0^* & \phi_1^* \end{pmatrix} \begin{pmatrix} \psi_0 \\ \psi_1 \end{pmatrix} = \phi_0^*\psi_0 + \phi_1^*\psi_1$$
            
            ### Key Properties
            
            | Inner Product | Value | Meaning |
            |--------------|-------|---------|
            | $\langle\psi\vert\psi\rangle$ | 1 | States are normalized |
            | $\langle 0\vert 1\rangle$ | 0 | Orthogonal (distinguishable) |
            | $\langle +\vert 0\rangle$ | $\frac{1}{\sqrt{2}}$ | Partial overlap |
            | $\vert\langle\phi\vert\psi\rangle\vert^2$ | Probability | Measurement probability |
            
            ### Connection to Measurement
            
            When measuring state $|\psi\rangle$ in the computational basis:
            
            - **Probability of getting 0**: $|\langle 0|\psi\rangle|^2 = |\alpha|^2$
            - **Probability of getting 1**: $|\langle 1|\psi\rangle|^2 = |\beta|^2$
            
            This is the **Born Rule** - the foundation of quantum measurement.
            
            ### Examples
            
            For state $|+\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)$:
            
            $$P(0) = |\langle 0|+\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$
            $$P(1) = |\langle 1|+\rangle|^2 = \left|\frac{1}{\sqrt{2}}\right|^2 = \frac{1}{2}$$
            
            Equal superposition → 50/50 measurement outcomes!
            """)
        
        # Multi-qubit States Section
        with gr.Accordion("🔗 Multi-Qubit States & Tensor Products", open=False):
            gr.Markdown(r"""
            ## Combining Qubits: The Tensor Product
            
            When we have multiple qubits, we combine their state spaces using the **tensor product** (⊗).
            
            ### Two-Qubit Computational Basis
            
            $$|00\rangle = |0\rangle \otimes |0\rangle = \begin{pmatrix} 1 \\ 0 \\ 0 \\ 0 \end{pmatrix}, \quad
            |01\rangle = |0\rangle \otimes |1\rangle = \begin{pmatrix} 0 \\ 1 \\ 0 \\ 0 \end{pmatrix}$$
            
            $$|10\rangle = |1\rangle \otimes |0\rangle = \begin{pmatrix} 0 \\ 0 \\ 1 \\ 0 \end{pmatrix}, \quad
            |11\rangle = |1\rangle \otimes |1\rangle = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}$$
            
            ### Tensor Product Formula
            
            For $|a\rangle = \begin{pmatrix} a_0 \\ a_1 \end{pmatrix}$ and $|b\rangle = \begin{pmatrix} b_0 \\ b_1 \end{pmatrix}$:
            
            $$|a\rangle \otimes |b\rangle = \begin{pmatrix} a_0 b_0 \\ a_0 b_1 \\ a_1 b_0 \\ a_1 b_1 \end{pmatrix}$$
            
            ### Important: Entanglement
            
            Some states **cannot** be written as tensor products. These are **entangled states**:
            
            $$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \neq |a\rangle \otimes |b\rangle$$
            
            This Bell state exhibits quantum correlations that have no classical analog!
            
            ### General n-Qubit State
            
            An n-qubit system has $2^n$ basis states and requires $2^n$ complex amplitudes:
            
            $$|\psi\rangle = \sum_{i=0}^{2^n-1} \alpha_i |i\rangle$$
            
            | Qubits | Basis States | Amplitudes |
            |--------|--------------|------------|
            | 1 | 2 | 2 |
            | 2 | 4 | 4 |
            | 3 | 8 | 8 |
            | 10 | 1,024 | 1,024 |
            | 50 | ~10^15 | ~10^15 |
            
            > **This exponential scaling is why quantum computers are powerful** - and why simulating them classically is hard!
            """)
        
        # Professional Notes Section
        with gr.Accordion("🎓 Professional Notes", open=False):
            gr.Markdown(r"""
            ## Advanced Concepts for Practitioners
            
            ### Density Matrices
            
            For mixed states (statistical ensembles), we use **density matrices**:
            
            $$\rho = \sum_i p_i |\psi_i\rangle\langle\psi_i|$$
            
            Properties:
            - $\text{Tr}(\rho) = 1$ (normalized)
            - $\rho^\dagger = \rho$ (Hermitian)
            - $\rho \geq 0$ (positive semidefinite)
            - $\text{Tr}(\rho^2) = 1$ for pure states, $< 1$ for mixed
            
            ### Bloch Sphere Representation
            
            Any single-qubit state can be written as:
            
            $$|\psi\rangle = \cos\frac{\theta}{2}|0\rangle + e^{i\phi}\sin\frac{\theta}{2}|1\rangle$$
            
            This maps to a point on the Bloch sphere at $(\theta, \phi)$.
            
            For density matrices:
            $$\rho = \frac{1}{2}(I + \vec{r} \cdot \vec{\sigma})$$
            
            where $\vec{r}$ is the Bloch vector and $\vec{\sigma} = (X, Y, Z)$ are Pauli matrices.
            
            ### Operator Notation
            
            - **Projectors**: $P_0 = |0\rangle\langle 0|$, $P_1 = |1\rangle\langle 1|$
            - **Completeness**: $|0\rangle\langle 0| + |1\rangle\langle 1| = I$
            - **Measurement**: State after measuring 0 is $\frac{P_0|\psi\rangle}{\sqrt{\langle\psi|P_0|\psi\rangle}}$
            
            ### Commutators
            
            The commutator $[A, B] = AB - BA$ is crucial:
            
            - $[X, Y] = 2iZ$ (cyclic)
            - $[H, X] \neq 0$ (don't commute)
            - Gates commute ⟺ can be reordered ⟺ no interference
            
            ### Useful Identities
            
            - $e^{i\theta X} = \cos\theta \cdot I + i\sin\theta \cdot X$ (Euler formula for Paulis)
            - $HZH = X$, $HXH = Z$ (basis change)
            - $(A \otimes B)(C \otimes D) = (AC) \otimes (BD)$ (tensor product of operators)
            """)


# Export the function
__all__ = ['add_dirac_notation_study_tab']
