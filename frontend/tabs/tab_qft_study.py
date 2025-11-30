"""
QFT Study Tab for QuantumArchitect-MCP
"""

import gradio as gr


def add_qft_study_tab():
    """Add the QFT Study tab to the Gradio interface."""
    with gr.TabItem("🌀 QFT Study", id="qft-study"):
        gr.Markdown("""
        # 🌀 Deep Dive: Quantum Fourier Transform

        The **QFT** is the quantum analog of the classical Discrete Fourier Transform.
        It's a key subroutine in Shor's algorithm and phase estimation.

        ---

        ## 📊 The 2-Qubit QFT Circuit
        """)

        qft_svg = '''
        <div style="display: flex; justify-content: center; padding: 20px;">
        <svg width="500" height="180" xmlns="http://www.w3.org/2000/svg"
             style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;">
            <defs>
                <filter id="qft-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>

            <!-- Title -->
            <text x="250" y="25" text-anchor="middle" fill="#42a5f5" font-size="16" font-weight="bold">
                2-Qubit Quantum Fourier Transform
            </text>

            <!-- Qubit labels -->
            <text x="30" y="75" fill="#90caf9" font-size="14" font-weight="bold">|q₀⟩</text>
            <text x="30" y="135" fill="#90caf9" font-size="14" font-weight="bold">|q₁⟩</text>

            <!-- Wires -->
            <line x1="70" y1="70" x2="450" y2="70" stroke="#42a5f5" stroke-width="2"/>
            <line x1="70" y1="130" x2="450" y2="130" stroke="#42a5f5" stroke-width="2"/>

            <!-- H gate on q0 -->
            <rect x="100" y="52" width="36" height="36" rx="6" fill="#00897B" filter="url(#qft-glow)"/>
            <text x="118" y="77" text-anchor="middle" fill="white" font-size="16" font-weight="bold">H</text>

            <!-- Controlled-S (CPHASE π/2) -->
            <line x1="180" y1="70" x2="180" y2="130" stroke="#ff7043" stroke-width="2"/>
            <circle cx="180" cy="130" r="8" fill="#ff7043"/>
            <rect x="160" y="52" width="40" height="36" rx="6" fill="#5c6bc0" filter="url(#qft-glow)"/>
            <text x="180" y="77" text-anchor="middle" fill="white" font-size="12" font-weight="bold">S†</text>

            <!-- SWAP arrows -->
            <line x1="260" y1="70" x2="280" y2="130" stroke="#ffd54f" stroke-width="3"/>
            <line x1="260" y1="130" x2="280" y2="70" stroke="#ffd54f" stroke-width="3"/>
            <text x="270" y="155" text-anchor="middle" fill="#ffd54f" font-size="10">SWAP</text>

            <!-- H gate on q1 (after swap position) -->
            <rect x="320" y="52" width="36" height="36" rx="6" fill="#00897B" filter="url(#qft-glow)"/>
            <text x="338" y="77" text-anchor="middle" fill="white" font-size="16" font-weight="bold">H</text>

            <!-- Output labels -->
            <text x="420" y="75" fill="#90caf9" font-size="12">|ω₀⟩</text>
            <text x="420" y="135" fill="#90caf9" font-size="12">|ω₁⟩</text>
        </svg>
        </div>
        '''
        gr.HTML(qft_svg)

        gr.Markdown("""
        ---

        ## 🔍 Key Components
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Hadamard Gates

                Create superposition of all basis states.

                Each H gate transforms:
                ```
                |0⟩ → (|0⟩ + |1⟩)/√2
                |1⟩ → (|0⟩ - |1⟩)/√2
                ```
                """)

            with gr.Column():
                gr.Markdown("""
                ### Controlled Rotations

                Apply phase shifts conditioned on other qubits.

                For n qubits, controlled-R_k:
                ```
                R_k = diag(1, e^(2πi/2^k))
                ```
                """)

            with gr.Column():
                gr.Markdown("""
                ### SWAP Network

                Reverses qubit order at the end.

                Required because QFT outputs in reversed bit order.
                """)

        gr.Markdown("""
        ---

        ## ⚡ Applications

        | Algorithm | How QFT Is Used |
        |-----------|-----------------|
        | **Shor's Algorithm** | Period finding for factoring |
        | **Phase Estimation** | Estimate eigenvalues of unitaries |
        | **Quantum Chemistry** | Energy level calculations |
        | **Signal Processing** | Quantum signal analysis |

        ---

        ## 📐 Mathematical Details

        **QFT Definition:**

        $$QFT|j\\rangle = \\frac{1}{\\sqrt{N}} \\sum_{k=0}^{N-1} e^{2\\pi ijk/N} |k\\rangle$$

        **Circuit Depth:** O(n²) for n qubits

        **Classical FFT Complexity:** O(n·2ⁿ)

        **Quantum Speedup:** Exponential advantage!
        """)