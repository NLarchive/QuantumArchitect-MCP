"""
GHZ State Study Tab for QuantumArchitect-MCP
"""

import gradio as gr


def add_ghz_state_study_tab():
    """Add the GHZ State Study tab to the Gradio interface."""
    with gr.TabItem("🌟 GHZ State Study", id="ghz-study"):
        gr.Markdown("""
        # 🌟 Deep Dive: The GHZ State

        The **GHZ (Greenberger–Horne–Zeilinger) State** extends entanglement to 3+ qubits.
        It's crucial for quantum error correction and multi-party quantum protocols.

        ---

        ## 📊 The Circuit
        """)

        ghz_svg = '''
        <div style="display: flex; justify-content: center; padding: 20px;">
        <svg width="550" height="220" xmlns="http://www.w3.org/2000/svg"
             style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;">
            <defs>
                <filter id="ghz-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>

            <!-- Title -->
            <text x="275" y="25" text-anchor="middle" fill="#ab47bc" font-size="16" font-weight="bold">
                GHZ State Circuit (3 qubits)
            </text>

            <!-- Qubit labels -->
            <text x="30" y="65" fill="#ce93d8" font-size="13" font-weight="bold">|q₀⟩ = |0⟩</text>
            <text x="30" y="115" fill="#ce93d8" font-size="13" font-weight="bold">|q₁⟩ = |0⟩</text>
            <text x="30" y="165" fill="#ce93d8" font-size="13" font-weight="bold">|q₂⟩ = |0⟩</text>

            <!-- Wires -->
            <line x1="110" y1="60" x2="480" y2="60" stroke="#ab47bc" stroke-width="2"/>
            <line x1="110" y1="110" x2="480" y2="110" stroke="#ab47bc" stroke-width="2"/>
            <line x1="110" y1="160" x2="480" y2="160" stroke="#ab47bc" stroke-width="2"/>

            <!-- Hadamard gate on q0 -->
            <rect x="140" y="42" width="40" height="36" rx="6" fill="#00897B" filter="url(#ghz-glow)"/>
            <text x="160" y="67" text-anchor="middle" fill="white" font-size="16" font-weight="bold">H</text>

            <!-- CNOT 1: q0 → q1 -->
            <line x1="230" y1="60" x2="230" y2="110" stroke="#F57C00" stroke-width="3"/>
            <circle cx="230" cy="60" r="8" fill="#F57C00"/>
            <circle cx="230" cy="110" r="14" fill="none" stroke="#F57C00" stroke-width="3"/>
            <line x1="230" y1="96" x2="230" y2="124" stroke="#F57C00" stroke-width="3"/>
            <line x1="216" y1="110" x2="244" y2="110" stroke="#F57C00" stroke-width="3"/>

            <!-- CNOT 2: q0 → q2 -->
            <line x1="310" y1="60" x2="310" y2="160" stroke="#F57C00" stroke-width="3"/>
            <circle cx="310" cy="60" r="8" fill="#F57C00"/>
            <circle cx="310" cy="160" r="14" fill="none" stroke="#F57C00" stroke-width="3"/>
            <line x1="310" y1="146" x2="310" y2="174" stroke="#F57C00" stroke-width="3"/>
            <line x1="296" y1="160" x2="324" y2="160" stroke="#F57C00" stroke-width="3"/>

            <!-- Measurement boxes -->
            <rect x="410" y="45" width="28" height="28" rx="4" fill="#263238" stroke="#546e7a"/>
            <rect x="410" y="95" width="28" height="28" rx="4" fill="#263238" stroke="#546e7a"/>
            <rect x="410" y="145" width="28" height="28" rx="4" fill="#263238" stroke="#546e7a"/>

            <!-- Output labels -->
            <text x="460" y="65" fill="#ce93d8" font-size="11">→ c₀</text>
            <text x="460" y="115" fill="#ce93d8" font-size="11">→ c₁</text>
            <text x="460" y="165" fill="#ce93d8" font-size="11">→ c₂</text>

            <!-- Step labels -->
            <text x="160" y="195" text-anchor="middle" fill="#78909c" font-size="10">Step 1</text>
            <text x="230" y="195" text-anchor="middle" fill="#78909c" font-size="10">Step 2</text>
            <text x="310" y="195" text-anchor="middle" fill="#78909c" font-size="10">Step 3</text>
        </svg>
        </div>
        '''
        gr.HTML(ghz_svg)

        gr.Markdown("""
        ---

        ## 🔍 Step-by-Step Breakdown
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Step 1: Hadamard

                Creates superposition on q₀:

                ```
                |ψ₁⟩ = (|000⟩ + |100⟩)/√2
                ```
                """)

            with gr.Column():
                gr.Markdown("""
                ### Step 2: CNOT q₀→q₁

                Entangles q₀ with q₁:

                ```
                |ψ₂⟩ = (|000⟩ + |110⟩)/√2
                ```
                """)

            with gr.Column():
                gr.Markdown("""
                ### Step 3: CNOT q₀→q₂

                Completes 3-qubit entanglement:

                ```
                |ψ₃⟩ = (|000⟩ + |111⟩)/√2
                ```

                🎉 **Full GHZ state!**
                """)

        gr.Markdown("""
        ---

        ## ⚡ Why Is GHZ Important?

        | Property | Explanation |
        |----------|-------------|
        | **Maximum Entanglement** | All qubits are maximally correlated |
        | **Quantum Error Correction** | Used in error detection codes |
        | **Multi-party Protocols** | Quantum secret sharing |
        | **Bell Inequality Tests** | Strongest violations of classical physics |
        | **Quantum Networks** | Foundation for distributed quantum computing |

        ---

        ## 📐 Mathematical Details

        **The GHZ State:**

        $$|GHZ\\rangle = \\frac{1}{\\sqrt{2}}(|000\\rangle + |111\\rangle)$$

        **Generalized n-qubit GHZ:**

        $$|GHZ_n\\rangle = \\frac{1}{\\sqrt{2}}(|0\\rangle^{\\otimes n} + |1\\rangle^{\\otimes n})$$
        """)