"""
Bell State Study Tab for QuantumArchitect-MCP
"""

import gradio as gr


def add_bell_state_study_tab():
    """Add the Bell State Study tab to the Gradio interface."""
    with gr.TabItem("🔔 Bell State Study", id="bell-study"):
        gr.Markdown("""
        # 🔔 Deep Dive: The Bell State

        The **Bell State** is the most important circuit to understand in quantum computing.
        It demonstrates **entanglement** - the key resource that makes quantum computers powerful.

        ---

        ## 📊 The Circuit
        """)

        # Visual Bell State circuit
        bell_svg = '''
        <div style="display: flex; justify-content: center; padding: 20px;">
        <svg width="500" height="180" xmlns="http://www.w3.org/2000/svg"
             style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;">
            <defs>
                <filter id="glow2" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                    <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
            </defs>

            <!-- Title -->
            <text x="250" y="25" text-anchor="middle" fill="#4fc3f7" font-size="16" font-weight="bold">
                Bell State Circuit (|Φ+⟩)
            </text>

            <!-- Qubit labels -->
            <text x="30" y="75" fill="#81d4fa" font-size="14" font-weight="bold">|q₀⟩ = |0⟩</text>
            <text x="30" y="135" fill="#81d4fa" font-size="14" font-weight="bold">|q₁⟩ = |0⟩</text>

            <!-- Wires -->
            <line x1="110" y1="70" x2="420" y2="70" stroke="#4fc3f7" stroke-width="2"/>
            <line x1="110" y1="130" x2="420" y2="130" stroke="#4fc3f7" stroke-width="2"/>

            <!-- Hadamard gate on q0 -->
            <rect x="140" y="52" width="40" height="36" rx="6" fill="#00897B" filter="url(#glow2)"/>
            <text x="160" y="77" text-anchor="middle" fill="white" font-size="16" font-weight="bold">H</text>

            <!-- CNOT gate -->
            <line x1="250" y1="70" x2="250" y2="130" stroke="#F57C00" stroke-width="3"/>
            <circle cx="250" cy="70" r="10" fill="#F57C00"/>
            <circle cx="250" cy="130" r="18" fill="none" stroke="#F57C00" stroke-width="3"/>
            <line x1="250" y1="112" x2="250" y2="148" stroke="#F57C00" stroke-width="3"/>
            <line x1="232" y1="130" x2="268" y2="130" stroke="#F57C00" stroke-width="3"/>

            <!-- Measurement boxes -->
            <rect x="350" y="55" width="30" height="30" rx="4" fill="#263238" stroke="#546e7a"/>
            <path d="M358 78 L365 63 L372 78" fill="none" stroke="#90a4ae" stroke-width="1.5"/>
            <rect x="350" y="115" width="30" height="30" rx="4" fill="#263238" stroke="#546e7a"/>
            <path d="M358 138 L365 123 L372 138" fill="none" stroke="#90a4ae" stroke-width="1.5"/>

            <!-- Output labels -->
            <text x="430" y="75" fill="#81d4fa" font-size="12">→ c₀</text>
            <text x="430" y="135" fill="#81d4fa" font-size="12">→ c₁</text>

            <!-- Step labels -->
            <text x="160" y="160" text-anchor="middle" fill="#78909c" font-size="11">Step 1</text>
            <text x="250" y="160" text-anchor="middle" fill="#78909c" font-size="11">Step 2</text>
            <text x="365" y="160" text-anchor="middle" fill="#78909c" font-size="11">Step 3</text>
        </svg>
        </div>
        '''
        gr.HTML(bell_svg)

        gr.Markdown("""
        ---

        ## 🔍 Step-by-Step Breakdown
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ### Step 1: Initial State

                Both qubits start in state |0⟩:

                ```
                |ψ₀⟩ = |0⟩ ⊗ |0⟩ = |00⟩
                ```

                **Probability:** 100% chance of measuring 00
                """)

            with gr.Column():
                gr.Markdown("""
                ### Step 2: Hadamard on q₀

                The H gate creates superposition:

                ```
                |ψ₁⟩ = (|0⟩ + |1⟩)/√2 ⊗ |0⟩
                     = (|00⟩ + |10⟩)/√2
                ```

                **Probability:** 50% |00⟩, 50% |10⟩
                """)

            with gr.Column():
                gr.Markdown("""
                ### Step 3: CNOT (CX)

                CNOT flips q₁ when q₀ is |1⟩:

                ```
                |ψ₂⟩ = (|00⟩ + |11⟩)/√2
                ```

                **Probability:** 50% |00⟩, 50% |11⟩

                🎉 **This is entanglement!**
                """)

        gr.Markdown("""
        ---

        ## ⚡ Why Is This Important?

        ### The "Spooky" Correlation

        After creating the Bell state, the two qubits are **entangled**:

        - If you measure q₀ and get **0**, q₁ will **always** be **0**
        - If you measure q₀ and get **1**, q₁ will **always** be **1**

        This correlation happens **instantly**, regardless of distance! Einstein called this
        "spooky action at a distance."

        ### Real-World Applications

        | Application | How Bell States Are Used |
        |-------------|-------------------------|
        | **Quantum Teleportation** | Transfer quantum state using entanglement |
        | **Superdense Coding** | Send 2 classical bits using 1 qubit |
        | **Quantum Key Distribution** | Secure cryptographic key exchange |
        | **Bell Test Experiments** | Prove quantum mechanics is correct |
        | **Quantum Error Correction** | Detect and correct errors |

        ---

        ## 🧪 Try It Yourself!

        1. Go to **⚛️ Circuit Builder**
        2. Set **Qubit 1 = 0**, **Qubit 2 = 1**
        3. Click **H** to add Hadamard to qubit 0
        4. Change **Qubit 1 = 0**, keep **Qubit 2 = 1**
        5. Click **CX** to add CNOT (control=0, target=1)
        6. Click **▶️ Simulate**

        **Expected Results:**
        - |00⟩: ~50%
        - |11⟩: ~50%
        - |01⟩: 0%
        - |10⟩: 0%

        The fact that you **never** see |01⟩ or |10⟩ proves entanglement!

        ---

        ## 📐 Mathematical Details

        **The Bell State |Φ+⟩:**

        $$|\\Phi^+\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)$$

        **Full Set of Bell States:**

        | Name | State | Circuit |
        |------|-------|---------|
        | |Φ+⟩ | (|00⟩ + |11⟩)/√2 | H, CX |
        | |Φ-⟩ | (|00⟩ - |11⟩)/√2 | H, CX, Z |
        | |Ψ+⟩ | (|01⟩ + |10⟩)/√2 | H, CX, X |
        | |Ψ-⟩ | (|01⟩ - |10⟩)/√2 | H, CX, X, Z |
        """)