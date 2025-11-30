
# =============================================================================
# CSS THEME - IBM Quantum Composer Style (Compact - No Dead Space)
# =============================================================================

IBM_COMPOSER_CSS = """
/* IBM Quantum Composer inspired theme - COMPACT VERSION */
.gradio-container {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
}

.dark {
    --body-background-fill: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    --block-background-fill: #21262d;
    --block-border-color: #30363d;
    --input-background-fill: #161b22;
    --button-primary-background-fill: #238636;
    --button-primary-background-fill-hover: #2ea043;
    --block-padding: 8px !important;
}

/* Global compact spacing */
.gradio-container > .main {
    padding: 8px !important;
}

/* Tab styling */
.tab-nav button {
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    color: #8b949e !important;
    font-weight: 500 !important;
    padding: 6px 12px !important;
}

.tab-nav button.selected {
    background: transparent !important;
    border-bottom: 2px solid #58a6ff !important;
    color: #58a6ff !important;
}

/* Card styling - reduced padding */
.gr-box {
    border-radius: 8px !important;
    border: 1px solid #30363d !important;
    background: #21262d !important;
    padding: 6px !important;
}

/* Button styling */
.gr-button-primary {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 6px 12px !important;
}

.gr-button-secondary {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    padding: 6px 12px !important;
}

/* Gate button categories - compact */
.gate-pauli button, button.gate-pauli {
    background: linear-gradient(135deg, #1565C0, #1976D2) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    min-width: 40px !important;
    padding: 4px 8px !important;
}
.gate-pauli button:hover, button.gate-pauli:hover {
    background: linear-gradient(135deg, #1976D2, #2196F3) !important;
    transform: scale(1.02);
}

.gate-hadamard button, button.gate-hadamard {
    background: linear-gradient(135deg, #00695C, #00897B) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    min-width: 40px !important;
    padding: 4px 8px !important;
}
.gate-hadamard button:hover, button.gate-hadamard:hover {
    background: linear-gradient(135deg, #00897B, #26A69A) !important;
}

.gate-phase button, button.gate-phase {
    background: linear-gradient(135deg, #00838F, #00ACC1) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    min-width: 40px !important;
    padding: 4px 8px !important;
}

.gate-rotation button, button.gate-rotation {
    background: linear-gradient(135deg, #6A1B9A, #7B1FA2) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    min-width: 40px !important;
    padding: 4px 8px !important;
}
.gate-rotation button:hover, button.gate-rotation:hover {
    background: linear-gradient(135deg, #7B1FA2, #9C27B0) !important;
}

.gate-multi button, button.gate-multi {
    background: linear-gradient(135deg, #E65100, #F57C00) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    min-width: 40px !important;
    padding: 4px 8px !important;
}
.gate-multi button:hover, button.gate-multi:hover {
    background: linear-gradient(135deg, #F57C00, #FF9800) !important;
}

.gate-multi3 button, button.gate-multi3 {
    background: linear-gradient(135deg, #C62828, #D32F2F) !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    min-width: 40px !important;
    padding: 4px 8px !important;
}
.gate-multi3 button:hover, button.gate-multi3:hover {
    background: linear-gradient(135deg, #D32F2F, #E53935) !important;
}

/* Category headers - compact */
.category-header {
    color: #8b949e !important;
    font-size: 0.8em !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 4px !important;
}

/* Code blocks - compact */
.code-wrap {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 4px !important;
}

/* Slider - compact */
input[type="range"] {
    accent-color: #58a6ff !important;
}

/* Headers - compact */
h1, h2, h3, h4, h5 {
    color: #f0f6fc !important;
    margin: 4px 0 !important;
}

/* Quantum-specific styling */
.quantum-wire {
    stroke: #58a6ff;
    stroke-width: 2px;
}

.quantum-gate {
    rx: 4px;
    fill: #238636;
    stroke: #2ea043;
    stroke-width: 2px;
}

/* Group styling for palette - compact */
.gr-group {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 6px !important;
    margin-bottom: 4px !important;
}

/* Row button alignment - compact */
.gr-button-sm {
    min-width: 40px !important;
    padding: 4px 8px !important;
}

/* =============================================================================
   BENTO GRID LAYOUT - COMPACT, NO DEAD SPACE
   ============================================================================= */

/* Bento grid container - tighter layout */
.bento-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
    gap: 4px !important;
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
}

/* Bento grid items - minimal height */
.bento-item {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 4px !important;
    transition: all 0.2s ease !important;
    overflow: hidden !important;
    min-height: fit-content !important;
    height: fit-content !important;
}

/* Large bento items */
.bento-item-large {
    grid-column: span 2 !important;
    min-height: fit-content !important;
    height: fit-content !important;
}

/* Medium bento items */
.bento-item-medium {
    grid-column: span 1 !important;
    min-height: fit-content !important;
    height: fit-content !important;
}

/* Small bento items */
.bento-item-small {
    grid-column: span 1 !important;
    min-height: fit-content !important;
    height: fit-content !important;
}

/* Responsive breakpoints for bento grid */
@media (max-width: 1200px) {
    .bento-grid {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) !important;
        gap: 4px !important;
    }
    .bento-item-large {
        grid-column: span 1 !important;
    }
}

@media (max-width: 768px) {
    .bento-grid {
        grid-template-columns: 1fr !important;
        gap: 4px !important;
    }
    .bento-item {
        padding: 4px !important;
        min-height: fit-content !important;
        height: fit-content !important;
    }
    .bento-item-large, .bento-item-medium {
        grid-column: span 1 !important;
    }
}

/* Bento item headers - compact */
.bento-header {
    font-size: 0.95em !important;
    font-weight: 600 !important;
    color: #f0f6fc !important;
    margin-bottom: 4px !important;
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
}

/* Visualization containers - fit content */
.bento-viz-container {
    width: 100% !important;
    height: fit-content !important;
    display: flex !important;
    align-items: flex-start !important;
    justify-content: center !important;
    overflow: visible !important;
    padding: 0 !important;
}

/* Ensure visualizations scale properly - compact */
.bento-viz-container svg,
.bento-viz-container .plotly-graph-div,
.bento-viz-container canvas {
    max-width: 100% !important;
    height: auto !important;
    width: auto !important;
    max-height: 250px !important;
}

/* Probability bars specific styling - compact */
.bento-probability {
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Q-sphere and Bloch sphere responsive scaling - fit content */
#qsphere-display svg,
#bloch-display .plotly-graph-div {
    width: 100% !important;
    height: auto !important;
    max-width: 250px !important;
    max-height: 250px !important;
    margin: 0 auto !important;
}

/* Statevector display - compact */
#statevector-display {
    font-family: 'Courier New', monospace !important;
    font-size: 0.7em !important;
    line-height: 1.1 !important;
    white-space: pre-wrap !important;
    word-break: break-all !important;
    padding: 2px !important;
    max-height: 150px !important;
    overflow-y: auto !important;
}

/* Raw results JSON - compact */
#raw-results {
    font-size: 0.65em !important;
    max-height: 120px !important;
    overflow-y: auto !important;
    padding: 2px !important;
}

/* Hover effects for bento items - subtle */
.bento-item:hover {
    border-color: #58a6ff !important;
    box-shadow: 0 2px 8px rgba(88, 166, 255, 0.1) !important;
}

/* Loading states - compact */
.bento-loading {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #8b949e !important;
    font-style: italic !important;
    width: 100% !important;
    height: fit-content !important;
    padding: 4px !important;
    font-size: 0.8em !important;
}

/* Accordion styling - compact */
.gr-accordion {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    margin-bottom: 2px !important;
    overflow: hidden !important;
}

/* Accordion labels - minimal padding */
.gr-accordion-label {
    color: #f0f6fc !important;
    font-weight: 600 !important;
    padding: 4px 8px !important;
    font-size: 0.85em !important;
}

/* Accordion content - minimal padding */
.gr-accordion-content {
    padding: 4px !important;
    background: #1a1a2e !important;
    border-top: 1px solid #30363d !important;
}

/* =============================================================================
   COMPACT FORM ELEMENTS
   ============================================================================= */

/* Input fields compact */
.gr-input, .gr-textbox textarea, .gr-code {
    padding: 4px 6px !important;
    font-size: 0.85em !important;
}

/* Slider compact */
.gr-slider input {
    height: 4px !important;
}

/* Number input compact */
.gr-number input {
    padding: 4px 6px !important;
    font-size: 0.85em !important;
}

/* Dropdown compact */
.gr-dropdown {
    padding: 4px 6px !important;
    font-size: 0.85em !important;
}

/* Label compact */
.gr-input-label, .gr-box > label {
    font-size: 0.8em !important;
    margin-bottom: 2px !important;
}

/* Form spacing */
.gr-form {
    gap: 4px !important;
}

/* Row spacing */
.gr-row {
    gap: 4px !important;
}

/* Column spacing */
.gr-column {
    gap: 4px !important;
}

/* =============================================================================
   COMPACT HTML DISPLAYS
   ============================================================================= */

/* HTML content containers */
.gr-html {
    padding: 0 !important;
    margin: 0 !important;
}

/* SVG sizing in HTML */
.gr-html svg {
    max-width: 100% !important;
    height: auto !important;
    display: block !important;
    margin: 0 auto !important;
}

/* Plotly chart containers */
.js-plotly-plot {
    max-height: 280px !important;
}

/* JSON display compact */
.gr-json {
    font-size: 0.7em !important;
    max-height: 150px !important;
    overflow-y: auto !important;
}

/* Code display compact */
.gr-code {
    font-size: 0.75em !important;
    line-height: 1.2 !important;
}

/* Markdown compact */
.gr-markdown {
    font-size: 0.9em !important;
    line-height: 1.4 !important;
}

.gr-markdown p {
    margin: 4px 0 !important;
}

.gr-markdown h3, .gr-markdown h4 {
    margin: 6px 0 4px 0 !important;
}

/* =============================================================================
   CIRCUIT BUILDER LAYOUT IMPROVEMENTS - NO DEAD SPACE
   ============================================================================= */

/* Taller circuit canvas to eliminate dead space */
.circuit-canvas-tall {
    min-height: 500px !important;
    max-height: 600px !important;
    height: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
}

#circuit-canvas {
    width: 100% !important;
    height: auto !important;
}

#circuit-canvas svg {
    max-width: none !important;
    height: auto !important;
}

/* Compact results layout - 3 column grid */
.result-compact {
    min-height: 200px !important;
    max-height: 350px !important;
    overflow-y: auto !important;
    padding: 8px !important;
}

.results-placeholder {
    color: #8b949e !important;
    text-align: center !important;
    padding: 20px !important;
    font-style: italic !important;
    font-size: 0.9em !important;
}

/* Equal height columns for results */
.gradio-row[equal_height] > .gradio-column {
    display: flex !important;
    flex-direction: column !important;
}

.gradio-row[equal_height] .gr-accordion {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Simulation results - compact probability bars */
#simulation-results {
    max-height: 300px !important;
    overflow-y: auto !important;
}

/* Q-Sphere compact display */
#qsphere-display svg {
    max-width: 100% !important;
    height: auto !important;
    max-height: 280px !important;
}

/* Bloch sphere compact display */
#bloch-display {
    max-height: 350px !important;
}

/* Statevector amplitudes compact */
#statevector-display {
    font-size: 0.8em !important;
    max-height: 300px !important;
    overflow-y: auto !important;
}

/* Raw results compact */
#raw-results {
    font-size: 0.7em !important;
    max-height: 250px !important;
    overflow-y: auto !important;
}

/* Accordion improvements for compact layout */
.gr-accordion {
    margin-bottom: 8px !important;
}

.gr-accordion-header {
    padding: 8px 12px !important;
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

.gr-accordion-content {
    padding: 8px !important;
    border: 1px solid #30363d !important;
    border-top: none !important;
    border-radius: 0 0 6px 6px !important;
}
"""
