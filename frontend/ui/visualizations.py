import json
import math
import cmath
import numpy as np
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore
    
from ..core.constants import GATE_LIBRARY, GATE_CATEGORIES

def plot_bloch_sphere_plotly(statevector_data):
    """
    Create an interactive 3D Bloch sphere using Plotly.
    
    Args:
        statevector_data: Can be:
            - dict with "real" and "imag" lists (from simulation)
            - dict with nested "statevector" key containing "real"/"imag"
            - list/tuple/array of complex numbers [alpha, beta]
            - None (returns default |0⟩ state)
    
    Returns:
        Plotly figure object compatible with gr.Plot(), or None if Plotly unavailable
    """
    if not PLOTLY_AVAILABLE or go is None:
        return None
    
    # Default to |0⟩ state
    alpha, beta = 1.0 + 0j, 0.0 + 0j
    
    try:
        # Handle different statevector formats
        if statevector_data is None:
            pass  # Use default |0⟩
            
        elif isinstance(statevector_data, dict):
            # Check for nested "statevector" key (from simulation result)
            if "statevector" in statevector_data:
                sv_data = statevector_data["statevector"]
                if isinstance(sv_data, dict):
                    real = sv_data.get("real", [1, 0])
                    imag = sv_data.get("imag", [0, 0])
                else:
                    real, imag = [1, 0], [0, 0]
            else:
                # Direct statevector dict {"real": [...], "imag": [...]}
                real = statevector_data.get("real", [1, 0])
                imag = statevector_data.get("imag", [0, 0])
            
            # Ensure we have at least 2 elements
            if len(real) >= 2 and len(imag) >= 2:
                alpha = complex(float(real[0]), float(imag[0]))
                beta = complex(float(real[1]), float(imag[1]))
            elif len(real) >= 2:
                alpha = complex(float(real[0]), 0)
                beta = complex(float(real[1]), 0)
                
        elif isinstance(statevector_data, (list, tuple, np.ndarray)):
            if len(statevector_data) >= 2:
                alpha = complex(statevector_data[0])
                beta = complex(statevector_data[1])
                
    except (TypeError, ValueError, IndexError) as e:
        # Fall back to |0⟩ state on any parsing error
        alpha, beta = 1.0 + 0j, 0.0 + 0j
    
    # Compute Bloch vector coordinates
    # For pure state |ψ⟩ = α|0⟩ + β|1⟩
    # x = 2*Re(α*β̄), y = 2*Im(α*β̄), z = |α|² - |β|²
    x = float(2 * np.real(alpha * np.conj(beta)))
    y = float(2 * np.imag(alpha * np.conj(beta)))
    z = float(np.abs(alpha)**2 - np.abs(beta)**2)
    
    fig = go.Figure()
    
    # Create sphere wireframe (more visible than surface)
    # Equator circle
    theta_eq = np.linspace(0, 2*np.pi, 60)
    fig.add_trace(go.Scatter3d(
        x=np.cos(theta_eq), y=np.sin(theta_eq), z=np.zeros_like(theta_eq),
        mode='lines',
        line=dict(color='rgba(100,150,255,0.4)', width=2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Meridian circles (XZ and YZ planes)
    for phi_offset in [0, np.pi/2]:
        phi_vals = np.linspace(0, 2*np.pi, 60)
        x_merid = np.cos(phi_offset) * np.sin(phi_vals)
        y_merid = np.sin(phi_offset) * np.sin(phi_vals)
        z_merid = np.cos(phi_vals)
        fig.add_trace(go.Scatter3d(
            x=x_merid, y=y_merid, z=z_merid,
            mode='lines',
            line=dict(color='rgba(100,150,255,0.3)', width=1),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Semi-transparent sphere surface
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.15,
        colorscale=[[0, 'rgb(70,130,180)'], [1, 'rgb(70,130,180)']],
        showscale=False,
        hoverinfo='skip'
    ))
    
    # Add coordinate axes with labels
    axis_length = 1.3
    
    # X-axis (red) - |+⟩ to |-⟩
    fig.add_trace(go.Scatter3d(
        x=[-axis_length, axis_length], y=[0, 0], z=[0, 0],
        mode='lines',
        line=dict(color='rgba(255,100,100,0.8)', width=3),
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[axis_length*1.1], y=[0], z=[0],
        mode='text',
        text=['X |+⟩'],
        textfont=dict(size=12, color='#ff6b6b'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Y-axis (green) - |R⟩ to |L⟩
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[-axis_length, axis_length], z=[0, 0],
        mode='lines',
        line=dict(color='rgba(100,255,100,0.8)', width=3),
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[axis_length*1.1], z=[0],
        mode='text',
        text=['Y |R⟩'],
        textfont=dict(size=12, color='#69db7c'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Z-axis (blue) - |0⟩ to |1⟩
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-axis_length, axis_length],
        mode='lines',
        line=dict(color='rgba(100,100,255,0.8)', width=3),
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[axis_length*1.1],
        mode='text',
        text=['Z |0⟩'],
        textfont=dict(size=12, color='#748ffc'),
        hoverinfo='skip',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-axis_length*1.1],
        mode='text',
        text=['|1⟩'],
        textfont=dict(size=12, color='#748ffc'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add state vector arrow (orange/gold)
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines',
        line=dict(color='#ffa500', width=6),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add state vector point with hover info
    # Calculate theta and phi for display
    r = np.sqrt(x**2 + y**2 + z**2)
    theta_angle = np.arccos(z / r) if r > 0 else 0
    phi_angle = np.arctan2(y, x)
    
    state_info = f"|ψ⟩<br>θ={theta_angle:.2f} rad<br>φ={phi_angle:.2f} rad<br>({x:.3f}, {y:.3f}, {z:.3f})"
    
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers',
        marker=dict(size=10, color='#ff4500', symbol='circle'),
        text=[state_info],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Update layout for nice appearance
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-1.5, 1.5], 
                showbackground=False, 
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            yaxis=dict(
                range=[-1.5, 1.5], 
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            zaxis=dict(
                range=[-1.5, 1.5], 
                showbackground=False,
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                title=''
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgba(20,20,30,0.9)'
        ),
        title=dict(
            text='<b>Bloch Sphere</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='#4fc3f7')
        ),
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=50),
        paper_bgcolor='rgba(20,20,30,0.95)',
        height=450
    )

    return fig


def create_placeholder_plot(message: str) -> go.Figure:
    """Creates an empty Plotly figure with a text message.
    
    Used to display informative messages when visualization is not available,
    such as for multi-qubit circuits where Bloch sphere doesn't apply.
    
    Args:
        message: The message to display in the plot
        
    Returns:
        Plotly figure object with centered message
    """
    if not PLOTLY_AVAILABLE or go is None:
        return None
    
    fig = go.Figure()
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#78909c")
            )
        ],
        paper_bgcolor='rgba(20,20,30,0.95)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=20, r=20, b=20, t=50),
        title=dict(
            text='<b>Bloch Sphere</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=16, color='#4fc3f7')
        ),
    )
    return fig


def render_bloch_sphere_svg(theta: float = 0, phi: float = 0, label: str = "|ψ⟩") -> str:
    """
    Render an interactive Bloch sphere as SVG.
    theta: polar angle (0 to π)
    phi: azimuthal angle (0 to 2π)
    """
    # Calculate Bloch vector coordinates
    x = math.sin(theta) * math.cos(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(theta)
    
    # Project 3D to 2D (isometric-ish projection)
    cx, cy = 110, 120  # Center
    scale = 70
    proj_x = cx + scale * (x * 0.866 - y * 0.5)
    proj_y = cy - scale * (z * 0.8 + (x * 0.5 + y * 0.866) * 0.3)
    
    return f'''
    <svg width="220" height="240" xmlns="http://www.w3.org/2000/svg" style="background: #1a1a2e; border-radius: 12px;">
        <defs>
            <radialGradient id="sphereGrad" cx="30%" cy="30%">
                <stop offset="0%" style="stop-color:#4fc3f7;stop-opacity:0.3"/>
                <stop offset="100%" style="stop-color:#0d47a1;stop-opacity:0.1"/>
            </radialGradient>
            <filter id="glow3" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        
        <!-- Title -->
        <text x="110" y="20" text-anchor="middle" fill="#4fc3f7" font-size="14" font-weight="bold">Bloch Sphere</text>
        
        <!-- Sphere outline -->
        <ellipse cx="100" cy="100" rx="70" ry="70" fill="url(#sphereGrad)" stroke="#4fc3f7" stroke-width="1.5" opacity="0.6"/>
        
        <!-- Equator ellipse -->
        <ellipse cx="100" cy="100" rx="70" ry="20" fill="none" stroke="#4fc3f7" stroke-width="0.5" stroke-dasharray="4,4" opacity="0.5"/>
        
        <!-- Meridian -->
        <ellipse cx="100" cy="100" rx="20" ry="70" fill="none" stroke="#4fc3f7" stroke-width="0.5" stroke-dasharray="4,4" opacity="0.5"/>
        
        <!-- Axes -->
        <line x1="100" y1="30" x2="100" y2="170" stroke="#81d4fa" stroke-width="1" opacity="0.6"/>
        <line x1="30" y1="100" x2="170" y2="100" stroke="#81d4fa" stroke-width="1" opacity="0.6"/>
        
        <!-- Axis labels -->
        <text x="100" y="22" text-anchor="middle" fill="#81d4fa" font-size="12" font-weight="bold">|0⟩</text>
        <text x="100" y="185" text-anchor="middle" fill="#81d4fa" font-size="12" font-weight="bold">|1⟩</text>
        <text x="180" y="104" text-anchor="start" fill="#81d4fa" font-size="10">+X</text>
        <text x="15" y="104" text-anchor="end" fill="#81d4fa" font-size="10">-X</text>
        
        <!-- State vector arrow -->
        <line x1="100" y1="100" x2="{proj_x}" y2="{proj_y}" stroke="#ff5722" stroke-width="3" filter="url(#glow3)"/>
        <circle cx="{proj_x}" cy="{proj_y}" r="6" fill="#ff5722" filter="url(#glow3)"/>
        
        <!-- State label -->
        <text x="{proj_x + 10}" y="{proj_y - 10}" fill="#ff5722" font-size="12" font-weight="bold">{label}</text>
        
        <!-- Info text -->
        <text x="110" y="220" text-anchor="middle" fill="#78909c" font-size="10">θ={theta:.2f}, φ={phi:.2f}</text>
    </svg>
    '''

def render_qsphere_svg(probabilities: dict, num_qubits: int = 2) -> str:
    """
    Render a Q-sphere visualization showing all basis states.
    Similar to IBM Quantum Composer's Q-sphere.
    """
    if not probabilities:
        probabilities = {"0" * num_qubits: 1.0}
    
    dim = 2 ** num_qubits
    cx, cy = 150, 150  # Center
    radius = 100
    
    svg = f'''
    <svg width="320" height="340" xmlns="http://www.w3.org/2000/svg" style="background: #1a1a2e; border-radius: 12px;">
        <defs>
            <radialGradient id="qsphereGrad" cx="30%" cy="30%">
                <stop offset="0%" style="stop-color:#4fc3f7;stop-opacity:0.2"/>
                <stop offset="100%" style="stop-color:#0d47a1;stop-opacity:0.05"/>
            </radialGradient>
            <filter id="glowQ" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
        </defs>
        
        <!-- Title -->
        <text x="160" y="25" text-anchor="middle" fill="#4fc3f7" font-size="14" font-weight="bold">Q-Sphere ({num_qubits} qubits)</text>
        
        <!-- Main sphere -->
        <circle cx="{cx}" cy="{cy}" r="{radius}" fill="url(#qsphereGrad)" stroke="#4fc3f7" stroke-width="1"/>
        
        <!-- Latitude lines -->
        <ellipse cx="{cx}" cy="{cy}" rx="{radius}" ry="30" fill="none" stroke="#4fc3f7" stroke-width="0.5" opacity="0.3"/>
        <ellipse cx="{cx}" cy="{cy-50}" rx="70" ry="20" fill="none" stroke="#4fc3f7" stroke-width="0.5" opacity="0.3"/>
        <ellipse cx="{cx}" cy="{cy+50}" rx="70" ry="20" fill="none" stroke="#4fc3f7" stroke-width="0.5" opacity="0.3"/>
    '''
    
    # Place states on the sphere based on Hamming weight
    max_prob = max(probabilities.values()) if probabilities else 1.0
    
    for i in range(dim):
        bitstring = format(i, f'0{num_qubits}b')
        prob = probabilities.get(bitstring, 0.0)
        
        if prob < 0.001:
            continue
        
        # Position based on Hamming weight (number of 1s)
        hamming = bitstring.count('1')
        layer_y = cy - radius + (2 * radius * hamming / num_qubits)
        
        # Spread states horizontally within each layer
        states_in_layer = math.comb(num_qubits, hamming)
        layer_radius = radius * math.sin(math.acos(1 - 2 * hamming / num_qubits)) if num_qubits > 0 else 0
        
        # Find position within layer
        layer_states = [format(j, f'0{num_qubits}b') for j in range(dim) if format(j, f'0{num_qubits}b').count('1') == hamming]
        idx = layer_states.index(bitstring)
        angle = 2 * math.pi * idx / len(layer_states) if len(layer_states) > 0 else 0
        
        state_x = cx + layer_radius * 0.7 * math.cos(angle)
        state_y = layer_y + layer_radius * 0.2 * math.sin(angle)
        
        # Size based on probability
        size = 5 + 15 * (prob / max_prob)
        
        # Color based on probability
        intensity = int(255 * (prob / max_prob))
        color = f"rgb({intensity}, {200}, {255})"
        
        svg += f'''
        <circle cx="{state_x}" cy="{state_y}" r="{size}" fill="{color}" opacity="0.8" filter="url(#glowQ)"/>
        <text x="{state_x}" y="{state_y + size + 12}" text-anchor="middle" fill="#b0bec5" font-size="9">|{bitstring}⟩</text>
        <text x="{state_x}" y="{state_y + 4}" text-anchor="middle" fill="#1a1a2e" font-size="8" font-weight="bold">{prob:.2f}</text>
        '''
    
    # North pole label
    svg += f'<text x="{cx}" y="50" text-anchor="middle" fill="#81d4fa" font-size="11">|{"0"*num_qubits}⟩</text>'
    # South pole label  
    svg += f'<text x="{cx}" y="270" text-anchor="middle" fill="#81d4fa" font-size="11">|{"1"*num_qubits}⟩</text>'
    
    # Legend
    svg += '''
        <rect x="10" y="290" width="300" height="40" fill="#21262d" rx="6"/>
        <text x="20" y="310" fill="#78909c" font-size="10">Size = Probability</text>
        <text x="20" y="325" fill="#78909c" font-size="10">Vertical = Hamming weight (# of 1s)</text>
    '''
    
    svg += '</svg>'
    return svg

def render_statevector_amplitudes(statevector_data: dict, num_qubits: int = 2) -> str:
    """
    Render statevector amplitudes as a visual table with phase information.
    """
    if not statevector_data:
        return "<p style='color: #78909c;'>No statevector data</p>"
    
    # Extract statevector
    sv = statevector_data.get("statevector", {})
    if not sv:
        return "<p style='color: #78909c;'>No statevector available</p>"
    
    real_parts = sv.get("real", [])
    imag_parts = sv.get("imag", [])
    
    if not real_parts:
        return "<p style='color: #78909c;'>Empty statevector</p>"
    
    dim = len(real_parts)
    
    html = '''
    <div style="background: #1a1a2e; border-radius: 12px; padding: 16px; max-height: 400px; overflow-y: auto;">
        <h4 style="color: #4fc3f7; margin: 0 0 12px 0; font-size: 14px;">📊 Statevector Amplitudes</h4>
        <table style="width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace;">
            <thead>
                <tr style="border-bottom: 1px solid #30363d;">
                    <th style="color: #81d4fa; padding: 8px; text-align: left; font-size: 12px;">State</th>
                    <th style="color: #81d4fa; padding: 8px; text-align: center; font-size: 12px;">Amplitude</th>
                    <th style="color: #81d4fa; padding: 8px; text-align: center; font-size: 12px;">Prob</th>
                    <th style="color: #81d4fa; padding: 8px; text-align: left; font-size: 12px;">Phase</th>
                </tr>
            </thead>
            <tbody>
    '''
    
    for i in range(min(dim, 32)):  # Limit to 32 states
        real = real_parts[i]
        imag = imag_parts[i] if i < len(imag_parts) else 0
        
        amplitude = complex(real, imag)
        prob = abs(amplitude) ** 2
        phase = cmath.phase(amplitude) if abs(amplitude) > 1e-10 else 0
        
        if prob < 0.0001:
            continue
        
        bitstring = format(i, f'0{num_qubits}b')
        
        # Phase visualization (small arc)
        phase_deg = math.degrees(phase)
        phase_color = "#4fc3f7" if phase >= 0 else "#ff5722"
        
        html += f'''
            <tr style="border-bottom: 1px solid #21262d;">
                <td style="color: #b0bec5; padding: 6px; font-size: 12px;">|{bitstring}⟩</td>
                <td style="color: #4fc3f7; padding: 6px; text-align: center; font-size: 11px;">{real:.3f}{'+' if imag >= 0 else ''}{imag:.3f}i</td>
                <td style="padding: 6px;">
                    <div style="display: flex; align-items: center; gap: 6px;">
                        <div style="flex: 1; height: 8px; background: #263238; border-radius: 4px; overflow: hidden;">
                            <div style="width: {prob*100}%; height: 100%; background: linear-gradient(90deg, #4fc3f7, #0d47a1);"></div>
                        </div>
                        <span style="color: #4fc3f7; font-size: 10px; min-width: 40px;">{prob:.3f}</span>
                    </div>
                </td>
                <td style="color: {phase_color}; padding: 6px; font-size: 11px;">{phase_deg:.1f}°</td>
            </tr>
        '''
    
    html += '''
            </tbody>
        </table>
    </div>
    '''
    return html

def render_visual_circuit(gates_json: str, num_qubits: int = 4) -> str:
    """
    Render an interactive visual circuit diagram like IBM Composer.
    Returns HTML/SVG for the circuit canvas.
    """
    try:
        gates = json.loads(gates_json) if gates_json and gates_json != "[]" else []
    except json.JSONDecodeError as e:
        # Return error SVG if gates JSON is invalid
        return f'<div style="color: #ef5350; padding: 10px;">Invalid gates JSON: {e}</div>'
    
    # SVG dimensions
    wire_spacing = 50
    gate_width = 40
    gate_spacing = 60
    left_margin = 80
    top_margin = 30
    canvas_width = max(600, left_margin + len(gates) * gate_spacing + 100)
    canvas_height = top_margin + num_qubits * wire_spacing + 50
    
    # Start SVG
    svg = f'''
    <svg width="{canvas_width}" height="{canvas_height}" xmlns="http://www.w3.org/2000/svg" 
         style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; font-family: 'IBM Plex Mono', 'Courier New', monospace;">
        <defs>
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            <linearGradient id="wireGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#4fc3f7;stop-opacity:0.3"/>
                <stop offset="50%" style="stop-color:#4fc3f7;stop-opacity:1"/>
                <stop offset="100%" style="stop-color:#4fc3f7;stop-opacity:0.3"/>
            </linearGradient>
        </defs>
    '''
    
    # Draw qubit labels and wires
    for i in range(num_qubits):
        y = top_margin + i * wire_spacing + 25
        # Qubit label with ket notation
        svg += f'''
        <text x="15" y="{y + 5}" fill="#81d4fa" font-size="14" font-weight="bold">|q{i}⟩</text>
        <text x="55" y="{y + 5}" fill="#546e7a" font-size="12">→</text>
        '''
        # Horizontal wire
        svg += f'''
        <line x1="{left_margin}" y1="{y}" x2="{canvas_width - 40}" y2="{y}" 
              stroke="url(#wireGrad)" stroke-width="2" stroke-linecap="round"/>
        '''
        # Measurement icon at end
        svg += f'''
        <rect x="{canvas_width - 55}" y="{y - 12}" width="24" height="24" rx="4" 
              fill="#263238" stroke="#546e7a" stroke-width="1"/>
        <path d="M{canvas_width - 49} {y + 6} L{canvas_width - 43} {y - 6} L{canvas_width - 37} {y + 6}" 
              fill="none" stroke="#90a4ae" stroke-width="1.5"/>
        '''
    
    # Draw gates
    for idx, gate in enumerate(gates):
        name = gate.get("name", "?")
        qubits = gate.get("qubits", [0])
        params = gate.get("params", [])
        gate_info = GATE_LIBRARY.get(name.lower(), {})
        
        x = left_margin + idx * gate_spacing + 20
        color = gate_info.get("color", "#607d8b")
        symbol = gate_info.get("symbol", name.upper())
        
        if len(qubits) == 1:
            # Single qubit gate
            y = top_margin + qubits[0] * wire_spacing + 25
            svg += f'''
            <g class="gate" style="cursor: pointer;" data-gate="{name}" data-idx="{idx}">
                <rect x="{x - gate_width//2}" y="{y - 18}" width="{gate_width}" height="36" rx="6" 
                      fill="{color}" stroke="{color}" stroke-width="2" filter="url(#glow)"
                      style="transition: all 0.2s ease;"/>
                <text x="{x}" y="{y + 5}" text-anchor="middle" fill="white" font-size="14" font-weight="bold">
                    {symbol}
                </text>
            </g>
            '''
            # Show parameter if exists
            if params:
                param_str = f"{params[0]:.2f}" if isinstance(params[0], float) else str(params[0])
                svg += f'''
                <text x="{x}" y="{y + 28}" text-anchor="middle" fill="#b0bec5" font-size="9">
                    θ={param_str}
                </text>
                '''
        
        elif len(qubits) >= 2:
            # Multi-qubit gate
            min_q = min(qubits)
            max_q = max(qubits)
            y1 = top_margin + min_q * wire_spacing + 25
            y2 = top_margin + max_q * wire_spacing + 25
            
            # Vertical connection line
            svg += f'''
            <line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" 
                  stroke="{color}" stroke-width="3" stroke-linecap="round"/>
            '''
            
            # Control dots and targets
            for i, q in enumerate(qubits):
                y = top_margin + q * wire_spacing + 25
                if name.lower() in ["cx", "ccx", "cy", "cz", "ch", "cswap"]:
                    if i < len(qubits) - 1:
                        # Control qubit - filled circle
                        svg += f'''
                        <circle cx="{x}" cy="{y}" r="8" fill="{color}" stroke="white" stroke-width="2"/>
                        '''
                    else:
                        # Target qubit
                        if name.lower() in ["cx", "ccx"]:
                            # X target (⊕)
                            svg += f'''
                            <circle cx="{x}" cy="{y}" r="16" fill="none" stroke="{color}" stroke-width="3"/>
                            <line x1="{x}" y1="{y - 16}" x2="{x}" y2="{y + 16}" stroke="{color}" stroke-width="3"/>
                            <line x1="{x - 16}" y1="{y}" x2="{x + 16}" y2="{y}" stroke="{color}" stroke-width="3"/>
                            '''
                        else:
                            # Other controlled gates
                            svg += f'''
                            <rect x="{x - 18}" y="{y - 18}" width="36" height="36" rx="6" 
                                  fill="{color}" stroke="white" stroke-width="2"/>
                            <text x="{x}" y="{y + 5}" text-anchor="middle" fill="white" font-size="12" font-weight="bold">
                                {symbol[-1] if len(symbol) > 1 else symbol}
                            </text>
                            '''
                elif name.lower() == "swap":
                    # SWAP gate (×)
                    svg += f'''
                    <line x1="{x - 10}" y1="{y - 10}" x2="{x + 10}" y2="{y + 10}" stroke="{color}" stroke-width="3"/>
                    <line x1="{x + 10}" y1="{y - 10}" x2="{x - 10}" y2="{y + 10}" stroke="{color}" stroke-width="3"/>
                    '''
    
    # Add "drop zone" indicator if no gates
    if not gates:
        center_y = top_margin + (num_qubits - 1) * wire_spacing // 2 + 25
        svg += f'''
        <rect x="{left_margin + 20}" y="{top_margin}" width="200" height="{num_qubits * wire_spacing}" 
              rx="8" fill="none" stroke="#546e7a" stroke-width="2" stroke-dasharray="8,4" opacity="0.5"/>
        <text x="{left_margin + 120}" y="{center_y}" text-anchor="middle" fill="#78909c" font-size="14">
            Click gates below to add
        </text>
        '''
    
    svg += '</svg>'
    return svg

def render_gate_palette() -> str:
    """Render the IBM Composer-style gate palette."""
    html = '''
    <style>
        .gate-palette { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 12px; 
            padding: 16px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 12px;
            margin: 10px 0;
        }
        .gate-category {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .category-label {
            font-size: 11px;
            color: #78909c;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }
        .gate-group {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .gate-btn {
            width: 44px;
            height: 44px;
            border-radius: 8px;
            border: 2px solid transparent;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'IBM Plex Mono', monospace;
            font-weight: 600;
            font-size: 14px;
            color: white;
            transition: all 0.2s ease;
            position: relative;
        }
        .gate-btn:hover {
            transform: scale(1.1);
            border-color: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .gate-btn::after {
            content: attr(data-name);
            position: absolute;
            bottom: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 9px;
            color: #90a4ae;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .gate-btn:hover::after {
            opacity: 1;
        }
    </style>
    <div class="gate-palette">
    '''
    
    # Group gates by category
    categories = {}
    for name, info in GATE_LIBRARY.items():
        cat = info.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))
    
    # Render each category
    for cat_key, cat_info in GATE_CATEGORIES.items():
        if cat_key in categories:
            html += f'''
            <div class="gate-category">
                <span class="category-label">{cat_info["name"]}</span>
                <div class="gate-group">
            '''
            for name, info in categories[cat_key]:
                html += f'''
                    <button class="gate-btn" 
                            id="gate-btn-{name}"
                            name="gate-{name}"
                            style="background: {info["color"]};"
                            data-gate="{name}"
                            data-name="{info["name"]}"
                            title="{info["name"]}: {info["formula"]}"
                            onclick="selectGate('{name}')">
                        {info["symbol"]}
                    </button>
                '''
            html += '</div></div>'
    
    html += '</div>'
    return html

def render_bloch_sphere_placeholder(statevector_json: str = "{}") -> str:
    """Render a Bloch sphere visualization placeholder."""
    return '''
    <div style="
        width: 200px; 
        height: 200px; 
        border-radius: 50%; 
        background: radial-gradient(circle at 30% 30%, #4fc3f7, #0d47a1);
        margin: 20px auto;
        box-shadow: inset -20px -20px 40px rgba(0,0,0,0.3), 0 4px 20px rgba(79, 195, 247, 0.3);
        position: relative;
    ">
        <div style="
            position: absolute;
            width: 100%;
            height: 100%;
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 50%;
        "></div>
        <div style="
            position: absolute;
            top: 50%;
            left: 50%;
            width: 80%;
            height: 2px;
            background: rgba(255,255,255,0.3);
            transform: translate(-50%, -50%);
        "></div>
        <div style="
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-weight: bold;
        ">|0⟩</div>
        <div style="
            position: absolute;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-weight: bold;
        ">|1⟩</div>
    </div>
    '''

def render_probability_bars(results: dict) -> str:
    """Render probability distribution as HTML bars."""
    counts = results.get("counts", {})
    probs = results.get("probabilities", {})
    
    if not probs:
        return "<p style='color: #78909c; text-align: center;'>No results available</p>"
    
    # Sort by bitstring
    sorted_keys = sorted(probs.keys())
    
    html = '<div style="padding: 10px;">'
    
    for key in sorted_keys:
        prob = probs[key]
        count = counts.get(key, 0)
        percent = prob * 100
        
        html += f'''
        <div style="margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px; color: #b0bec5; font-family: monospace;">
                <span>|{key}⟩</span>
                <span>{percent:.1f}% ({count})</span>
            </div>
            <div style="height: 24px; background: #263238; border-radius: 4px; overflow: hidden;">
                <div style="width: {percent}%; height: 100%; background: linear-gradient(90deg, #4fc3f7, #0d47a1); transition: width 0.5s ease;"></div>
            </div>
        </div>
        '''
    
    html += '</div>'
    return html
