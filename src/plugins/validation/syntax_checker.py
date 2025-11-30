"""
Syntax Checker - Basic sanity checks for quantum circuits.
Validates gate names, qubit indices, and parameter counts.
"""

from typing import Any

from ...core.exceptions import (
    QubitIndexError,
    CircuitParseError,
    ValidationError,
)


# Valid gate definitions: name -> (num_qubits, num_params)
VALID_GATES: dict[str, tuple[int, int]] = {
    # Single qubit gates (no params)
    "id": (1, 0), "i": (1, 0),
    "x": (1, 0), "y": (1, 0), "z": (1, 0),
    "h": (1, 0),
    "s": (1, 0), "sdg": (1, 0),
    "t": (1, 0), "tdg": (1, 0),
    "sx": (1, 0), "sxdg": (1, 0),
    
    # Single qubit gates (1 param)
    "rx": (1, 1), "ry": (1, 1), "rz": (1, 1),
    "p": (1, 1), "u1": (1, 1),
    
    # Single qubit gates (2 params)
    "u2": (1, 2),
    
    # Single qubit gates (3 params)
    "u": (1, 3), "u3": (1, 3),
    
    # Two qubit gates (no params)
    "cx": (2, 0), "cnot": (2, 0),
    "cy": (2, 0), "cz": (2, 0),
    "swap": (2, 0), "iswap": (2, 0),
    "ch": (2, 0),
    
    # Two qubit gates (1 param)
    "cp": (2, 1), "crx": (2, 1), "cry": (2, 1), "crz": (2, 1),
    "rxx": (2, 1), "ryy": (2, 1), "rzz": (2, 1),
    
    # Three qubit gates
    "ccx": (3, 0), "toffoli": (3, 0),
    "cswap": (3, 0), "fredkin": (3, 0),
    
    # Special operations
    "measure": (1, 0),
    "reset": (1, 0),
    "barrier": (-1, 0),  # -1 means any number of qubits
}


def check_syntax(circuit_data: dict[str, Any]) -> dict[str, Any]:
    """
    Perform comprehensive syntax validation on a circuit.
    
    Args:
        circuit_data: Circuit dictionary with gates and metadata
    
    Returns:
        Validation result dictionary
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    num_qubits = circuit_data.get("num_qubits", 0)
    num_classical_bits = circuit_data.get("num_classical_bits", 0)
    gates = circuit_data.get("gates", [])
    
    # Check basic circuit structure
    if num_qubits <= 0:
        errors.append("Circuit must have at least 1 qubit")
    
    if not gates:
        warnings.append("Circuit has no gates")
    
    # Validate each gate
    for idx, gate in enumerate(gates):
        gate_errors = _validate_gate(gate, num_qubits, num_classical_bits, idx)
        errors.extend(gate_errors)
    
    # Check for potential issues
    circuit_warnings = _check_circuit_warnings(gates, num_qubits)
    warnings.extend(circuit_warnings)
    
    is_valid = len(errors) == 0
    
    return {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "gates_checked": len(gates),
        "summary": "Circuit syntax is valid" if is_valid else f"Found {len(errors)} syntax error(s)",
    }


def _validate_gate(
    gate: dict[str, Any],
    num_qubits: int,
    num_classical_bits: int,
    gate_idx: int
) -> list[str]:
    """Validate a single gate."""
    errors: list[str] = []
    
    name = gate.get("name", "").lower()
    qubits = gate.get("qubits", [])
    params = gate.get("params", [])
    classical_bits = gate.get("classical_bits", [])
    
    prefix = f"Gate {gate_idx} ({name})"
    
    # Check gate name
    if not name:
        errors.append(f"{prefix}: Missing gate name")
        return errors
    
    if name not in VALID_GATES:
        errors.append(f"{prefix}: Unknown gate '{name}'")
        return errors
    
    expected_qubits, expected_params = VALID_GATES[name]
    
    # Check qubit count
    if expected_qubits > 0 and len(qubits) != expected_qubits:
        errors.append(
            f"{prefix}: Expected {expected_qubits} qubit(s), got {len(qubits)}"
        )
    
    # Check qubit indices
    for q in qubits:
        if not isinstance(q, int):
            errors.append(f"{prefix}: Qubit index must be integer, got {type(q).__name__}")
        elif q < 0 or q >= num_qubits:
            errors.append(
                f"{prefix}: Qubit index {q} out of range (valid: 0-{num_qubits-1})"
            )
    
    # Check for duplicate qubits
    if len(qubits) != len(set(qubits)):
        errors.append(f"{prefix}: Duplicate qubit indices detected")
    
    # Check parameter count
    param_count = len([p for p in params if not (isinstance(p, str) and p.startswith("param:"))])
    if expected_params > 0 and param_count < expected_params:
        # Allow parameterized circuits with symbolic params
        if not any(isinstance(p, str) and p.startswith("param:") for p in params):
            errors.append(
                f"{prefix}: Expected {expected_params} parameter(s), got {param_count}"
            )
    
    # Check classical bits for measurements
    if name == "measure":
        if not classical_bits:
            errors.append(f"{prefix}: Measurement requires classical bit(s)")
        for c in classical_bits:
            if not isinstance(c, int):
                errors.append(f"{prefix}: Classical bit index must be integer")
            elif c < 0 or c >= num_classical_bits:
                errors.append(
                    f"{prefix}: Classical bit {c} out of range (valid: 0-{num_classical_bits-1})"
                )
    
    return errors


def _check_circuit_warnings(gates: list[dict[str, Any]], num_qubits: int) -> list[str]:
    """Check for potential issues in the circuit."""
    warnings: list[str] = []
    
    # Track which qubits have been used
    used_qubits: set[int] = set()
    measured_qubits: set[int] = set()
    gates_after_measure: dict[int, int] = {}
    
    for gate in gates:
        qubits = gate.get("qubits", [])
        name = gate.get("name", "").lower()
        
        for q in qubits:
            used_qubits.add(q)
            
            if q in measured_qubits and name not in ("measure", "reset", "barrier"):
                gates_after_measure[q] = gates_after_measure.get(q, 0) + 1
        
        if name == "measure":
            for q in qubits:
                measured_qubits.add(q)
    
    # Warn about unused qubits
    unused = set(range(num_qubits)) - used_qubits
    if unused:
        warnings.append(f"Unused qubits: {sorted(unused)}")
    
    # Warn about gates after measurement
    for q, count in gates_after_measure.items():
        warnings.append(
            f"Qubit {q} has {count} gate(s) after measurement (may collapse state)"
        )
    
    return warnings


def validate_qasm_syntax(qasm_string: str) -> dict[str, Any]:
    """
    Validate OpenQASM syntax without parsing into circuit.
    
    Args:
        qasm_string: QASM code string
    
    Returns:
        Validation result
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    lines = qasm_string.strip().split('\n')
    has_openqasm = False
    has_include = False
    has_qreg = False
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        if not line or line.startswith('//'):
            continue
        
        if line.startswith('OPENQASM'):
            has_openqasm = True
            if '2.0' not in line and '3.0' not in line:
                warnings.append(f"Line {line_num}: Unusual OPENQASM version")
        
        elif line.startswith('include'):
            has_include = True
        
        elif line.startswith('qreg'):
            has_qreg = True
            if '[' not in line or ']' not in line:
                errors.append(f"Line {line_num}: Invalid qreg syntax")
        
        elif line.startswith('creg'):
            if '[' not in line or ']' not in line:
                errors.append(f"Line {line_num}: Invalid creg syntax")
        
        elif not line.endswith(';'):
            if not line.startswith('gate ') and not line.startswith('{') and line != '}':
                errors.append(f"Line {line_num}: Statement should end with semicolon")
    
    if not has_openqasm:
        warnings.append("Missing OPENQASM header")
    
    if not has_qreg:
        errors.append("No quantum register defined")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "lines_checked": len(lines),
    }
