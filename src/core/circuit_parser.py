"""
Circuit Parser - Converts QASM strings and other formats into the internal DAG representation.
"""

import re
from typing import Any

from .dag_representation import CircuitDAG
from .exceptions import CircuitParseError, InvalidQASMError


class CircuitParser:
    """
    Parser for quantum circuit representations.
    Supports OpenQASM 2.0/3.0 and JSON circuit dictionaries.
    """
    
    # Standard gate patterns for QASM 2.0
    QASM2_GATE_PATTERN = re.compile(
        r'^(\w+)(?:\(([\d\.,\s\*\/\+\-\w]+)\))?\s+([\w\[\],\s]+);$'
    )
    
    # Qubit/register patterns
    QREG_PATTERN = re.compile(r'^qreg\s+(\w+)\[(\d+)\];$')
    CREG_PATTERN = re.compile(r'^creg\s+(\w+)\[(\d+)\];$')
    QUBIT_REF_PATTERN = re.compile(r'(\w+)\[(\d+)\]')
    
    # Single qubit gates (no parameters)
    SINGLE_GATES = {'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg', 'id', 'sx', 'sxdg'}
    
    # Parameterized single qubit gates
    PARAM_SINGLE_GATES = {'rx', 'ry', 'rz', 'p', 'u1', 'u2', 'u3', 'u'}
    
    # Two qubit gates
    TWO_QUBIT_GATES = {'cx', 'cz', 'cy', 'swap', 'iswap', 'cnot', 'cp', 'crx', 'cry', 'crz'}
    
    # Three qubit gates
    THREE_QUBIT_GATES = {'ccx', 'cswap', 'toffoli', 'fredkin'}
    
    @classmethod
    def parse_qasm(cls, qasm_string: str) -> CircuitDAG:
        """
        Parse OpenQASM 2.0/3.0 string into CircuitDAG.
        
        Args:
            qasm_string: Valid QASM code
            
        Returns:
            CircuitDAG representation
        """
        lines = qasm_string.strip().split('\n')
        
        # Track quantum and classical registers
        qregs: dict[str, tuple[int, int]] = {}  # name -> (start_index, size)
        cregs: dict[str, tuple[int, int]] = {}
        total_qubits = 0
        total_cbits = 0
        
        # First pass: find registers
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('OPENQASM') or line.startswith('include'):
                continue
            
            qreg_match = cls.QREG_PATTERN.match(line)
            if qreg_match:
                name, size = qreg_match.groups()
                qregs[name] = (total_qubits, int(size))
                total_qubits += int(size)
                continue
            
            creg_match = cls.CREG_PATTERN.match(line)
            if creg_match:
                name, size = creg_match.groups()
                cregs[name] = (total_cbits, int(size))
                total_cbits += int(size)
                continue
        
        if total_qubits == 0:
            raise InvalidQASMError("No quantum registers defined", qasm_string[:200])
        
        # Create DAG
        dag = CircuitDAG(total_qubits, total_cbits)
        
        # Second pass: parse gates
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('OPENQASM') or line.startswith('include'):
                continue
            if cls.QREG_PATTERN.match(line) or cls.CREG_PATTERN.match(line):
                continue
            
            # Parse gate
            try:
                cls._parse_gate_line(line, dag, qregs, cregs, line_num)
            except Exception as e:
                raise CircuitParseError(str(e), line_num)
        
        return dag
    
    @classmethod
    def _parse_gate_line(
        cls,
        line: str,
        dag: CircuitDAG,
        qregs: dict[str, tuple[int, int]],
        cregs: dict[str, tuple[int, int]],
        line_num: int
    ) -> None:
        """Parse a single gate line and add to DAG."""
        # Handle barrier
        if line.startswith('barrier'):
            qubit_str = line[7:].strip().rstrip(';')
            qubits = cls._resolve_qubits(qubit_str, qregs)
            dag.add_gate('barrier', qubits)
            return
        
        # Handle measurement
        if line.startswith('measure'):
            parts = line[7:].strip().rstrip(';').split('->')
            if len(parts) != 2:
                raise CircuitParseError(f"Invalid measure syntax: {line}", line_num)
            qubits = cls._resolve_qubits(parts[0].strip(), qregs)
            cbits = cls._resolve_cbits(parts[1].strip(), cregs)
            for q, c in zip(qubits, cbits):
                dag.add_gate('measure', [q], classical_bits=[c])
            return
        
        # Handle reset
        if line.startswith('reset'):
            qubit_str = line[5:].strip().rstrip(';')
            qubits = cls._resolve_qubits(qubit_str, qregs)
            for q in qubits:
                dag.add_gate('reset', [q])
            return
        
        # General gate pattern
        gate_match = cls.QASM2_GATE_PATTERN.match(line)
        if not gate_match:
            raise CircuitParseError(f"Cannot parse line: {line}", line_num)
        
        gate_name = gate_match.group(1).lower()
        params_str = gate_match.group(2)
        qubits_str = gate_match.group(3)
        
        # Parse parameters
        params: list[float] = []
        if params_str:
            try:
                params = [cls._eval_param(p.strip()) for p in params_str.split(',')]
            except Exception:
                raise CircuitParseError(f"Invalid parameters: {params_str}", line_num)
        
        # Parse qubits
        qubits = cls._resolve_qubits(qubits_str, qregs)
        
        # Validate gate
        if gate_name in cls.SINGLE_GATES:
            if len(qubits) != 1:
                raise CircuitParseError(
                    f"Gate {gate_name} expects 1 qubit, got {len(qubits)}", line_num
                )
        elif gate_name in cls.PARAM_SINGLE_GATES:
            if len(qubits) != 1:
                raise CircuitParseError(
                    f"Gate {gate_name} expects 1 qubit, got {len(qubits)}", line_num
                )
        elif gate_name in cls.TWO_QUBIT_GATES:
            if len(qubits) != 2:
                raise CircuitParseError(
                    f"Gate {gate_name} expects 2 qubits, got {len(qubits)}", line_num
                )
        elif gate_name in cls.THREE_QUBIT_GATES:
            if len(qubits) != 3:
                raise CircuitParseError(
                    f"Gate {gate_name} expects 3 qubits, got {len(qubits)}", line_num
                )
        
        dag.add_gate(gate_name, qubits, params)
    
    @classmethod
    def _resolve_qubits(cls, qubit_str: str, qregs: dict[str, tuple[int, int]]) -> list[int]:
        """Resolve qubit references to absolute indices."""
        qubits: list[int] = []
        for match in cls.QUBIT_REF_PATTERN.finditer(qubit_str):
            reg_name, idx = match.groups()
            if reg_name not in qregs:
                raise CircuitParseError(f"Unknown register: {reg_name}")
            start_idx, size = qregs[reg_name]
            qubit_idx = int(idx)
            if qubit_idx >= size:
                raise CircuitParseError(
                    f"Qubit index {qubit_idx} out of range for register {reg_name}[{size}]"
                )
            qubits.append(start_idx + qubit_idx)
        return qubits
    
    @classmethod
    def _resolve_cbits(cls, cbit_str: str, cregs: dict[str, tuple[int, int]]) -> list[int]:
        """Resolve classical bit references to absolute indices."""
        cbits: list[int] = []
        for match in cls.QUBIT_REF_PATTERN.finditer(cbit_str):
            reg_name, idx = match.groups()
            if reg_name not in cregs:
                raise CircuitParseError(f"Unknown classical register: {reg_name}")
            start_idx, size = cregs[reg_name]
            cbit_idx = int(idx)
            if cbit_idx >= size:
                raise CircuitParseError(
                    f"Classical bit index {cbit_idx} out of range for register {reg_name}[{size}]"
                )
            cbits.append(start_idx + cbit_idx)
        return cbits
    
    @classmethod
    def _eval_param(cls, param_str: str) -> float:
        """Safely evaluate a parameter expression."""
        import math
        # Allow pi and basic math
        safe_dict = {
            'pi': math.pi,
            'PI': math.pi,
            'e': math.e,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'exp': math.exp,
        }
        # Clean the expression
        param_str = param_str.replace('^', '**')
        return float(eval(param_str, {"__builtins__": {}}, safe_dict))
    
    @classmethod
    def parse_json_circuit(cls, circuit_dict: dict[str, Any]) -> CircuitDAG:
        """
        Parse a JSON circuit dictionary into CircuitDAG.
        
        Expected format:
        {
            "num_qubits": 2,
            "num_classical_bits": 2,
            "gates": [
                {"name": "h", "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
                {"name": "measure", "qubits": [0], "classical_bits": [0]}
            ]
        }
        """
        num_qubits = circuit_dict.get('num_qubits', 0)
        num_cbits = circuit_dict.get('num_classical_bits', 0)
        
        if num_qubits == 0:
            raise CircuitParseError("num_qubits must be specified and > 0")
        
        dag = CircuitDAG(num_qubits, num_cbits)
        
        for gate_dict in circuit_dict.get('gates', []):
            name = gate_dict.get('name', '').lower()
            qubits = gate_dict.get('qubits', [])
            params = gate_dict.get('params', [])
            classical_bits = gate_dict.get('classical_bits', [])
            
            if not name:
                raise CircuitParseError("Gate must have a 'name' field")
            if not qubits:
                raise CircuitParseError(f"Gate '{name}' must have 'qubits' field")
            
            dag.add_gate(name, qubits, params, classical_bits)
        
        return dag
