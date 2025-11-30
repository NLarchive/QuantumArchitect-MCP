"""
Gate Library - Definitions of standard quantum gates and their matrix representations.
"""

import numpy as np
from typing import Callable
import math


# Type alias for gate matrices
GateMatrix = np.ndarray


class GateLibrary:
    """
    Library of standard quantum gates with their matrix representations.
    Follows IBM Qiskit conventions for gate definitions.
    """
    
    # Pauli gates
    @staticmethod
    def I() -> GateMatrix:
        """Identity gate."""
        return np.array([[1, 0], [0, 1]], dtype=complex)
    
    @staticmethod
    def X() -> GateMatrix:
        """Pauli-X (NOT) gate."""
        return np.array([[0, 1], [1, 0]], dtype=complex)
    
    @staticmethod
    def Y() -> GateMatrix:
        """Pauli-Y gate."""
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    @staticmethod
    def Z() -> GateMatrix:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Hadamard and phase gates
    @staticmethod
    def H() -> GateMatrix:
        """Hadamard gate - creates superposition."""
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    
    @staticmethod
    def S() -> GateMatrix:
        """S gate (sqrt(Z))."""
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    
    @staticmethod
    def Sdg() -> GateMatrix:
        """S-dagger gate."""
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    
    @staticmethod
    def T() -> GateMatrix:
        """T gate (sqrt(S))."""
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    
    @staticmethod
    def Tdg() -> GateMatrix:
        """T-dagger gate."""
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
    
    @staticmethod
    def SX() -> GateMatrix:
        """Sqrt(X) gate."""
        return np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex) / 2
    
    # Rotation gates
    @staticmethod
    def Rx(theta: float) -> GateMatrix:
        """Rotation around X-axis."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    
    @staticmethod
    def Ry(theta: float) -> GateMatrix:
        """Rotation around Y-axis."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    @staticmethod
    def Rz(phi: float) -> GateMatrix:
        """Rotation around Z-axis."""
        return np.array([
            [np.exp(-1j * phi / 2), 0],
            [0, np.exp(1j * phi / 2)]
        ], dtype=complex)
    
    @staticmethod
    def P(phi: float) -> GateMatrix:
        """Phase gate."""
        return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    
    @staticmethod
    def U(theta: float, phi: float, lam: float) -> GateMatrix:
        """General single-qubit unitary U3 gate."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [c, -np.exp(1j * lam) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
        ], dtype=complex)
    
    # Two-qubit gates (4x4 matrices)
    @staticmethod
    def CX() -> GateMatrix:
        """Controlled-X (CNOT) gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
    
    @staticmethod
    def CY() -> GateMatrix:
        """Controlled-Y gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=complex)
    
    @staticmethod
    def CZ() -> GateMatrix:
        """Controlled-Z gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
    
    @staticmethod
    def SWAP() -> GateMatrix:
        """SWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @staticmethod
    def iSWAP() -> GateMatrix:
        """iSWAP gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
    
    @staticmethod
    def CP(phi: float) -> GateMatrix:
        """Controlled-Phase gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * phi)]
        ], dtype=complex)
    
    @staticmethod
    def CRx(theta: float) -> GateMatrix:
        """Controlled-Rx gate."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -1j*s],
            [0, 0, -1j*s, c]
        ], dtype=complex)
    
    @staticmethod
    def CRy(theta: float) -> GateMatrix:
        """Controlled-Ry gate."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, c, -s],
            [0, 0, s, c]
        ], dtype=complex)
    
    @staticmethod
    def CRz(theta: float) -> GateMatrix:
        """Controlled-Rz gate."""
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j * theta / 2), 0],
            [0, 0, 0, np.exp(1j * theta / 2)]
        ], dtype=complex)
    
    # Three-qubit gates (8x8 matrices)
    @staticmethod
    def CCX() -> GateMatrix:
        """Toffoli (CCX) gate."""
        mat = np.eye(8, dtype=complex)
        mat[6, 6] = 0
        mat[7, 7] = 0
        mat[6, 7] = 1
        mat[7, 6] = 1
        return mat
    
    @staticmethod
    def CSWAP() -> GateMatrix:
        """Fredkin (CSWAP) gate."""
        mat = np.eye(8, dtype=complex)
        mat[5, 5] = 0
        mat[6, 6] = 0
        mat[5, 6] = 1
        mat[6, 5] = 1
        return mat
    
    @classmethod
    def get_gate(cls, name: str, params: list[float] | None = None) -> GateMatrix:
        """Get gate matrix by name with optional parameters."""
        name = name.lower()
        params = params or []
        
        gate_map: dict[str, Callable[..., GateMatrix]] = {
            'id': cls.I,
            'i': cls.I,
            'x': cls.X,
            'y': cls.Y,
            'z': cls.Z,
            'h': cls.H,
            's': cls.S,
            'sdg': cls.Sdg,
            't': cls.T,
            'tdg': cls.Tdg,
            'sx': cls.SX,
            'rx': cls.Rx,
            'ry': cls.Ry,
            'rz': cls.Rz,
            'p': cls.P,
            'u': cls.U,
            'u3': cls.U,
            'cx': cls.CX,
            'cnot': cls.CX,
            'cy': cls.CY,
            'cz': cls.CZ,
            'swap': cls.SWAP,
            'iswap': cls.iSWAP,
            'cp': cls.CP,
            'crx': cls.CRx,
            'cry': cls.CRy,
            'crz': cls.CRz,
            'ccx': cls.CCX,
            'toffoli': cls.CCX,
            'cswap': cls.CSWAP,
            'fredkin': cls.CSWAP,
        }
        
        if name not in gate_map:
            raise ValueError(f"Unknown gate: {name}")
        
        gate_func = gate_map[name]
        
        # Check parameter count
        if name in ('rx', 'ry', 'rz', 'p', 'crx', 'cry', 'crz', 'cp'):
            if len(params) != 1:
                raise ValueError(f"Gate {name} requires 1 parameter")
            return gate_func(params[0])
        elif name in ('u', 'u3'):
            if len(params) != 3:
                raise ValueError(f"Gate {name} requires 3 parameters")
            return gate_func(*params)
        else:
            return gate_func()
    
    @classmethod
    def get_gate_info(cls, name: str) -> dict[str, any]:
        """Get information about a gate."""
        name = name.lower()
        
        single_gates = {'id', 'i', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx'}
        param_single_gates = {'rx', 'ry', 'rz', 'p'}
        u_gates = {'u', 'u3'}
        two_qubit_gates = {'cx', 'cnot', 'cy', 'cz', 'swap', 'iswap'}
        param_two_qubit_gates = {'cp', 'crx', 'cry', 'crz'}
        three_qubit_gates = {'ccx', 'toffoli', 'cswap', 'fredkin'}
        
        if name in single_gates:
            return {'qubits': 1, 'params': 0, 'type': 'single'}
        elif name in param_single_gates:
            return {'qubits': 1, 'params': 1, 'type': 'single_param'}
        elif name in u_gates:
            return {'qubits': 1, 'params': 3, 'type': 'single_param'}
        elif name in two_qubit_gates:
            return {'qubits': 2, 'params': 0, 'type': 'two'}
        elif name in param_two_qubit_gates:
            return {'qubits': 2, 'params': 1, 'type': 'two_param'}
        elif name in three_qubit_gates:
            return {'qubits': 3, 'params': 0, 'type': 'three'}
        else:
            raise ValueError(f"Unknown gate: {name}")
