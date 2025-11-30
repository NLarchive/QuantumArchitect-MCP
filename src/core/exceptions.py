"""
Custom exceptions for the Quantum Circuit Engine.
Provides readable feedback to the Agent/User.
"""


class QuantumArchitectError(Exception):
    """Base exception for all QuantumArchitect errors."""
    pass


class QubitIndexError(QuantumArchitectError):
    """Raised when a qubit index is out of range."""
    def __init__(self, qubit_index: int, max_qubits: int):
        self.qubit_index = qubit_index
        self.max_qubits = max_qubits
        super().__init__(
            f"Qubit index {qubit_index} is out of range. "
            f"Circuit has {max_qubits} qubits (valid indices: 0-{max_qubits-1})."
        )


class GateNotSupportedError(QuantumArchitectError):
    """Raised when a gate is not supported by the target hardware."""
    def __init__(self, gate_name: str, hardware_name: str, supported_gates: list[str]):
        self.gate_name = gate_name
        self.hardware_name = hardware_name
        self.supported_gates = supported_gates
        super().__init__(
            f"Gate '{gate_name}' is not supported by {hardware_name}. "
            f"Supported gates: {', '.join(supported_gates)}."
        )


class ConnectivityError(QuantumArchitectError):
    """Raised when qubits are not connected on the hardware topology."""
    def __init__(self, qubit1: int, qubit2: int, hardware_name: str):
        self.qubit1 = qubit1
        self.qubit2 = qubit2
        self.hardware_name = hardware_name
        super().__init__(
            f"Qubits {qubit1} and {qubit2} are not connected on {hardware_name}. "
            f"Insert SWAP gates to route this operation."
        )


class CircuitParseError(QuantumArchitectError):
    """Raised when circuit parsing fails."""
    def __init__(self, message: str, line_number: int = None):
        self.line_number = line_number
        if line_number:
            super().__init__(f"Parse error at line {line_number}: {message}")
        else:
            super().__init__(f"Parse error: {message}")


class InvalidQASMError(QuantumArchitectError):
    """Raised when QASM code is invalid."""
    def __init__(self, message: str, qasm_snippet: str = None):
        self.qasm_snippet = qasm_snippet
        full_message = f"Invalid QASM: {message}"
        if qasm_snippet:
            full_message += f"\nProblematic code: {qasm_snippet[:100]}..."
        super().__init__(full_message)


class UnitarityError(QuantumArchitectError):
    """Raised when a circuit violates unitarity constraints."""
    def __init__(self, message: str):
        super().__init__(f"Unitarity violation: {message}")


class HardwareNotFoundError(QuantumArchitectError):
    """Raised when a hardware profile is not found."""
    def __init__(self, hardware_name: str, available: list[str]):
        self.hardware_name = hardware_name
        self.available = available
        super().__init__(
            f"Hardware profile '{hardware_name}' not found. "
            f"Available profiles: {', '.join(available)}."
        )


class SimulationError(QuantumArchitectError):
    """Raised when simulation fails."""
    def __init__(self, message: str):
        super().__init__(f"Simulation error: {message}")


class ValidationError(QuantumArchitectError):
    """Raised when validation fails."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed with {len(errors)} error(s): " + "; ".join(errors))
