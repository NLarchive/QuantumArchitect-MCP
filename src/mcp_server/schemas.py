"""
MCP Server Schemas - JSON schemas defining input/output formats for circuits.
Ensures the Agent sends valid structured data.
"""

from typing import Any
from pydantic import BaseModel, Field


class GateSchema(BaseModel):
    """Schema for a quantum gate."""
    name: str = Field(..., description="Gate name (e.g., 'h', 'cx', 'rx')")
    qubits: list[int] = Field(..., description="Target qubit indices")
    params: list[float] = Field(default_factory=list, description="Gate parameters")
    classical_bits: list[int] = Field(default_factory=list, description="Classical bits for measurement")


class CircuitSchema(BaseModel):
    """Schema for a quantum circuit."""
    name: str = Field(default="", description="Circuit name")
    num_qubits: int = Field(..., ge=1, le=1000, description="Number of qubits")
    num_classical_bits: int = Field(default=0, ge=0, description="Number of classical bits")
    gates: list[GateSchema] = Field(default_factory=list, description="List of gates")
    parameters: list[str] = Field(default_factory=list, description="Parameter names for variational circuits")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format used by plugins."""
        return {
            "name": self.name,
            "num_qubits": self.num_qubits,
            "num_classical_bits": self.num_classical_bits,
            "gates": [g.model_dump() for g in self.gates],
            "parameters": self.parameters,
        }


class QASMInput(BaseModel):
    """Schema for QASM input."""
    qasm_string: str = Field(..., description="OpenQASM 2.0 or 3.0 code")


class HardwareTarget(BaseModel):
    """Schema for hardware target specification."""
    hardware_name: str = Field(..., description="Hardware profile name (e.g., 'ibm_brisbane')")


class SimulationRequest(BaseModel):
    """Schema for simulation request."""
    circuit: CircuitSchema
    shots: int = Field(default=1024, ge=1, le=1000000, description="Number of shots")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class ValidationResult(BaseModel):
    """Schema for validation result."""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""


class CreationRequest(BaseModel):
    """Schema for circuit creation request."""
    circuit_type: str = Field(..., description="Type of circuit to create")
    num_qubits: int = Field(default=2, ge=1, le=100)
    options: dict[str, Any] = Field(default_factory=dict, description="Additional options")


class ScoringResult(BaseModel):
    """Schema for scoring result."""
    score: float = Field(..., ge=0, le=10)
    grade: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    """Schema for validation response."""
    valid: bool = Field(..., description="Whether the circuit is valid")
    checks: dict[str, Any] = Field(default_factory=dict, description="Individual check results")
    errors: list[str] = Field(default_factory=list, description="List of validation errors")
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    circuit_info: dict[str, Any] = Field(default_factory=dict, description="Circuit metadata")


class SimulationResponse(BaseModel):
    """Schema for simulation response."""
    success: bool = Field(..., description="Whether simulation succeeded")
    num_qubits: int = Field(default=0, description="Number of qubits simulated")
    statevector: dict[str, Any] | None = Field(default=None, description="Statevector result")
    counts: dict[str, int] = Field(default_factory=dict, description="Measurement counts")
    probabilities: dict[str, float] = Field(default_factory=dict, description="State probabilities")
    noise_estimate: dict[str, Any] | None = Field(default=None, description="Noise estimation")
    estimated_fidelity: float | None = Field(default=None, description="Estimated fidelity")
    resources: dict[str, Any] = Field(default_factory=dict, description="Resource estimation")
    error: str | None = Field(default=None, description="Error message if failed")


class ScoreResponse(BaseModel):
    """Schema for scoring response."""
    success: bool = Field(..., description="Whether scoring succeeded")
    complexity: dict[str, Any] = Field(default_factory=dict, description="Complexity metrics")
    hardware_fitness: dict[str, Any] | None = Field(default=None, description="Hardware fitness score")
    estimated_fidelity: float | None = Field(default=None, description="Estimated fidelity")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall normalized score")
    error: str | None = Field(default=None, description="Error message if failed")


# MCP Tool definitions for Gradio
MCP_TOOLS = {
    "create_bell_state": {
        "description": "Create a Bell state (maximally entangled 2-qubit state)",
        "parameters": {
            "variant": {
                "type": "integer",
                "description": "Bell state variant (0-3): 0=Φ+, 1=Φ-, 2=Ψ+, 3=Ψ-",
                "default": 0,
            }
        },
    },
    "create_ghz_state": {
        "description": "Create a GHZ state (multi-qubit entangled state)",
        "parameters": {
            "num_qubits": {
                "type": "integer",
                "description": "Number of qubits (minimum 3)",
                "default": 3,
            }
        },
    },
    "create_qft": {
        "description": "Create a Quantum Fourier Transform circuit",
        "parameters": {
            "num_qubits": {
                "type": "integer",
                "description": "Number of qubits",
                "default": 3,
            }
        },
    },
    "create_grover": {
        "description": "Create Grover's search algorithm circuit",
        "parameters": {
            "num_qubits": {
                "type": "integer",
                "description": "Number of qubits",
                "default": 3,
            },
            "iterations": {
                "type": "integer",
                "description": "Number of Grover iterations (default: optimal)",
                "default": None,
            }
        },
    },
    "create_vqe_ansatz": {
        "description": "Create a VQE variational ansatz for quantum chemistry",
        "parameters": {
            "num_qubits": {
                "type": "integer",
                "description": "Number of qubits",
                "default": 4,
            },
            "layers": {
                "type": "integer",
                "description": "Number of variational layers",
                "default": 2,
            },
            "entanglement": {
                "type": "string",
                "description": "Entanglement pattern: linear, full, circular",
                "default": "linear",
            }
        },
    },
    "create_qaoa": {
        "description": "Create a QAOA circuit for optimization problems",
        "parameters": {
            "num_qubits": {
                "type": "integer",
                "description": "Number of qubits",
                "default": 4,
            },
            "p": {
                "type": "integer",
                "description": "Number of QAOA layers",
                "default": 1,
            }
        },
    },
    "validate_circuit": {
        "description": "Validate circuit syntax and structure",
        "parameters": {
            "circuit": {
                "type": "object",
                "description": "Circuit data as JSON",
            }
        },
    },
    "validate_connectivity": {
        "description": "Check if circuit is compatible with hardware topology",
        "parameters": {
            "circuit": {
                "type": "object",
                "description": "Circuit data as JSON",
            },
            "hardware": {
                "type": "string",
                "description": "Hardware name (e.g., ibm_brisbane)",
            }
        },
    },
    "simulate_circuit": {
        "description": "Simulate circuit and get statevector/probabilities",
        "parameters": {
            "circuit": {
                "type": "object",
                "description": "Circuit data as JSON",
            }
        },
    },
    "estimate_noise": {
        "description": "Estimate circuit noise and expected fidelity",
        "parameters": {
            "circuit": {
                "type": "object",
                "description": "Circuit data as JSON",
            }
        },
    },
    "score_complexity": {
        "description": "Calculate circuit complexity metrics",
        "parameters": {
            "circuit": {
                "type": "object",
                "description": "Circuit data as JSON",
            }
        },
    },
    "score_hardware_fitness": {
        "description": "Score how well circuit fits specific hardware",
        "parameters": {
            "circuit": {
                "type": "object",
                "description": "Circuit data as JSON",
            },
            "hardware": {
                "type": "string",
                "description": "Hardware name",
            }
        },
    },
    "get_available_hardware": {
        "description": "List available hardware profiles",
        "parameters": {},
    },
}
