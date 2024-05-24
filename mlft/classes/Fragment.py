from typing import (
    Mapping,
    Optional
)

import cirq

class Fragment:
    """
    Data structure for representing a fragment of a cut-up circuit.

    Fragments input/output qubits are partitioned into "circuit inputs", "circuit outputs",
    "quantum inputs", and "quantum outputs".
    - Circuit inputs/outputs = degrees of freedom at the beginning/end of the full (uncut) circuit.
      - Circuit inputs always start in |0>.
      - Circuit outputs are always measured in the computational basis.
    - Quantum inputs/outputs = degrees of freedom adjacent to a cut in the full circuit.
    Quantum inputs/outputs are specified by a dictionary that maps a qubit to a cut_name.
    """

    def __init__(
        self,
        circuit: cirq.AbstractCircuit,
        quantum_inputs: Optional[Mapping[cirq.Qid, str]] = None,
        quantum_outputs: Optional[Mapping[cirq.Qid, str]] = None,
    ) -> None:
        self.circuit = cirq.Circuit(circuit)
        self.quantum_inputs = quantum_inputs or {}
        self.quantum_outputs = quantum_outputs or {}
        assert all(qubit in circuit.all_qubits() for qubit in self.quantum_inputs)
        assert all(qubit in circuit.all_qubits() for qubit in self.quantum_outputs)
        self.circuit_outputs = sorted(circuit.all_qubits() - set(self.quantum_outputs))
