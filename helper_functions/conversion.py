import cirq, qiskit
from typing import List, Tuple

# Taken from qusetta (https://github.com/qcware/qusetta)
# Translating quantum circuits
# Added position conversion

PARAMETER_FREE_GATES = frozenset({
    "I", "H", "X", "Y", "Z", "S", "T", "CX", "CZ", "SWAP", "CCX"
})

PARAMETER_GATES = frozenset({'RX', 'RY', 'RZ'})
MAPPING = {"RX": "rx", "RY": "ry", "RZ": "rz"}

def qiskit_to_cirq(circuit: qiskit.QuantumCircuit, position) -> cirq.Circuit:
    qusetta_circ, qusetta_pos = qiskit_to_qusetta(circuit, position)
    return cirq_from_qusetta(qusetta_circ, qusetta_pos)

def qiskit_to_qusetta(circuit: qiskit.QuantumCircuit, position) -> List[str]:
        """Convert a qiskit circuit to a qusetta circuit.

        Parameters
        ----------
        circuit : qiskit.QuantumCircuit object.

        Returns
        -------
        qs_circuit : list of strings.
            See ``help(qusetta)`` for more details on how the list of
            strings should be formatted.

        """
        qs_circuit = []
        for gate, qubits, _ in circuit:  # _ refers to classical bits
            g = gate.name.upper()

            if g == "MEASURE":  # ignore measure gates
                continue
            elif g == "ID":
                g = "I"
            elif g == "U1":
                g = "RZ"  # same up to a phase factor
            elif g == "U2":
                # see below for why we reverse the qubits
                r = circuit.num_qubits - qubits[0].index - 1
                qs_circuit.extend([
                    "RZ(%g - PI/2)(%d)" % (gate.params[1], r),
                    "RX(PI/2)(%d)" % r,
                    "RZ(%g + PI/2)(%d)" % (gate.params[0], r)
                ])
                continue
            elif g == "U3":
                # see below for why we reverse the qubits
                r = circuit.num_qubits - qubits[0].index - 1
                qs_circuit.extend([
                    "RZ(%g - PI/2)(%d)" % (gate.params[2], r),
                    "RX(%g)(%d)" % (gate.params[0], r),
                    "RZ(%g + PI/2)(%d)" % (gate.params[1], r)
                ])
                continue

            if gate.params:
                g += "(" + ", ".join(str(x) for x in gate.params) + ")"
            # ibm is weird and reversed their qubits from everyone else.
            # So we reverse them here.
            g += "(" + ", ".join(
                str(circuit.num_qubits - q.index - 1)
                for q in qubits
            ) + ")"
            qs_circuit.append(g)
        qs_position = {}
        if position:
            for wire, gate in position:
                wire = str(circuit.num_qubits - wire.index - 1)
                name, qubits, occurrence = gate
                name += "(" + ", ".join(
                    str(circuit.num_qubits - q.index - 1)
                    for q in qubits
                ) + ")"
                qs_position[name] = (occurrence, wire)
        return qs_circuit, qs_position

def cirq_from_qusetta(circuit: List[str], position) -> cirq.Circuit:
        """Convert a qusetta circuit to a cirq circuit.

        Parameters
        ----------
        circuit : list of strings.
            See ``help(qusetta)`` for more details on how the list of
            strings should be formatted.

        Returns
        -------
        cirq_circuit : cirq.Circuit.

        """
        cirq_circuit = cirq.Circuit()
        # print(position)
        gate_list = list(position.keys())
        # print(gate_list)
        mlft_cuts = []
        gate_occurrence_map = {}
        for gate in circuit:
            g, params, qubits = gate_info(gate)
            gate_name = g + str(qubits)
            qubits = [cirq.LineQubit(x) for x in qubits]
            cirq_gate = getattr(cirq, MAPPING.get(g, g))
            if params:
                cirq_gate = cirq_gate(*params)
            cirq_circuit.append(cirq_gate(*qubits))
            if len(qubits) > 1:
                if gate_name in gate_occurrence_map:
                    gate_occurrence_map[gate_name] += 1
                else: gate_occurrence_map[gate_name] = 1
                # if gate_name in gate_list and gate_occurrence_map[gate_name] == position[gate_name]:
                if gate_name in gate_list:
                    occurrence, qubit = position[gate_name]
                    if gate_occurrence_map[gate_name] == occurrence:
                        # print(gate_name, int(qubit))
                        cut = (len(cirq_circuit), cirq.LineQubit(int(qubit)))
                        mlft_cuts.append(cut)
        return cirq_circuit, mlft_cuts

def gate_info(gate: str) -> Tuple[str, Tuple[float, ...], Tuple[int, ...]]:
    """Get the gate info from a string gate.

    Parameters
    ----------
    gate : str.
        See ``help(qusetta)`` for how the gate should be specifed. As an
        example, a gate could be ``H(0)`` or ``RX(PI/2)(1)``.

    Returns
    -------
    res : tuple (str, tuple of floats, tuple of ints).
        The first element is the gate name, the second is the
        parameters (often empty), and the third is the qubits.

    """
    i = gate.index('(')
    g = gate[:i].strip().upper()
    gate = gate[i+1:]
    if g in PARAMETER_GATES:
        j = gate.index(")")
        params = tuple(float(eval(x)) for x in gate[:j].split(','))
        i = gate.index('(')
        gate = gate[i+1:]
        j = gate.index(')')
        qubits = tuple(int(x) for x in gate[:j].split(','))
    elif g in PARAMETER_FREE_GATES:
        j = gate.index(")")
        qubits = tuple(int(x) for x in gate[:j].split(','))
        params = tuple()
    else:
        raise NotImplementedError("%s is not recognized" % g)

    return g, params, qubits