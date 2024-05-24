import cirq
from typing import (
    Dict,
    Iterable,
    Tuple
)
from mlft.classes.Fragment import Fragment

def cut_circuit(
    circuit: cirq.AbstractCircuit, cuts: Iterable[Tuple[int, cirq.Qid]]
) -> Dict[str, Fragment]:
    """
    Cut a circuit into fragments.

    Strategy: rename qubits downstream of every cut in the circuit, then factorize circuit into
    separable subcircuits, and collect subcircuits and qubit routing data into Fragment objects.

    Args:
        - circuit: the circuit to be cut.
        - cuts: an iterable (e.g. tuple or list) cuts.  Each cut is specified by a
          (moment_index, qubit), such that all operations addressing the given qubit at or after the
          given moment_index are considered "downstream" of the cut.

    Returns:
        - fragments: dictionary that maps a fragment_key to a fragment.
    """
    circuit = cirq.Circuit(circuit)

    # keep track of all quantum inputs/outputs with dictionaries that map a qubit to a cut_name
    quantum_inputs: Dict[cirq.Qid, str] = {}
    quantum_outputs: Dict[cirq.Qid, str] = {}

    # map that keeps track of where qubits get routed through cuts
    initial_to_final_qubit_map: Dict[cirq.Qid, cirq.Qid] = {}

    cut_index = 0
    for cut_moment_index, cut_qubit in sorted(cuts):
        cut_name = f"cut_{cut_index}"

        # identify the qubits immediately before and after the cut
        old_qubit = initial_to_final_qubit_map.get(cut_qubit) or cut_qubit
        new_qubit = cirq.NamedQubit(cut_name)

        if old_qubit not in circuit[:cut_moment_index].all_qubits():
            # this is a "trivial" cut; there are no operations upstream of it
            continue

        # rename the old_qubit to the new_qubit in all operations downstream of the cut
        replacements = []  # ... to make to the circuit
        for moment_index, moment in enumerate(circuit[cut_moment_index:], start=cut_moment_index):
            for old_op in moment:
                if old_qubit in old_op.qubits:
                    new_op = old_op.transform_qubits({old_qubit: new_qubit})
                    replacements.append((moment_index, old_op, new_op))
                    break

        if not replacements:
            # this is a "trivial" cut; there are no operations downstream of it
            continue

        circuit.batch_replace(replacements)
        initial_to_final_qubit_map[cut_qubit] = new_qubit
        quantum_outputs[old_qubit] = quantum_inputs[new_qubit] = cut_name
        cut_index += 1

    # Factorize the circuit into independent subcircuits, and collect these subcircuits together
    # with their quantum inputs/outputs into Fragment objects.
    fragments = {}
    for fragment_index, subcircuit_moments in enumerate(circuit.factorize()):
        subcircuit = cirq.Circuit(subcircuit_moments)
        fragment_qubits = subcircuit.all_qubits()
        fragments[f"fragment_{fragment_index}"] = Fragment(
            subcircuit,
            {qubit: cut for qubit, cut in quantum_inputs.items() if qubit in fragment_qubits},
            {qubit: cut for qubit, cut in quantum_outputs.items() if qubit in fragment_qubits},
        )
    return fragments