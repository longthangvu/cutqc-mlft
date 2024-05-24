import cirq
import functools
import operator
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)
import quimb.tensor as qtn # type: ignore


from mlft.classes.const import BitString
from mlft.classes.Fragment import Fragment
from mlft.classes.FragmentModel import FragmentModel

def get_contraction_path(fragment_models: Dict[str, FragmentModel]) -> List[Tuple[int, int]]:
    """Compute a tensor network contraction path for the given fragment models."""
    tensors = [next(iter(model.blocks()))[1] for model in fragment_models.values()]
    return qtn.TensorNetwork(tensors).contraction_path()


def get_outcome_combiner(
    fragments: Dict[str, Fragment],
    qubit_order: Optional[Sequence[cirq.Qid]] = None,
) -> Callable[[Dict[str, BitString]], BitString]:
    """
    Construct a function that that combines substrings at the circuit outputs of fragments.

    Args:
        - fragments: the fragments that need to be recombined, stored in a dictionary that maps a
          fragment_key to a Fragment.
        - qubit_order (optional): the order of the qubits in the reconstructed circuit.  If a qubit
          order is not provided, this defaults to qubit_order = sorted(reconstructed_circuit_qubits)

    Returns:
        - outcome_combiner: a function that recombines measurement outcomes at fragments into a
          measurment outcome for the full, uncut circuit.
            Args:
                - fragment_outcomes: a dictionary that maps a fragment_key to a measurement outcome
                  (BitString) at the circuit outputs of the corresponding fragment.
            Returns:
                - recombined_circuit_output: a recombined measurement outcome (BitString).
    """
    # collect some data about qubits and cuts
    circuit_qubits: List[cirq.Qid] = []  # all qubits addressed by the reconstructed circuit
    qubit_to_cut: Dict[cirq.Qid, str] = {}  # map from a qubit before a cut to the cut_name
    cut_to_qubit: Dict[str, cirq.Qid] = {}  # map from a cut_name to the qubit after the cut
    for fragment in fragments.values():
        circuit_qubits.extend(fragment.circuit.all_qubits() - set(fragment.quantum_inputs))
        qubit_to_cut.update(fragment.quantum_outputs)
        cut_to_qubit.update({cut: qubit for qubit, cut in fragment.quantum_inputs.items()})

    # construct a map that tracks where each qubit gets routed through the fragments
    initial_to_final_qubit_map = {}
    for circuit_qubit in circuit_qubits:
        initial_to_final_qubit_map[circuit_qubit] = circuit_qubit
        while (qubit := initial_to_final_qubit_map[circuit_qubit]) in qubit_to_cut:
            initial_to_final_qubit_map[circuit_qubit] = cut_to_qubit[qubit_to_cut[qubit]]

    # identify the order of the "final" qubits at the ends of fragments
    if qubit_order is None:
        qubit_order = sorted(circuit_qubits)
    final_qubit_order = [initial_to_final_qubit_map.get(qubit) or qubit for qubit in qubit_order]

    # Identify the permutation that needs to be applied to the concatenation of fragment substrings
    # in order to get the bits of the combined string in the right order.
    fragment_keys = list(fragments.keys())
    fragment_outputs = [fragments[key].circuit_outputs for key in fragment_keys]
    contacenated_qubits = functools.reduce(operator.add, fragment_outputs)
    bit_permutation = [contacenated_qubits.index(qubit) for qubit in final_qubit_order]
    print(bit_permutation)

    def outcome_combiner(fragment_substrings: Dict[str, BitString]) -> BitString:
        """
        Combine measurement outcomes at the circuit outputs of fragments into an overall
        measurement outcome for the full, uncut circuit.
        """
        substrings = [fragment_substrings[key] for key in fragment_keys]
        concatenated_substring = functools.reduce(operator.add, substrings)
        return tuple(concatenated_substring[index] for index in bit_permutation)

    return outcome_combiner
