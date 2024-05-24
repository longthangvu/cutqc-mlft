import cirq
import itertools
from typing import (
    Dict,
    Optional,
    Sequence
)
import quimb.tensor as qtn # type: ignore

from mlft.helper_functions.post_process import get_contraction_path, get_outcome_combiner
from mlft.classes.const import BitString
from mlft.classes.FragmentModel import FragmentModel

def recombine_fragment_models(
    fragment_models: Dict[str, FragmentModel],
    qubit_order: Optional[Sequence[cirq.Qid]] = None,
) -> Dict[BitString, float]:
    """
    Recombine fragment models into a probability distribution over BitString measurement outcomes
    for the full, un-cut circuit.
    """
    recombined_distribution = {}

    fragments = {
        fragment_index: fragment_model.fragment
        for fragment_index, fragment_model in fragment_models.items()
    }
    contraction_path = get_contraction_path(fragment_models)
    outcome_combiner = get_outcome_combiner(fragments, qubit_order)

    # loop over all choices of bitstrings at the circuit outputs of fragments
    frag_keys = list(fragment_models.keys())
    frag_circuit_outputs = [
        list(frag_model.substrings()) for frag_model in fragment_models.values()
    ]
    print('md:', fragment_models['fragment_0'].__dict__)
    for fragment_substrings in itertools.product(*frag_circuit_outputs):
        # Combine measurement outcomes on fragments into a measurement outcome on
        # the full, uncut circuit.
        measurement_outcomes = dict(zip(frag_keys, fragment_substrings))
        combined_measurement_outcome = outcome_combiner(measurement_outcomes)
        print('_'*5)
        print(fragment_substrings)
        # print(measurement_outcomes)
        # print(combined_measurement_outcome)
        # collect all fragment tensors into a tensor network, and contract it
        tensors = [
            fragment_models[frag_key].block(substring)
            for frag_key, substring in measurement_outcomes.items()
        ]
        print('ts',tensors)
        network = qtn.TensorNetwork(tensors)
        print('nw',network)
        recombined_distribution[combined_measurement_outcome] = network.contract(
            optimize=contraction_path
        ).real
        print('rd',recombined_distribution[combined_measurement_outcome])

    return recombined_distribution