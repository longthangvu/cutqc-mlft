from typing import (
    Dict,
    Optional
)
import scipy
import quimb.tensor as qtn

from mlft.classes.const import BitString
from mlft.classes.FragmentTomographyData import FragmentTomographyData
from mlft.classes.FragmentModel import FragmentModel
from mlft.helper_functions.transform_functions import condition_iterator, interrogation_matrix

def build_fragment_models(
    fragment_tomography_data_dict: Dict[str, FragmentTomographyData],
    *,
    rank_cutoff: float = 1e-8,
) -> Dict[str, FragmentModel]:
    """Convert a collection of fragment tomography data into a collection of fragment models."""
    return {
        fragment_key: build_single_fragment_model(
            frag_tomo_data, fragment_key=fragment_key, rank_cutoff=rank_cutoff
        )
        for fragment_key, frag_tomo_data in fragment_tomography_data_dict.items()
    }


def build_single_fragment_model(
    fragment_tomography_data: FragmentTomographyData,
    *,
    fragment_key: Optional[str] = None,
    rank_cutoff: float = 1e-8,
) -> FragmentModel:
    """Convert fragment tomography data into a fragment model."""
    data = {
        substring: build_conditional_fragment_model(fragment_tomography_data, substring)
        for substring in fragment_tomography_data.substrings()
    }
    return FragmentModel(fragment_tomography_data.fragment, data)


def build_conditional_fragment_model(
    fragment_tomography_data: FragmentTomographyData,
    substring: BitString,
    *,
    fragment_key: Optional[str] = None,
    rank_cutoff: float = 1e-8,
) -> qtn.Tensor:
    """
    Build a reduced Choi matrix (as a qtn.Tensor) that represents a fragment after conditioning on
    a fixed BitString measurement outcome at the fragment's circuit output.

    Args:
        - fragment_tomgraphy_data: the data collected from fagment tomography.
        - substring: a BitString at the circuit outputs of the fragment.
        - fragment_key (optional): a hashable key to identify this fragment.
        - rank_cutoff (optional): see documentation for the 'cond' argument of scipy.linalg.lstsq:
          https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html

    Returns:
        tensor: a qtn.Tensor object representing a reduced Choi matrix for this fragment.
    """
    # Identify the fragment data conditioned on the given circuit output, as well as the total
    # number of quantum inputs/outputs, and the tomographically complete basis used to prepare
    # states at the quantum inputs.
    conditional_fragment_data = fragment_tomography_data.condition_on(substring)
    prep_basis = fragment_tomography_data.prep_basis
    num_inputs = len(fragment_tomography_data.fragment.quantum_inputs)
    num_outputs = len(fragment_tomography_data.fragment.quantum_outputs)

    # Compute the reduced Choi matrix by least squares fitting to tomography data, by solving
    # a linear system of equations, 'A x = b', where
    # - 'x' is the (vectorized) Choi matrix,
    # - 'b' is a vector of probabilities for different measurement outcomes, and
    # - 'A' is a matrix whose rows are the (vectorized) operators whose expectation values (with
    #   respect to the Choi matrix 'x') are the probabilities in 'b'.
    interrogation_operators = interrogation_matrix(num_inputs, num_outputs, prep_basis)  # 'A'
    interrogation_outcomes = [  # 'b'
        conditional_fragment_data.get(condition) or 0
        for condition in condition_iterator(num_inputs, num_outputs, prep_basis)
    ]
    reduced_choi_matrix = scipy.linalg.lstsq(
        interrogation_operators,
        interrogation_outcomes,
        cond=rank_cutoff,
    )[0]

    # Factorize the reduced Choi matrix into tensor factors associated with individual qubit degrees
    # of freedom, and return as a qtn.Tensor object in which each tensor factor (index) is labeled
    # by a corresponding cut_name.
    choi_tensor = reduced_choi_matrix.reshape((4,) * (num_inputs + num_outputs))
    cuts_at_inputs = list(fragment_tomography_data.fragment.quantum_inputs.values())
    cuts_at_outputs = list(fragment_tomography_data.fragment.quantum_outputs.values())
    cut_indices = cuts_at_inputs + cuts_at_outputs
    tags = (fragment_key,) if fragment_key is not None else None
    return qtn.Tensor(choi_tensor, inds=cut_indices, tags=tags)
