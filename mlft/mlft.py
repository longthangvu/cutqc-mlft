import numpy as np
import numpy.typing as npt
from typing import Dict
import quimb.tensor as qtn
from mlft.classes.FragmentModel import FragmentModel

####################################################################################################
# applying maximum-likelihood corrections to fragment models


def corrected_fragment_models(
    fragment_models: Dict[str, FragmentModel]
) -> Dict[str, FragmentModel]:
    """
    Apply maximum-likelihood corrections to a collection of fragment models.

    The general strategy, taken from arXiv:1106.5458, is to diagonalize the Choi matrix for each
    fragment, and eliminate negative eigenvalues one by one in order of decreasing magnitude.  Every
    time a negative eigenvalue is eliminated, its value is distributed evenly among all other
    eigenvalues, such that the trace of the Choi matrix is preserved.  This process is repeated
    until the Choi matrix has no more negative eigenvalues.

    TODO: find the maximum-likelihood model for an isometric channel.
    """
    return {
        fragment_key: corrected_single_fragment_model(fragment_model)
        for fragment_key, fragment_model in fragment_models.items()
    }

def corrected_single_fragment_model(fragment_model: FragmentModel) -> FragmentModel:
    """Apply maximum-likelihood corrections to a fragment model."""
    num_inputs = len(fragment_model.fragment.quantum_inputs)
    num_outputs = len(fragment_model.fragment.quantum_outputs)
    num_qubits = num_inputs + num_outputs

    # compute all eigenvalues and eigenvectors of the Choi matrix for this fragment
    eigenvalues = np.empty((fragment_model.num_blocks(), 2**num_qubits))
    eigenvectors = np.empty(
        (fragment_model.num_blocks(), 2**num_qubits, 2**num_qubits), dtype=complex
    )
    for idx, (circuit_outcome, block_tensor) in enumerate(fragment_model.blocks()):
        # construct the Choi matrix for this block
        block_choi_matrix = np.moveaxis(
            block_tensor.data.reshape((2,) * (num_qubits * 2)),
            range(1, 2 * num_qubits, 2),
            range(num_qubits, 2 * num_qubits),
        ).reshape((2**num_qubits,) * 2)
        block_eig_vals, block_eig_vecs = np.linalg.eigh(block_choi_matrix)
        eigenvalues[idx] = block_eig_vals
        eigenvectors[idx] = block_eig_vecs
    eigenvalues = correct_probability_distribution(eigenvalues)

    # Iterate over all blocks of the Choi matrix, and reconstruct them from the corrected
    # eigenvalues and their corresponding eigenvectors.
    corrected_data = {}
    for (circuit_outcome, block_tensor), block_eig_vals, block_eig_vecs in zip(
        fragment_model.blocks(), eigenvalues, eigenvectors
    ):
        corrected_block_choi_matrix = sum(
            val * np.outer(vec, vec.conj()) for val, vec in zip(block_eig_vals, block_eig_vecs.T)
        )
        # convert the corrected Choi matrix back into a tensor
        corrected_block_tensor_data = np.moveaxis(
            corrected_block_choi_matrix.reshape((2,) * (num_qubits * 2)),
            range(num_qubits, 2 * num_qubits),
            range(1, 2 * num_qubits, 2),
        ).reshape((4,) * num_qubits)
        corrected_data[circuit_outcome] = qtn.Tensor(
            corrected_block_tensor_data, inds=block_tensor.inds, tags=block_tensor.tags
        )
    print(fragment_model.fragment.circuit)
    print(corrected_block_tensor_data)
    return FragmentModel(fragment_model.fragment, corrected_data)

def correct_probability_distribution(probabilities: npt.NDArray[float]) -> npt.NDArray[float]:
    """Apply maximum-likelihood corrections to a classical probability distribution.

    Eliminate negative probabilities one by one in order of decreasing magnitude, and distribute
    their values among all other probabilities.  Method taken from arXiv:1106.5458.
    """
    prob_order = np.argsort(probabilities.ravel())
    sorted_probabilities = probabilities.ravel()[prob_order]
    for idx, val in enumerate(sorted_probabilities):
        if val >= 0:
            break
        sorted_probabilities[idx] = 0
        num_vals_remaining = probabilities.size - idx - 1
        sorted_probabilities[idx + 1 :] += val / num_vals_remaining
    inverse_sort = np.arange(probabilities.size)[np.argsort(prob_order)]
    corrected_probabilities = sorted_probabilities[inverse_sort]
    corrected_probabilities.shape = probabilities.shape
    return corrected_probabilities