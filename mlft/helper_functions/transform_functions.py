import functools
import itertools
import numpy as np
import numpy.typing as npt
# import quimb.tensor as qtn
from typing import Iterator, Tuple

from mlft.classes.const import BitString, PrepBasis, DEFAULT_PREP_BASIS, PrepStates, MeasBases, PAULI_OPS
# from mlft.classes.FragmentModel import FragmentModel
from mlft.helper_functions.prep_functions import prep_state_to_proj, get_prep_states

@functools.lru_cache(maxsize=None)
def interrogation_matrix(
    num_inputs: int,
    num_outputs: int,
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
) -> npt.NDArray[np.complex_]:
    """
    Construct a matrix whose rows are the operators (flattened to 1-D arrays) that are interrogated
    by fragment tomography.
    """
    condition_vecs = [
        conditions_to_vec(prep_states, meas_bases, meas_outcome)
        for prep_states, meas_bases, meas_outcome in condition_iterator(
            num_inputs, num_outputs, prep_basis
        )
    ]
    print('x',np.array(condition_vecs))
    return np.array(condition_vecs)

def conditions_to_vec(
    prep_states: PrepStates,
    meas_bases: MeasBases,
    meas_outcome: BitString,
) -> npt.NDArray[np.complex_]:
    """
    Convert a choice of (prepared_states, measurement_bases, measurement_outcome) at the quantum
    inputs/outputs of a fragment into the operator (flattened to a 1-D array) for a corresponding
    matrix element of that fragment.
    """
    out_strs = [basis + ("+" if bit == 0 else "-") for basis, bit in zip(meas_bases, meas_outcome)]
    out_vecs = [prep_state_to_proj(out_str) for out_str in out_strs]
    inp_vecs = [prep_state_to_proj(inp_str) for inp_str in prep_states]
    out_vec = functools.reduce(np.kron, out_vecs, np.array([1]))
    inp_vec = functools.reduce(np.kron, inp_vecs, np.array([1]))
    return np.kron(inp_vec.conj(), out_vec)


def condition_iterator(
    num_inputs: int,
    num_outputs: int,
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
) -> Iterator[Tuple[PrepStates, MeasBases, BitString]]:
    """
    Iterate over all choices of (prepared_states, measurement_bases, measurment_outcome) for the
    quantum inputs/outputs of a fragment with a given number of quantum inputs/outputs (and a given
    choice of tomographically complete basis of qubit states).
    """
    for prep_states in itertools.product(get_prep_states(prep_basis), repeat=num_inputs):
        for meas_bases in itertools.product(PAULI_OPS, repeat=num_outputs):
            for meas_outcome in itertools.product([0, 1], repeat=num_outputs):
                yield prep_states, meas_bases, meas_outcome