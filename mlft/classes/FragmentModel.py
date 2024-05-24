from typing import (
    Dict,
    Iterator,
    Tuple,
)

import quimb.tensor as qtn # type: ignore
from mlft.classes.const import BitString
from mlft.classes.Fragment import Fragment

####################################################################################################
# building fragment models from fragment tomography data

class FragmentModel:
    """
    Data structure for representing a quantitative model of a fragment.

    Each fragment can be represented by a Choi matrix from the fragment's quantum inputs to its
    quantum outputs + circuit outputs.  This full Choi matrix is block diagonal, where each "block"
    is obtained by projecting onto a measurement outcome at the circuit outputs.  Model data is thus
    collected into a dictionary that maps a bitstring at the circuit outputs to a block of the Choi
    matrix.  Each block is, in turn, represented by a tensor whose indices are in one-to-one
    correspondence with the quantum inputs + quantum outputs of the fragment.
    """

    def __init__(self, fragment: Fragment, data: Dict[BitString, qtn.Tensor]) -> None:
        self.fragment = fragment
        self.data = data

    def substrings(self) -> Iterator[BitString]:
        """Iterate over measurement outcomes at the circuit outputs of this fragment."""
        yield from self.data.keys()

    def block(self, substring: BitString) -> qtn.Tensor:
        """Return the tensor in a single block of this fragment's Choi matrix."""
        return self.data[substring]

    def blocks(self) -> Iterator[Tuple[BitString, qtn.Tensor]]:
        """Iterate over all blocks of this fragment's Choi matrix."""
        yield from self.data.items()

    def num_blocks(self) -> int:
        return len(self.data)
