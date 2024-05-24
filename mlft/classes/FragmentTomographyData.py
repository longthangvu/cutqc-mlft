from typing import (
    Dict,
    Iterator,
    Tuple,
)
from mlft.classes.Fragment import Fragment
from mlft.classes.const import PrepStates, MeasBases, BitString, PrepBasis

"""
Fragment tomomography data is collected into a dictionary with type signature
  'Dict[BitString, ConditionalFragmentData]',
where the BitString key is a measurement outcome at the circuit output of the fragment, and
  'ConditionalFragmentData = Dict[Tuple[PrepStates, MeasBases, BitString], float]'
maps a (prepared_state, measurement_basis, measurement_outcome) at the quantum inputs/outputs to a
probability (float) that...
(1) the previously specified 'BitString's at the circuit/quantum outputs are measured, when
(2) preparing the given prepared_state at the quantum input, and
(3) measuring in the given measurement_basis at the quantum output.
"""
ConditionalFragmentData = Dict[Tuple[PrepStates, MeasBases, BitString], float]
class FragmentTomographyData:
    """Data structure for storing data collected from fragment tomography."""

    def __init__(
        self,
        fragment: Fragment,
        tomography_data: Dict[BitString, ConditionalFragmentData],
        prep_basis: PrepBasis,
    ) -> None:
        self.fragment = fragment
        self.data = tomography_data
        self.prep_basis = prep_basis

    def substrings(self) -> Iterator[BitString]:
        """Iterate over all measurement outcomes at the circuit outputs of this fragment."""
        yield from self.data.keys()

    def condition_on(self, substring: BitString) -> ConditionalFragmentData:
        """
        Get the ConditionalFragmentData associated with a fixed measurement outcome at this
        fragment's circuit outputs.
        """
        return self.data[substring]