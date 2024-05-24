from typing import (
    Literal,
    Tuple
)

BitString = Tuple[int, ...]
PrepBasis = Literal["Pauli", "SIC"]
PrepState = Literal["Z+", "Z-", "X+", "X-", "Y+", "Y-", "S0", "S1", "S2", "S3"]
MeasBasis = Literal["Z", "X", "Y"]

PrepStates = Tuple[PrepState, ...]
MeasBases = Tuple[MeasBasis, ...]

DEFAULT_PREP_BASIS: PrepBasis = "SIC"
PAULI_OPS: MeasBases = ("Z", "X", "Y")