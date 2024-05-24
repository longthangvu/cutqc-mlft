import cirq
import functools
import numpy as np
import numpy.typing as npt
from typing import Iterable, Iterator
from mlft.classes.const import MeasBases, PrepBasis, PrepStates, PrepState

@functools.lru_cache(maxsize=None)
def prep_state_to_proj(prep_state: PrepState) -> npt.NDArray[np.complex_]:
    """Convert a string into a projector onto the state of a qubit (flattened into a 1-D array)."""
    if prep_state == "Z+" or prep_state == "S0":
        vec = np.array([1, 0])
    elif prep_state == "Z-":
        vec = np.array([0, 1])
    elif prep_state == "X+":
        vec = np.array([1, 1]) / np.sqrt(2)
    elif prep_state == "X-":
        vec = np.array([1, -1]) / np.sqrt(2)
    elif prep_state == "Y+":
        vec = np.array([1, 1j]) / np.sqrt(2)
    elif prep_state == "Y-":
        vec = np.array([1, -1j]) / np.sqrt(2)
    elif prep_state in ["S1", "S2", "S3"]:
        corner_index = int(prep_state[1]) - 1
        azimuthal_angle = 2 * np.pi * corner_index / 3
        vec = np.array([1, np.exp(1j * azimuthal_angle) * np.sqrt(2)]) / np.sqrt(3)
    else:
        raise ValueError(f"state not recognized: {prep_state}")
    return np.outer(vec, vec.conj()).ravel()

def get_prep_states(prep_basis: PrepBasis) -> PrepStates:
    """
    Convert a string that specifies a tomographically complete basis of qubit states into a list of
    stings that specify the individual states.
    """
    states: PrepStates
    if prep_basis == "Pauli":
        states = ("Z+", "Z-", "X+", "X-", "Y+", "Y-")
    elif prep_basis == "SIC":
        states = ("S0", "S1", "S2", "S3")
    else:
        raise ValueError(f"tomographic basis not recognized: {prep_basis}")
    return states

def prep_state_ops(prep_states: PrepStates, qubits: Iterable[cirq.Qid]) -> Iterator[cirq.Operation]:
    """Return a circuit that prepares the given state on the given qubits (assumed to be in |0>)."""
    for prep_state, qubit in zip(prep_states, qubits):
        if prep_state == "Z+" or prep_state == "S0":
            continue
        elif prep_state == "Z-":
            yield cirq.X.on(qubit)
        elif prep_state == "X+":
            yield cirq.H.on(qubit)
        elif prep_state == "X-":
            yield cirq.X.on(qubit)
            yield cirq.H.on(qubit)
        elif prep_state == "Y+":
            yield cirq.H.on(qubit)
            yield cirq.S.on(qubit)
        elif prep_state == "Y-":
            yield cirq.H.on(qubit)
            yield cirq.inverse(cirq.S).on(qubit)
        elif prep_state in ["S1", "S2", "S3"]:
            polar_angle = 2 * np.arccos(1 / np.sqrt(3))  # cos(polar_angle/2) = 1/sqrt(3)
            yield cirq.ry(polar_angle).on(qubit)
            corner_index = int(prep_state[1]) - 1
            if corner_index != 0:
                azimuthal_angle = 2 * np.pi * corner_index / 3
                yield cirq.rz(azimuthal_angle).on(qubit)
        else:
            raise ValueError(f"state not recognized: {prep_state}")
        
def meas_basis_ops(meas_bases: MeasBases, qubits: Iterable[cirq.Qid]) -> Iterator[cirq.Operation]:
    """Return operations that map the given Pauli measurment basis onto the computational basis."""
    for basis, qubit in zip(meas_bases, qubits):
        if basis == "X":
            yield cirq.H.on(qubit)
        elif basis == "Y":
            yield cirq.inverse(cirq.S).on(qubit)
            yield cirq.H.on(qubit)