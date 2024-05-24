import collections
import itertools
from typing import (
    Dict,
    Optional,
)

import cirq
import numpy as np
from mlft.classes.Fragment import Fragment
from mlft.classes.FragmentTomographyData import FragmentTomographyData, ConditionalFragmentData
from mlft.classes.const import BitString, PrepBasis, DEFAULT_PREP_BASIS, PAULI_OPS
from mlft.helper_functions.prep_functions import get_prep_states, prep_state_ops, meas_basis_ops



####################################################################################################
# cutting a circuit into fragments


def perform_fragment_tomography(
    fragments: Dict[str, Fragment],
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
    repetitions: Optional[int] = None,
) -> Dict[str, FragmentTomographyData]:
    """Perform fragment tomography on a collection of fragments."""
    num_variants = sum(
        len(prep_basis) ** len(fragment.quantum_inputs)
        * len(PAULI_OPS) ** len(fragment.quantum_outputs)
        for fragment in fragments.values()
    )
    repetitions_per_variant = repetitions // num_variants
    return {
        fragment_key: perform_single_fragment_tomography(
            fragment, prep_basis, repetitions_per_variant
        )
        for fragment_key, fragment in fragments.items()
    }

def perform_single_fragment_tomography(
    fragment: Fragment,
    prep_basis: PrepBasis = DEFAULT_PREP_BASIS,
    repetitions_per_variant: Optional[int] = None,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> FragmentTomographyData:
    """
    Perform fragment tomography on the given fragment, using the specified tomographically complete
    basis of qubit states (which specifies which states to prepare at a fragment's quantum inputs).
    1. For a specified tomographically complete basis of qubit states, prepare all distinct
       combinations of quantum inputs.
    2. For each choice of prepared_state at the quantum inputs, measure quantum outputs in all
       distict combinations of Pauli bases.
    2. For each choice of (prepared_state, measurement_basis), compute a probability distribution
       over BitString measurement outcomes at the output qubits of the fragment.
    3. Split observed outcomes into the bitstrings at circuit outputs vs. quantum outputs, and
       organize all data into a FragmentTomographyData object.

    WARNING: this method makes no guarantee of efficiency.
    """
    quantum_inputs = list(fragment.quantum_inputs.keys())
    quantum_outputs = list(fragment.quantum_outputs.keys())
    circuit_outputs = fragment.circuit_outputs
    qubit_order = circuit_outputs + quantum_outputs
    num_qubits = len(qubit_order)
    if repetitions_per_variant:
        measurement = cirq.measure_each(*qubit_order)
        simulator = cirq.Simulator(seed=seed)
        repetitions_per_variant = max(10**4, 2**len(qubit_order))
        # repetitions_per_variant = 2**len(qubit_order)
    # print(repetitions_per_variant)
    tomography_data: Dict[BitString, ConditionalFragmentData] = collections.defaultdict(dict)
    # print("basis",prep_basis)
    # print('prep_st') #PRINT
    # combinations = itertools.product(get_prep_states(prep_basis), repeat=len(quantum_inputs))#PRINT
    # for combination in combinations:#PRINT
    #     print(combination)#PRINT
    # print('x'*5)#PRINT
    for prep_states in itertools.product(get_prep_states(prep_basis), repeat=len(quantum_inputs)):
        circuit_with_prep = prep_state_ops(prep_states, quantum_inputs) + fragment.circuit

        if not repetitions_per_variant:
            # construct the state at the end of the fragment when preparing the prep_states
            prep_fragment_state = cirq.final_state_vector(
                circuit_with_prep, qubit_order=qubit_order
            )
        # print('meas_bases') #PRINT
        # combinations = itertools.product(PAULI_OPS, repeat=len(quantum_outputs))#PRINT
        # for combination in combinations:#PRINT
        #     print(combination)#PRINT
        # print('x'*5)#PRINT
        for meas_bases in itertools.product(PAULI_OPS, repeat=len(quantum_outputs)):
            # construct sub-circuit to measure in the 'meas_bases'
            meas_ops = meas_basis_ops(meas_bases, quantum_outputs)

            if not repetitions_per_variant:
                # get exact probability distribution over measurement outcomes
                final_state = cirq.final_state_vector(
                    cirq.Circuit(meas_ops),
                    initial_state=prep_fragment_state,
                    qubit_order=qubit_order,
                )
                probabilities = np.reshape(abs(final_state) ** 2, (2,) * num_qubits)
                # print('prob', probabilities)#PRINT

                # collect exact probabilities into the tomography_data object
                for circuit_outcome in itertools.product([0, 1], repeat=len(circuit_outputs)):
                    for quantum_outcome in itertools.product([0, 1], repeat=len(quantum_outputs)):
                        conditions = (prep_states, meas_bases, quantum_outcome)
                        probability = probabilities[circuit_outcome + quantum_outcome]
                        tomography_data[circuit_outcome][conditions] = probability

            else:  # simulate sampling from the true probability distribution
                full_circuit = circuit_with_prep + meas_ops + measurement
                results = simulator.run(full_circuit, repetitions=repetitions_per_variant)
                outcome_counter = results.multi_measurement_histogram(keys=qubit_order)
                # print('oc', outcome_counter)#PRINT
                keys = list(itertools.product([0, 1], repeat=len(qubit_order)))
                print(full_circuit)
                prob_arr = np.array([outcome_counter[outcome] / repetitions_per_variant for outcome in keys])#PRINT
                print(prob_arr)

                # collect measurement outcomes into the tomography_data object
                for outcome, counts in outcome_counter.items():
                    circuit_outcome = outcome[: len(circuit_outputs)]
                    quantum_outcome = outcome[len(circuit_outputs) :]
                    # Record the fraction of times we observed this measurement outcome with the
                    # given prep_states/meas_bases.
                    conditions = (prep_states, meas_bases, quantum_outcome)
                    tomography_data[circuit_outcome][conditions] = counts / repetitions_per_variant
                    result_string = '-'.join(''.join(str(inner_tuple)) for inner_tuple in conditions)
                    # prob_arr
                    print(result_string)
                    # with open(f'./tmp/{result_string}.txt', 'w') as file:
                    #     write_data = ', '.join(f"{key}: {value}" for key, value in tomography_data.items())
                    #     # Write the text to the file
                    #     file.write(write_data)
                # print('-'*3)#PRINT
                # print(result_string)#PRINT
                # print(full_circuit)#PRINT
                # print('td', tomography_data)#PRINT
                # print('dict', extrac_prob(tomography_data))#PRINT
                
                # from helper_functions.plot_data import plot_data # DELETE
                # plot_data(prob_arr, f'./out/{result_string}.png', True) # DELETE
                
                # print('pb', prep_basis)#PRINT
                # print('fg', fragment)#PRINT
    # identify the cut indices at quantum inputs/outputs and return a FragmentTomographyData object
    return FragmentTomographyData(fragment, tomography_data, prep_basis)

def extrac_prob(d): # DELETE
    # Determine the length of the tuples (assuming all keys are the same length)
    tuple_length = len(next(iter(d)))
    # Generate all binary combinations for the given tuple length
    keys = list(itertools.product([0, 1], repeat=tuple_length))

    # Extract values in the order of the keys
    values = [list(d[key].values())[0] for key in keys]

    # Convert list to a numpy array if desired
    values_array = np.array(values)
    return values_array