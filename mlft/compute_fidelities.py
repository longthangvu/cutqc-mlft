#!/usr/bin/env python3
"""
Demo script for circuit cutting, fragment tomography, and maximum-likelihood corrections.

Author: Michael A. Perlin (github.com/perlinm)
"""
import collections
from time import perf_counter
from typing import Dict, Iterable, Tuple

import cirq
import numpy as np
import numpy.typing as npt

from mlft import circuit_ansatz
from mlft import cutting_methods as cm
from mlft import tomography as tm
from mlft import build_fragments as bf
from mlft import rebuild_fragments as rf
from mlft import mlft as mlft


def get_fidelity(
    approx_dist: Dict[Tuple[int, ...], float], exact_dist: npt.NDArray[np.float_]
) -> float:
    """
    Compute the fidelity between two classical probability distributions.

    The first distribution is represented by a dictionary mapping bitstrings to their probability of
    measurement, while the second distribution is represented by an array that is indexed by
    bitstrings.  Neither distribution is assumed to be normalized.
    """
    overlap = sum(np.sqrt(prob * exact_dist[bits]) for bits, prob in approx_dist.items())
    norms = sum(prob for prob in approx_dist.values()) * exact_dist.sum()
    ret = overlap**2 / norms
    # # print('old', ret)
    # bitstrings = [''.join(str(bit) for bit in key) for key in approx_dist.keys()]
    # values = list(approx_dist.values())

    # exact_dist = exact_dist.flatten()
    # # Sorting bitstrings and values together based on bitstrings
    # sorted_pairs = sorted(zip(bitstrings, values), key=lambda x: x[0])
    # _, sorted_values = zip(*sorted_pairs)
    # # Calculate the element-wise square root of the product of probabilities for corresponding bitstrings
    # overlap = np.sum(np.sqrt(sorted_values * exact_dist))
    
    # # Calculate the product of the sums (norms) of each distribution's probabilities
    # norms = np.sum(sorted_values) * np.sum(exact_dist)
    # print(overlap**2 / norms)
    return ret


def get_fidelities(
    circuit: cirq.Circuit,
    cuts: Iterable[Tuple[int, cirq.Qid]],
    repetitions: int,
    actual_probs=None,
    folder=None
) -> Tuple[float, float, float]:
    """
    Compute the fidelities of reconstructing the output of a circuit using
    1. full circuit execution,
    2. the original circuit cutting method in arXiv:1904.00102, and
    3. the maximum likelyhood fragment tomography method in arXiv:2005.12702.
    """
    qubit_order = sorted(circuit.all_qubits())
    num_qubits = len(qubit_order)

    # compute the actual probability distribution over measurement outcomes for the circuit
    if not actual_probs is not None:
        actual_probs = np.abs(cirq.final_state_vector(circuit, qubit_order=qubit_order)) ** 2
    actual_probs.shape = (2,) * num_qubits  # reshape into array indexed by bitstrings

    # get a probability distribution over measurement outcomes by sampling
    circuit_samples = np.random.choice(
        range(actual_probs.size), size=repetitions, p=actual_probs.ravel()
    )
    full_circuit_probs = {
        tuple(int(bit) for bit in bin(outcome)[2:].zfill(num_qubits)): counts / repetitions
        for outcome, counts in collections.Counter(circuit_samples).items()
    }
    exec_data = {}

    # cut the circuit, and perform fragment tomography to build fragment models
    cutter_begin = perf_counter()
    fragments = cm.cut_circuit(circuit, cuts)

    tomography_begin = perf_counter()
    exec_data['mlft_cutting_time'] = tomography_begin - cutter_begin
    print(f"Cutting took {exec_data['mlft_cutting_time']} seconds")
    tomo_data = tm.perform_fragment_tomography(fragments, repetitions=repetitions)

    build_begin = perf_counter()
    exec_data['mlft_evaluate'] = build_begin - tomography_begin
    print(f"Tomography took {exec_data['mlft_evaluate']} seconds")
    direct_models = bf.build_fragment_models(tomo_data)

    mlft_begin = perf_counter()
    exec_data['mlft_build'] = mlft_begin - build_begin
    print(f"Building full model took {exec_data['mlft_build']} seconds")
    likely_models = mlft.corrected_fragment_models(direct_models)

    exec_data['mlft_time'] = perf_counter() - mlft_begin
    print(f"MLFT correction took {exec_data['mlft_time']} seconds")
    # recombine fragments to infer the distribution over measurement outcomes for the full circuit
    direct_probs = rf.recombine_fragment_models(direct_models, qubit_order=qubit_order)
    likely_probs = rf.recombine_fragment_models(likely_models, qubit_order=qubit_order)

    # apply "naive" corrections to the "direct" probability distribution: throw out negative values
    direct_probs = {
        bistring: probability for bistring, probability in direct_probs.items() if probability >= 0
    }

    # print(full_circuit_probs)
    # plot_diagram(full_circuit_probs, './figs/full.png')
    # plot_diagram(direct_probs, './figs/direct.png')
    # plot_diagram(likely_probs, './figs/likely.png')

    plot_mlft(full_circuit_probs, direct_probs, likely_probs, folder)

    exec_data['full_fidelity'] = get_fidelity(full_circuit_probs, actual_probs)
    exec_data['direct_fidelity'] = get_fidelity(direct_probs, actual_probs)
    exec_data['likely_fidelity'] = get_fidelity(likely_probs, actual_probs)
    return exec_data

def plot_mlft(full_circuit_probs, direct_probs, likely_probs, folder):
    from helper_functions.plot_data import plot_data

    _, sorted_full_probs = convert_to_probability_list(full_circuit_probs)
    plot_data(sorted_full_probs, f'{folder}/figs/full.png')
    _, sorted_direct_probs = convert_to_probability_list(direct_probs)
    plot_data(sorted_direct_probs, f'{folder}/figs/direct.png')
    _, sorted_likely_probs = convert_to_probability_list(likely_probs)
    plot_data(sorted_likely_probs, f'{folder}/figs/likely.png')


import numpy as np
import matplotlib.pyplot as plt

def plot_diagram(data, figure_name):
    # Convert tuple keys to string keys for plotting
    bitstrings = [''.join(str(bit) for bit in key) for key in data.keys()]
    values = list(data.values())

    # Sorting bitstrings and values together based on bitstrings
    sorted_pairs = sorted(zip(bitstrings, values), key=lambda x: x[0])
    sorted_bitstrings, sorted_values = zip(*sorted_pairs)
    # print(sorted_values)
    # print(figure_name, sorted_values)
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.bar(sorted_bitstrings, sorted_values)
    plt.ylabel('Probability')
    plt.xticks(rotation=90)  # Rotate x-labels for better readability
    plt.tight_layout()
    plt.savefig(f'./{figure_name}')

def convert_to_probability_list(data):
    bitstrings = [''.join(str(bit) for bit in key) for key in data.keys()]
    values = list(data.values())

    # Sorting bitstrings and values together based on bitstrings
    sorted_pairs = sorted(zip(bitstrings, values), key=lambda x: x[0])
    sorted_bitstrings, sorted_values = zip(*sorted_pairs)

    return sorted_bitstrings, sorted_values

if __name__ == "__main__":
    num_qubits = 4
    num_clusters = 2
    repetitions = 10**6

    # construct and a random clustered circuit, and identify where it should be cut
    circuit, cuts = circuit_ansatz.random_clustered_circuit(num_qubits, num_clusters)
    print(circuit)
    print('cuts', cuts)

    # compute and print fidelities
    full_fidelity, direct_fidelity, likely_fidelity = get_fidelities(circuit, cuts, repetitions)
    print("full circuit fidelity:", full_fidelity)
    print("direct fidelity:", direct_fidelity)
    print("likely fidelity:", likely_fidelity)
