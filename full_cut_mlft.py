import random, os
import argparse

from helper_functions.benchmarks import generate_circ
from helper_functions.conversion import qiskit_to_cirq

# Function to check if file exists and is not empty
def file_exists_and_not_empty(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def export_data(data, fieldnames, file_path):
    import csv

    # Open the file in append mode ('a') so you don't overwrite existing data
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # If the file is empty, write the header
        if not file_exists_and_not_empty(file_path):
            writer.writeheader()
        # Append the new dictionary to the CSV
        writer.writerow(data)

def export_error(circuit_type, circuit_size, reason):
    export_data({'size': circuit_size, 'type': circuit_type, 'reason': reason},
                ['type', 'size', 'reason'], 
                'failed_circuit.csv')


def main(circuit_size, circuit_type, folder):
    seed = random.randint(1000,9999)
    gen_count = 0
    while True:
        if gen_count == 20: break
        try:
            qc = generate_circ(
                    num_qubits=circuit_size,
                    depth=1,
                    circuit_type=circuit_type,
                    reg_name="q",
                    connected_only=True,
                    seed=seed)
            
            if not qc:
                print("Can't create circuit")
                export_error(circuit_type, circuit_size, "Can't create circuit")
                raise ValueError(f"Can't create circuit: {circuit_type}_{circuit_size}")
            # print(qc.draw('text'))
            break
        except ValueError:
            seed = random.randint(1000, 9999)
            print(seed)
            gen_count = gen_count + 1
    
    try:
        qiskit_to_cirq(qc, None)
    except:
        print("Can't convert to cirq")
        export_error(circuit_type, circuit_size, f"Can't convert to cirq: {seed}")
        raise Exception(f"Can't convert to cirq: {circuit_type}_{circuit_size}_{seed}")

    import math
    from cutqc.main import CutQC
    cutqc = CutQC(
        name="test",
        circuit=qc,
        cutter_constraints={
            "max_subcircuit_width": math.ceil(qc.num_qubits / 4 * 3),
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3],
        },
        verbose=True,
    )

    cutqc.cut()
    if not cutqc.has_solution:
        export_error(circuit_type, circuit_size, "No viable cuts")
        raise Exception("The input circuit and constraints have no viable cuts")

    # CutQC 
    recursion_depth = 5
    # cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    # figure_name = f'./figs/cutqc_{recursion_depth}.png'
    cutqc.evaluate(eval_mode="qasm", num_shots_fn=None)
    figure_name = f'{folder}/figs/cutqc_qasm_{recursion_depth}.png'
    cutqc.build(mem_limit=32, recursion_depth=recursion_depth)
    print("Cut: %d recursions." % (cutqc.num_recursions))
    probabilities, ground_truth, _ = cutqc.verify()

    from helper_functions.plot_data import plot_data
    plot_data(probabilities, figure_name)
    plot_data(ground_truth, f'{folder}/figs/cutqc_groundtruth.png')

    # print(cutqc.times)
    cutqc.clean_data()

    print("Start MLFT")
    # MLFT


    from mlft.cutting_methods import cut_circuit
    from mlft.compute_fidelities import get_fidelities

    try:
        circuit, cuts = qiskit_to_cirq(qc, cutqc.positions)
    except:
        print("Can't convert positions to cirq")
        export_error(circuit_type, circuit_size, "Can't convert positions to cirq")
        raise Exception(f"Can't convert positions to cirq: {circuit_type}_{circuit_size}")

    # print("Circuit:")
    # print(circuit)

    repetitions = 10**6

    fragments = cut_circuit(circuit, cuts)
    [print(f'{fragment_name}:\n{fragments[fragment_name].circuit}') for fragment_name in fragments]

    # compute and print fidelities
    exec_data = get_fidelities(circuit, cuts, repetitions, actual_probs=ground_truth, folder=folder)
    print('*'*3,'Results','*'*3)
    print("Full circuit fidelity:", exec_data['full_fidelity'])
    print("Direct fidelity:", exec_data['direct_fidelity'])
    print("Likely fidelity:", exec_data['likely_fidelity'])

    exec_data['cutqc_fidelity'] = cutqc.fidelity
    exec_data['cutter'] = cutqc.times['cutter']
    exec_data['cutqc_evaluate'] = cutqc.times['evaluate']
    exec_data['cutqc_build'] = cutqc.times['build']

    exec_data['type'] = circuit_type
    exec_data['size'] = circuit_size
    exec_data['seed'] = seed
    # Specify the file path and the fieldnames (column headers)
    file_path = 'benchmark.csv'
    fieldnames = ['size', 'type', 'cutter', 'seed',
                 'mlft_cutting_time', 'mlft_evaluate', 'mlft_build', 'mlft_time', 
                 'cutqc_evaluate', 'cutqc_build', 
                 'full_fidelity', 'direct_fidelity', 'likely_fidelity', 'cutqc_fidelity']

    export_data(exec_data, fieldnames, file_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Process circuit information.")
    parser.add_argument('circuit_type', type=str, help='Type of the circuit')
    parser.add_argument('circuit_size', type=int, help='Size of the circuit')
    parser.add_argument('folder', type=str, help='Directory to store results')
    return vars(parser.parse_args())  # Return a dictionary

if __name__ == "__main__":
    args = parse_args()
    main(**args)