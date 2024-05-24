import itertools, copy, pickle, subprocess, psutil, os
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate

from helper_functions.non_ibmq_functions import find_process_jobs, scrambled

def get_num_workers(num_jobs, ram_required_per_worker):
    ram_avail = psutil.virtual_memory().available / 1024**3
    ram_avail = ram_avail / 4 * 3
    num_cpus = int(os.cpu_count() / 4 * 3)
    num_workers = int(min(ram_avail / ram_required_per_worker, num_jobs, num_cpus))
    return num_workers

def run_subcircuit_instances(
    subcircuits, subcircuit_instances, eval_mode, num_shots_fn, data_folder
):
    """
    subcircuit_instance_probs[subcircuit_idx][(instance_init,instance_meas)] = measured probability

    eval_mode:
    sv: statevector simulation
    qasm: noiseless qasm simulation
    runtime: for benchmarking, pseudo QPU backend generates uniform distribution
    """
    instance_init_meas_ids = {}
    for subcircuit_idx in subcircuit_instances:
        jobs = subcircuit_instances[subcircuit_idx]
        instance_init_meas_ids[subcircuit_idx] = {jobs[i]: i for i in range(len(jobs))}
        jobs = scrambled(jobs)
        pickle.dump(
            {
                "subcircuits": subcircuits,
                "eval_mode": eval_mode,
                "num_shots_fn": num_shots_fn,
                "instance_init_meas_ids": instance_init_meas_ids,
            },
            open("%s/meta_info.pckl" % data_folder, "wb"),
        )
        num_workers = get_num_workers(
            num_jobs=len(jobs),
            ram_required_per_worker=2 ** subcircuits[subcircuit_idx].num_qubits
            * 4
            / 1e9,
        )
        procs = []
        for rank in range(num_workers):
            rank_jobs = find_process_jobs(jobs=jobs, rank=rank, num_workers=num_workers)
            if len(rank_jobs) > 0:
                pickle.dump(
                    rank_jobs, open("%s/rank_%d.pckl" % (data_folder, rank), "wb")
                )
                python_command = (
                    "python -m cutqc.parallel_run_subcircuits --data_folder %s --subcircuit_idx %d --rank %d"
                    % (data_folder, subcircuit_idx, rank)
                )
                proc = subprocess.Popen(python_command.split(" "))
                procs.append(proc)
        [proc.wait() for proc in procs]

def attribute_shots(subcircuit_entries, subcircuits, eval_mode, data_folder):
    meta_info = pickle.load(open("%s/meta_info.pckl" % data_folder, "rb"))
    instance_init_meas_ids = meta_info["instance_init_meas_ids"]
    num_workers = 20
    entry_init_meas_ids = {}
    for subcircuit_idx in subcircuit_entries:
        entry_init_meas_ids[subcircuit_idx] = {}
        i = 0
        for key in subcircuit_entries[subcircuit_idx]:
            entry_init_meas_ids[subcircuit_idx][key] = i
            i += 1
        jobs = scrambled(list(subcircuit_entries[subcircuit_idx].keys()))
        # print('eas', entry_init_meas_ids)
        # print('ias', instance_init_meas_ids)
        pickle.dump(
            {
                "subcircuits": subcircuits,
                "eval_mode": eval_mode,
                "instance_init_meas_ids": instance_init_meas_ids,
                "entry_init_meas_ids": entry_init_meas_ids,
            },
            open("%s/meta_info.pckl" % data_folder, "wb"),
        )
        num_workers = get_num_workers(
            num_jobs=len(jobs),
            ram_required_per_worker=2 ** subcircuits[subcircuit_idx].num_qubits
            * 4
            / 1e9,
        )
        procs = []
        for rank in range(num_workers):
            rank_jobs = find_process_jobs(jobs=jobs, rank=rank, num_workers=num_workers)
            rank_jobs = {
                key: subcircuit_entries[subcircuit_idx][key] for key in rank_jobs
            }
            if len(rank_jobs) > 0:
                pickle.dump(
                    rank_jobs, open("%s/rank_%d.pckl" % (data_folder, rank), "wb")
                )
                python_command = (
                    "python -m cutqc.parallel_attribute_shots --data_folder %s --subcircuit_idx %d --rank %d"
                    % (data_folder, subcircuit_idx, rank)
                )
                proc = subprocess.Popen(python_command.split(" "))
                procs.append(proc)
        [proc.wait() for proc in procs]

def modify_subcircuit_instance(subcircuit, init, meas):
    """
    Modify the different init, meas for a given subcircuit
    Returns:
    Modified subcircuit_instance
    List of mutated measurements
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)
    return subcircuit_instance_circuit

def mutate_measurement_basis(meas):
    """
    I and Z measurement basis correspond to the same logical circuit
    """
    if all(x != "I" for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != "I":
                mutated_meas.append([x])
            else:
                mutated_meas.append(["I", "Z"])
        mutated_meas = list(itertools.product(*mutated_meas))
        return mutated_meas
    
def measure_state(full_state, meas):
    """
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1]):
        if meas_bit == "1" and meas_basis != "I" and meas_basis != "comp":
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))
    return sigma, effective_state

def measure_prob(unmeasured_prob, meas):
    if meas.count("comp") == len(meas) or type(unmeasured_prob) is float:
        return unmeasured_prob
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        # print('Measuring in',meas)
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            measured_prob[effective_state] += sigma * p
        return measured_prob