# Utility methods for (Hybrid) VQC training.

import os
from typing import List
from matplotlib import backend_bases
import pennylane as pnl
import pennylane_qiskit

from .terminal_colors import tcols


def create_output_folder(output_folder):
    """
    Creates output folder for the qsvm.
    @output_folder :: Name of the output folder for this particular
                      version of the qsvm.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def print_model_info(ae_path: str, qdata, vqc):
    """
    Print information about the model.
    @ae_path :: String of path to the autoencoder.
    @qdata   :: The data object used to train the qsvm.
    @vqc     :: The qiskit vqc object.
    """
    print("\n-------------------------------------------")
    print(f"Autoencoder model: {ae_path}")
    print(f"Data path: {qdata.ae_data.data_folder}")
    print(
        f"ntrain = {len(qdata.ae_data.trtarget)}, "
        f"nvalid = {len(qdata.ae_data.vatarget)}, "
        f"ntest  = {len(qdata.ae_data.tetarget)}, "
    )
    print("-------------------------------------------\n")

    print(tcols.OKCYAN + "The VQC circuit about to be trained." + tcols.ENDC)
    vqc.draw()

def get_private_config(pconfig_path: str) -> dict:
    """Import the private configuration file. This is necessary for running on IBM
    quatum computers and contains the access token. A template of this file can be
    found in the bin directory, where you can input your details.

    Args:
        pconfig_path: The path to the private configuration file.

    Returns:
        The private configuration dictionary loaded from the file found at the
        pconfig_path.
    """
    try:
        with open(pconfig_path) as pconfig:
            private_config = json.load(pconfig)
    except OSError:
        FileNotFoundError("Error in reading private config: process aborted.")

    return private_config

def config_ideal(name: str, shots=None: int) -> dict:
    """The configuration loading of the ideal simulation.

    Args:
        name: The name of the simulator to be used in simulating the quantum circuit.
        shots: The number of shots to be used in the simulation.

    Returns:
        Dictionary of the configuration for the ideal simulation of the quantum circuit.
    """
    config_ideal = {"name": name, "shots": shots}

    return config_ideal

def config_noisy(shots: int, optimization_level: int, transpiler_seed: int,
                 initial_layout: List[int], seed_simulator: int, private_config: dict)
-> dict:
    """The configuration loading for the noisy simulation.

    Args:
        shots: Number of shots to be used in the noisy simulation of the qcircuit.
        optimization_level: Level of optimisation for the qiskit transpiler.
        transpiler_seed: Seed for the transpilation process of the quantum circuit.
        initial_layout: Initial layout used by a quantum computer, in case the noisy
            simulation tries to mimic a certain quantum computer (e.g., IBM Cairo).
        seed_simulator: The seed for the overarching simulation.
        private_config: Dictionary specifying mainly the ibmq token to access ibm
            real quantum computers, in case the simulation should mimic a certain
            real quantum computer.
    Returns:
        Dictionary of the configuration for the noisy simulation of the quantum circuit.
    """
    config_noisy = {
    "backend_config": {"shots": shots,
                       "optimization_level": optimization_level,
                       "transpiler_seed": transpiler_seed,
                       "initial_layout": initial_layout,
                       "seed_simulator": seed_simulator
                       }
    "ibmq_api": private_config["IBMQ"],
    }

    return config_noisy

def config_hardware(shots: int, optimization_level: int, transpiler_seed: int,
                    initial_layout: List[int], private_config: dict) -> dict:
    """The configuration loading for running our circuits on real quantum computers.

    Args:
        shots: Number of shots to be used in the noisy simulation of the qcircuit.
        optimization_level: Level of optimisation for the qiskit transpiler.
        initial_layout: Initial layout used by a quantum computer (e.g., IBM Cairo).
        private_config: Dictionary specifying mainly the ibmq token to access ibm
            real quantum computers.
    Returns:
        Dictionary of the configuration for running our circuit on a real quantum compt.
    """
    config_hardware = {
    "backend_config": {"shots": shots,
                       "optimization_level": optimization_level,
                       "transpiler_seed": seed,
                       "initial_layout": initial_layout,
                      },
    "ibmq_api": private_config["IBMQ"],
    }

    return config_hardware

def get_qdevice(
    run_type: str, wires: int, backend_name: str, config: dict
) -> pnl.device:
    """
    Configure a pennylane quantum device object used to construct qnodes.

    Args:
        run_type: Specifies the mode to run the (hybrid) VQC training: Ideal,
                  noisy simulation or on real quantum hardware. Possible values:
                  'ideal', 'noisy', 'hardware'.
        wires: Number of qubits.
        backend_name: Name of the quantum computer (IBMQ backend). Format: ibm(q)_<city_name>.
                      None, for the case of ideal simulation.
        config: Dictionary containing the configuration parameters of the backend
                device based on the run_type.
    Returns:
        Pennylane device based on the required run_type. For ideal, noisy (qiskit.Aer)
        or real quantum computer runs (IBMQ backends).
    """
    switcher = {
        "ideal": lambda: ideal_simulation(wires, config),
        # "noisy": lambda: noisy_simulation(wires, config),
        "hardware": lambda: hardware_run(wires, backend_name, config),
    }

    qdev = switcher.get(run_type, lambda: None)()
    if qdev is None:
        raise TypeError(
            tcols.FAIL + "Specified programme run type does not exist!" + tcols.ENDC
        )
    return qdev

def ideal_simulation(wires: int, config: dict) -> pnl.device:
    """
    Loads a pennylane device for ideal simulation.

    Args:
        wires: Number of qubits.
        config: keyword arguments for the pennylane device.
    Returns:
        Pennylane device for ideal simulation.
    """
    print_device_config(config)
    print(tcols.BOLD + "\nInitialising ideal (statevector) simulation." + tcols.ENDC)
    return pnl.device(wires=wires, **config)

def hardware_run(
    wires: int, backend_name: str, config: dict
) -> pennylane_qiskit.AerDevice:
    """
    Configure a IBMQ backend for a run on real quantum computers, using the pennylane
    interface.

    Args:
        wires: Number of qubits.
        backend_name: Name of the quantum computer (IBMQ backend). Format: ibm(q)_<city_name>.
        config: Keyword arguments for the pennylane device and qiskit backend. It
                contains arguments for the qiskit transpile and run methods.
    Returns:
        Pennylane device for noisy simulation on qiskit.Aer backend.
    """
    print(tcols.BOLD + "\nInitialising run on a quantum computer." + tcols.ENDC)
    print_device_config(config)
    qdev_hardware = pnl.device(
        "qiskit.ibmq",
        wires=wires,
        backend=backend_name,
        ibmqx_token=config["ibmq_api"]["token"],
        hub=config["ibmq_api"]["hub"],
        group=config["ibmq_api"]["group"],
        project=config["ibmq_api"]["project"],
        **config["backend_config"],
    )
    return qdev_hardware

def import_hyperparams(hyperparams_file) -> dict:
    """
    Import hyperparameters of an ae from json file.
    @model_path :: String of the path to a trained pytorch model folder
                   to import hyperparameters from the json file inside
                   that folder.

    returns :: Imported dictionary of hyperparams from .json file inside
        the trained model folder.
    """
    hyperparams_file = open(hyperparams_file)
    hyperparams = json.load(hyperparams_file)
    hyperparams_file.close()

    return hyperparams

def print_device_config(config: dict):
    """
    Print the configuration parameters of the used device.
    """
    print(tcols.OKCYAN + "Device configuration parameters:\n" + tcols.ENDC, config)
