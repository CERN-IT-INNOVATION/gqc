# Utility methods for (Hybrid) VQC training.

import os
from typing import List
from typing import Tuple
from typing import Union
import subprocess
import json
import pennylane as pnl
import pennylane_qiskit

from .terminal_colors import tcols
from .vqc import VQC
from .vqc_hybrid import VQCHybrid
from torch.utils.data import DataLoader


def create_output_folder(output_folder):
    """
    Creates output folder for the qsvm.
    @output_folder :: Name of the output folder for this particular
                      version of the (hybrid) VQC.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


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
    private_config = None
    try:
        with open(pconfig_path) as pconfig:
            private_config = json.load(pconfig)
    except OSError:
        FileNotFoundError("Error in reading private config: process aborted.")

    return private_config


def get_freest_gpu():
    """Returns the number of the freest GPU."""

    # Get the list of GPUs via nvidia-smi.
    smi_query_result = subprocess.check_output(
        "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    )

    # Extract the usage information
    gpu_info = smi_query_result.decode("utf-8").split("\n")
    gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
    gpu_info = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info]

    # Find the most free gpu.
    _, freest_gpu = min((val, idx) for idx, val in enumerate(gpu_info))
    return str(freest_gpu)


def config_ideal(name: str, shots: int = None) -> dict:
    """The configuration loading of the ideal simulation.

    Args:
        name: The name of the simulator to be used in simulating the quantum circuit.
        shots: The number of shots to be used in the simulation.

    Returns:
        Dictionary of the configuration for the ideal simulation of the quantum circuit.
    """
    config_ideal = {"name": name, "shots": shots}

    return config_ideal


def config_noisy(
    shots: int,
    optimization_level: int,
    transpiler_seed: int,
    initial_layout: List[int],
    seed_simulator: int,
    private_config: dict,
) -> dict:
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
        "backend_config": {
            "shots": shots,
            "optimization_level": optimization_level,
            "transpiler_seed": transpiler_seed,
            "initial_layout": initial_layout,
            "seed_simulator": seed_simulator,
        },
        "ibmq_api": private_config["IBMQ"],
    }

    return config_noisy


def config_hardware(
    shots: int,
    optimization_level: int,
    transpiler_seed: int,
    initial_layout: List[int],
    private_config: dict,
) -> dict:
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
        "backend_config": {
            "shots": shots,
            "optimization_level": optimization_level,
            "transpiler_seed": transpiler_seed,
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
    print(tcols.BOLD + "\nInitialising ideal (statevector) simulation.\n" + tcols.ENDC)
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


def get_model(args) -> Tuple:
    """Choose the type of VQC to train. The normal vqc takes the latent space
    data produced by a chosen auto-encoder. The hybrid vqc takes the same
    data that an auto-encoder would take, since it has an encoder or a full
    auto-encoder attached to it.

    Args:
        args: Dictionary of hyperparameters for the vqc.

    Returns:
        An instance of the vqc object with the given specifications (hyperparams).
    """
    qdevice = get_qdevice(
        args["run_type"],
        wires=args["nqubits"],
        backend_name=args["backend_name"],
        config=args["config"],
    )

    if args["hybrid"]:
        vqc_hybrid = VQCHybrid(qdevice, device="cpu", hpars=args)
        return vqc_hybrid

    vqc = VQC(qdevice, args)
    return vqc


def get_data(qdata, args):
    """Load the appropriate data depending on the type of vqc that is used.

    Args:
        qdata: Class object with the loaded data.
        args: Dictionary containing specifications relating to the data.

    Returns:
        The validation and test data tailored to the type of vqc that one
        is testing.
    """
    if args["hybrid"]:
        return get_hybrid_data(qdata, args)

    return get_nonhybrid_data(qdata, args)


def get_nonhybrid_data(qdata, args) -> Tuple:
    """Loads the data from pre-trained autoencoder latent space when we have non
    hybrid VQC testing.
    """
    train_loader = None
    if "ntrain" in args:
        train_features = qdata.batchify(
            qdata.get_latent_space("train"), args["batch_size"]
        )
        train_labels = qdata.batchify(qdata.ae_data.trtarget, args["batch_size"])
        train_loader = [train_features, train_labels]

    valid_features = qdata.get_latent_space("valid")
    valid_labels = qdata.ae_data.vatarget
    valid_loader = [valid_features, valid_labels]

    test_features = qdata.get_latent_space("test")
    test_labels = qdata.ae_data.tetarget
    test_loader = [test_features, test_labels]

    return train_loader, valid_loader, test_loader


def get_hybrid_data(qdata, args) -> Tuple:
    """Loads the raw input data for hybrid testing."""
    train_loader = None
    test_loader = None
    if "ntrain" in args:
        train_loader = qdata.ae_data.get_loader(
            "train", "cpu", args["batch_size"], True
        )
    print("From qdata:")
    #for batch in train_loader:
    #    print(batch)
    #exit(1)
    valid_loader = qdata.ae_data.get_loader("valid", "cpu", shuffle=True)
    if "ntest" in args:
        test_loader = qdata.ae_data.get_loader("test", "cpu", shuffle=True)

    return train_loader, valid_loader, test_loader


def split_data_loader(data_loader: Union[DataLoader, List]) -> Tuple:
    """
    Splits a data loader object to the corresponding data features (x_data) and
    labels (y_data). For the VQC training the data loader is simply a list of the
    form: [x_data, y_data]. For a PyTorch DataLoader object, which is not subscriptable,
    on needs to do a transformation to an iterable in order to access its elements.

    Args:
        data_loader: The object which we want to split.
    Returns:
        The array of the data vectors and their corresponding labels.
    """
    try:
        x_data, y_data = data_loader[0], data_loader[1]
    except TypeError:
        x_data, y_data = iter(data_loader).next()
        x_data, y_data = x_data.cpu().numpy(), y_data.cpu().numpy()
    return x_data, y_data


def get_kfolded_data(qdata, args: dict):
    """Get the kfolded data from the qdata and mold it with respect to what kind of
    vqc we're working with, either hybrid or non-hybrid.

    Args:
        qdata: Class object with the loaded data.
        args: Dictionary containing specifications relating to the data.

    Returns:
        The validation and test data tailored to the type of vqc that one
        is testing.
    """
    if args["hybrid"]:
        x_valid, y_valid = qdata.get_kfolded_data(datat="valid", latent=False)
        x_test, y_test = qdata.get_kfolded_data(datat="test", latent=False)

        return x_valid, y_valid, x_test, y_test

    x_valid, y_valid = qdata.get_kfolded_data(datat="valid", latent=True)
    x_test, y_test = qdata.get_kfolded_data(datat="test", latent=True)

    return x_valid, y_valid, x_test, y_test
