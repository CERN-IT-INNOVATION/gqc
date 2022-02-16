# Utility methods for (Hybrid) VQC training.

import os
from turtle import back
from matplotlib import backend_bases
import pennylane as pnl
import pennylane_qiskit

from .terminal_colors import tcols
from qsvm.util import connect_quantum_computer, get_backend_configuration


def create_output_folder(output_folder):
    """
    Creates output folder for the qsvm.
    @output_folder :: Name of the output folder for this particular
                      version of the qsvm.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def print_model_info(ae_path, qdata, vqc):
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

def get_qdevice(run_type: str, wires: int, backend_name: str, config: dict) -> pnl.device:
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

def hardware_run(wires:int , backend_name: str, config: dict) -> pennylane_qiskit.AerDevice:
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

def print_device_config(config: dict):
    """
    Print the configuration parameters of the used device.
    """
    print( tcols.OKCYAN + "Device configuration parameters:\n" + tcols.ENDC, config)
