# Utility methods for the qsvm.

import os
import warnings
import joblib
from datetime import datetime
from typing import Tuple, Type
from qiskit import IBMQ
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.circuit import ParameterVector
from qiskit.visualization import plot_circuit_layout
from qiskit.providers.aer.backends import AerSimulator

from .terminal_colors import tcols


def print_accuracies(test_accuracy, train_accuracy):
    """
    Prints the accuracies of the qsvm.
    @test_accuracy  :: Numpy array of the test data set accuracies.
    @train_accuracy :: Numpy array of the train data set accuracies.
    """
    print(tcols.OKGREEN + f"Training Accuracy = {train_accuracy}")
    print(f"Test Accuracy     = {test_accuracy}" + tcols.ENDC)


def create_output_folder(args, qsvm):
    """
    Creates output folder for the qsvm and returns the path (str)
    @args (dict)         :: The argument dictionary defined in the qsvm_launch
                            script.
    @qsvm (sklearn.SVC)  :: QSVM object.
    Returns:
            @out_path (str), the path where all files relevant to the qsvm
            will be saved.
    """
    args["output_folder"] = args["output_folder"] + f"_c={qsvm.C}" \
                            + f"_{args['run_type']}"
    if (args["backend_name"] is not None): 
        args["output_folder"] += f'_{args["backend_name"]}'
    out_path = "models/" + args["output_folder"]
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    return out_path


def save_qsvm(model, path):
    """
    Saves the qsvm model to a certain path.
    @model :: qsvm model object.
    @path  :: String of full path to save the model in.
    """
    joblib.dump(model, path)
    print("Trained model saved in: " + path)


def load_qsvm(path):
    """
    Load model from pickle file, i.e., deserialisation.
    @path  :: String of full path to save the model in.

    returns :: Joblib object that can be loaded by qiskit.
    """
    return joblib.load(path)


def print_model_info(ae_path, qdata, qsvm):

    print("\n-------------------------------------------")
    print(f"Autoencoder model: {ae_path}")
    print(f"Data path: {qdata.ae_data.data_folder}")
    print(
        f"ntrain = {len(qdata.ae_data.trtarget)}, "
        f"ntest = {len(qdata.ae_data.tetarget)}, "
        f"C = {qsvm.C}"
    )
    print("-------------------------------------------\n")


def save_model(qdata, qsvm, train_acc, test_acc, output_folder, ae_path):
    """
    Save the model and a log of useful info regarding the saved model.
    @qdata         :: The data that was processed by the qsvm.
    @qsvm          :: The qiskit qsvm object.
    @train_acc     :: Numpy array of the training accuracies.
    @test_acc      :: Numpy array of the testing accuracies.
    @output_folder :: String of the output folder where the saving is.
    @ae_path       :: The path to the ae used in reducing the qdata.
    """
    save_qsvm(qsvm, "qsvm_models/" + output_folder + "/qsvm_model")


def get_quantum_kernel_circuit(quantum_kernel, path, output_format='mpl',
                                **kwargs):
    '''
    Save the transpiled quantum kernel circuit
    Args:
         @quantum_kernel (QuantumKernel) :: QuantumKernel object used in the
                                            QSVM training.
         @path (str)                     :: Path to save the output figure.
         @output_format (str)            :: The format of the image. Formats:
                                            'text', 'mlp', 'latex', 'latex_source'.
         @kwargs                         :: Keyword arguemnts for 
                                            QuantumCircuit.draw()
    Returns:
            Transpiled QuantumCircuit that represents the quantum kernel.
            i.e., the circuit that will be executed on the backend.
    '''
    print('\nCreating the quanntum kernel circuit...')
    n_params = quantum_kernel.feature_map.num_parameters
    feature_map_params_x = ParameterVector("x", n_params)
    feature_map_params_y = ParameterVector("y", n_params)
    qc_kernel_circuit = quantum_kernel.construct_circuit(
    feature_map_params_x,
    feature_map_params_y
    )
    qc_transpiled = quantum_kernel.quantum_instance\
                    .transpile(qc_kernel_circuit)[0]
    
    path += '/quantum_kernel_circuit_plot'
    print(tcols.OKCYAN + 'Saving quantum kernel circuit in: ' + tcols.ENDC, 
          path)
    qc_transpiled.draw(
        output = output_format,
        filename = path,
        **kwargs,
    )
    return qc_transpiled


def save_circuit_physical_layout(circuit, backend, save_path):
    '''
    Plot and save the quantum circuit and its physical layout on the backend 
    .
    Args:
         @circuit (QuantumCircuit) :: Circuit to plot on the backend.
         @backend                  :: The physical quantum computer or 
                                      thereof.
         @save_path (str)          :: Path to save figure.
    '''
    fig = plot_circuit_layout(circuit,backend)
    save_path += '/circuit_physical_layout'
    print(tcols.OKCYAN + 'Saving physical circuit layout in: ' + tcols.ENDC, 
          save_path)    
    fig.savefig(save_path)
    

def connect_quantum_computer(ibmq_token, backend_name):
    '''
    Load a IBMQ-experience backend using a token (IBM-CERN hub credentials)
    This backend (i.e. quantum computer) can either be used for running on
    the real device or to load the calibration (noise/error info). With the
    latter data we can do a simulation of the hardware behaviour.
    
    Args:

    @ibmq_token (string)   :: Token for authentication by IBM to give access
                              to non-pubic hardware.
    @backend_name (string) :: Quantum computer name.
    
    Returns: IBMQBackend qiskit object.
    '''
    print('Enabling IBMQ account using provided token...', end="")
    IBMQ.enable_account(ibmq_token)
    provider = IBMQ.get_provider(hub='ibm-q-cern', group='internal', 
                                 project='qvsm4higgs')
    try: 
        quantum_computer_backend = provider.get_backend(backend_name)
    except QiskitBackendNotFoundError:
        raise AttributeError(tcols.FAIL + 'Backend name not found in provider\'s '
                             'list'+tcols.ENDC)
    print(tcols.OKGREEN +' Loaded IBMQ backend: ' + backend_name+'.' + tcols.ENDC)
    return quantum_computer_backend


def get_backend_configuration(backend) -> Tuple:
    '''
    @ DEPRECATED
    
    Gather backend configuration and properties from the calibration data.
    
    Args:
    @backend :: IBMQBackend object representing a a real quantum computer.

    Returns:
            @noise_model from the 1-gate, 2-gate (CX) errors, thermal relaxation,
            etc.
            @coupling_map: connectivity of the physical qubits.
            @basis_gates: gates that are physically implemented on the hardware.
            the transpiler decomposes the generic/abstract circuit to these
            physical basis gates, taking into acount also the coupling_map.
    '''
    noise_model = NoiseModel.from_backend(backend)
    coupling_map = backend.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    return noise_model, coupling_map, basis_gates


def ideal_simulation(**kwargs) -> QuantumInstance:
    '''
    Defines QuantumInstance for an ideal (statevector) simulation (no noise, no
    sampling statistics uncertainties). 
    
    Args:
         Keyword arguments of the QuantumInstance object.
    '''
    
    print(tcols.BOLD + '\nInitialising ideal (statevector) simulation.'\
          + tcols.ENDC)
    quantum_instance = QuantumInstance(
         backend = Aer.get_backend('aer_simulator_statevector'),
         **kwargs
         )
    # None needed to specify that no backend device is loaded for ideal sim.
    return quantum_instance, None 

#def custom_aer_simulation(**kwargs):
    '''
    Method that prepares aer_simulator (i.e. with statistical measurement uncertainty)
    and on top one can add whatever tunable noise type they want (1-gate errors, 2-gate errors)
    etc.
    '''


def noisy_simulation(ibmq_token,backend_name,**kwargs)\
                      -> QuantumInstance:
    '''
    Prepare a QuantumInstance object for simulation with noise based on the 
    real quantum computer calibration data.

    Args:
         @qubit_layout :: Map of abstract circuit qubits to physical qubits as
                          defined on the hardware.
    Returns:
            @QuantumInstance object to be used for the simulation.
            @backend on which the noisy simulation is based.
    '''
    print(tcols.BOLD + '\nInitialising noisy simulation.' + tcols.ENDC)
    quantum_computer_backend = connect_quantum_computer(ibmq_token,
                                                        backend_name)
    backend = AerSimulator.from_backend(quantum_computer_backend)
    
    quantum_instance = QuantumInstance(
        backend = backend,
        **kwargs
        )
    return quantum_instance, quantum_computer_backend


def hardware_run(backend_name, ibmq_token, **kwargs):
    '''
    Configure QuantumInstance based on a quantum computer. The circuits will
    be sent as jobs to be exececuted on the specified device in IBMQ.
    
    Args:
         @backend_name :: Name of the quantum computer, form ibmq_<city_name>.
         @ibmq_token   :: Token for authentication by IBM to give access
                          to non-pubic hardware.

    Returns:
            @QuantumInstance object with quantum computer backend.
            @The quantum computer backend object.
    '''
    print(tcols.BOLD + '\nInitialising run on a quantum computer.' + tcols.ENDC)
    quantum_computer_backend = connect_quantum_computer(ibmq_token,backend_name)
    quantum_instance = QuantumInstance(
        backend = quantum_computer_backend, 
        **kwargs
        )
    return quantum_instance, quantum_computer_backend


def configure_quantum_instance(ibmq_token, run_type, backend_name = None,
                                **kwargs) -> QuantumInstance:
    '''
    Gives the QuantumInstance object required for running the Quantum kernel.
    The quantum instance can be configured for a simulation of a backend with
    noise, an ideal (statevector) simulation or running on a real quantum 
    device.
    Args:
         @ibmq_token (string)   :: Token for authentication by IBM to give 
                                   access to non-pubic hardware.
         @run_type (string)     :: Takes values the possible values {ideal, 
                                   noisy, hardware} to specify what type of 
                                   backend will be provided to the quantum
                                   instance object.
         @backend_name (string) :: Name of the quantum computer to run or base
                                   the noisy simulation on. For ideal runs it 
                                   can be set ton "none".
         @**kwargs     (dict)   :: Dictionary of keyword arguments for the 
                                   QuantumInstance.
    Returns:
            @QuantumInstance object to be used in the QuantumKernel training.
            @backend that is being used. None if an ideal simulation is initi-
             ated.
    '''
    if (run_type == 'noisy' or run_type == 'hardware') and (backend_name is None):
        raise TypeError(tcols.FAIL + 'Need to specify backend name (\'ibmq_<city_name>\')'
                        ' when running a noisy simulation or running on hardware!'
                        + tcols.ENDC)
    
    switcher = {
            'ideal'    : lambda: ideal_simulation(**kwargs),
            'noisy'    : lambda: noisy_simulation(ibmq_token=ibmq_token,
                                                    backend_name=backend_name,
                                                    **kwargs),
            'hardware' : lambda: hardware_run(backend_name = backend_name, 
                                              ibmq_token = ibmq_token, 
                                              **kwargs
                )
        }
    #FIXME why is the (), callable needed? 
    quantum_instance, backend = switcher.get(run_type, lambda: None)()
    if quantum_instance is None:
        raise TypeError('Specified programme run type does not exist!')
    return quantum_instance, backend