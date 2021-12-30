# Utility methods for the qsvm.

import os
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

from .terminal_colors import tcols


def print_accuracies(test_accuracy, train_accuracy):
    """
    Prints the accuracies of the qsvm.
    @test_accuracy  :: Numpy array of the test data set accuracies.
    @train_accuracy :: Numpy array of the train data set accuracies.
    """
    print(tcols.OKGREEN + f"Training Accuracy = {train_accuracy}")
    print(f"Test Accuracy     = {test_accuracy}" + tcols.ENDC)


def create_output_folder(output_folder):
    """
    Creates output folder for the qsvm.
    @output_folder :: Name of the output folder for this particular
                      version of the qsvm.
    """
    if not os.path.exists("qsvm_models/" + output_folder):
        os.makedirs("qsvm_models/" + output_folder)


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


def get_quatum_kernel_circuit(quantum_kernel, path, output_format='mpl',
                                **kwargs):
    '''
    Print the transpiled quantum kernel circuit
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
    n_params = quantum_kernel.feature_map.num_parameters
    feature_map_params_x = ParameterVector("x", n_params)
    feature_map_params_y = ParameterVector("y", n_params)
    qc_kernel_circuit = quantum_kernel.construct_circuit(
    feature_map_params_x,
    feature_map_params_y
    )
    qc_transpiled = quantum_kernel.quantum_instance\
                    .transpile(qc_kernel_circuit)[0]
    #save the circuit image FIXME save in qsvm_model/model_folder/
    path += '/quantum_kernel_circuit_plot'
    print('\nSaving quantum kernel circuit in: ', path)
    qc_transpiled.draw(
        output = output_format,
        filename = path,
        **kwargs,
    )
    return qc_transpiled

#FIXME plot_circuit_layout() raises error because gets None out of:
# cmap = backend.configuration().coupling_map in the source code/
# If we do it with AerSimulator() instead of NoiseModel then it should
# work.
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
    fig.savefig(save_path)

def save_kernel_matrix():
    '''
    Save kernel matrix as a numpy array. To compare between ideal
    computations, simulations and hardware runs.
    '''
    # TODO


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
    #FIXME it is printed in one go together with the other print()
    print('Enabling IBMQ account using provided token...', end="")
    IBMQ.enable_account(ibmq_token)
    provider = IBMQ.get_provider(hub='ibm-q-cern', group='internal', 
                                 project='qvsm4higgs')
    try: 
        quantum_computer_backend = provider.get_backend(backend_name)
    except QiskitBackendNotFoundError:
        raise AttributeError('Backend name not found in provider\'s list')
    print(tcols.OKGREEN +' Loaded IBMQ backend: ' + backend_name+'.' + tcols.ENDC)
    return quantum_computer_backend


def get_backend_configuration(backend) -> Tuple:
    '''
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


def ideal_simulation(seed) -> QuantumInstance:
    '''
    Defines QuantumInstance for an ideal (statevector) simulation (no noise, no
    sampling statistics uncertainties). 
    
    Args:
         @seed :: FIXME Does seed matter for statevector simulation and QuantumKernel?
    '''
    
    print('Initialising ideal (statevector) simulation.')
    quantum_instance = QuantumInstance(
         backend = Aer.get_backend('aer_simulator_statevector'),
         seed_simulator = seed
         )
    return quantum_instance

#def custom_aer_simulation(**kwargs):
    '''
    Method that prepares aer_simulator (i.e. with statistical measurement uncertainty)
    and on top one can add whatever tunable noise type they want (1-gate errors, 2-gate errors)
    etc.
    '''

#FIXME backend = AerSimulator.from_backend(quantum_computer_backend)
# ^ Panos recommendation. to do noisy sim efficiently.
def noisy_simulation(ibmq_token,backend_name,**kwargs)\
                      -> QuantumInstance:
    '''
    Prepare a QuantumInstance object for simulation with noise based on the 
    real quantum computer calibration data.

    Args:
         @qubit_layout :: Map of abstract circuit qubits to physical qubits as
                          defined on the hardware.
    '''
    backend = connect_quantum_computer(ibmq_token,backend_name)
    noise_model, coupling_map, basis_gates = \
                get_backend_configuration(backend)
    
    quantum_instance = QuantumInstance(
        backend = Aer.get_backend('aer_simulator'),
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        noise_model=noise_model,
        **kwargs
        )
    return quantum_instance

#TODO is the seed needed here?
def hardware_run(backend_name, ibmq_token, **kwargs):
    '''
    Configure QuantumInstance based on a quantum computer. The circuits will
    be sent as jobs to be exececuted on the specified device in IBMQ.
    
    Args:
         @backend_name :: Name of the quantum computer, form ibmq_<city_name>.
         @ibmq_token   :: Token for authentication by IBM to give access
                          to non-pubic hardware.

    Returns:
            QuantumInstance object with quantum computer backend.
    '''
    quantum_computer_backend = connect_quantum_computer(ibmq_token,backend_name)
    quantum_instance = QuantumInstance(
        backend = quantum_computer_backend, 
        **kwargs
        )
    return quantum_instance


def configure_quantum_instance(ibmq_token, sim_type,backend_name = None,
                                **kwargs) -> QuantumInstance:
    '''
    Gives the final quantum_instance required for running the Quantum
    kernel.
    '''
    #FIXME better way to write the if-statements below
    if sim_type == 'ideal' and backend_name is not None:
        raise TypeError(tcols.FAIL + 'Why give real backend when you want' 
                        'ideal simulation?' +tcols.ENDC )
    if (sim_type == 'noisy' or sim_type == 'hardware') and (backend_name is None):
        raise TypeError(tcols.FAIL + 'Need to specify backend name (\'ibmq_<city_name>\')'
                        ' when running a noisy simulation or running on hardware!'
                        + tcols.ENDC)
    
    switcher = {
            #TODO backend name for ideal_simulation? Edge cases?
            'ideal'    : lambda: ideal_simulation(**kwargs),
            'noisy'    : lambda: noisy_simulation(ibmq_token=ibmq_token,
                                                    backend_name=backend_name,
                                                    **kwargs),
            'hardware' : lambda: hardware_run(backend_name = backend_name, 
                                              ibmq_token = ibmq_token, **kwargs
                )
        }
    #FIXME why is the (), callable needed? 
    quantum_instance = switcher.get(sim_type, lambda: None)()
    if quantum_instance is None:
        raise TypeError('Specified simulation type does not exist!')
    return quantum_instance