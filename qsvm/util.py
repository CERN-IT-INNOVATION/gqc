# Utility methods for the qsvm.

import os
import joblib
from datetime import datetime
#from typing import String FIXME caused: 
# "ImportError: cannot import name 'String' from 'typing' 
# (/work/vabelis/miniconda3/envs/ae_qml/lib/python3.8/typing.py)"
from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel

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

def save_kernel_matrix():
    '''
    Save kernel matrix as a numpy array. To compare between ideal
    computations, simulations and hardware runs.
    '''
    # TODO
def configure_backend(ibmq_token, backend_name):
    '''
    Load a IBMQ-experience backend using a token (IBM-CERN hub credentials)
    This backend (i.e. quantum computer) can either be used for running on
    the real device or to load the calibration (noise/error info). With the
    latter data we can do a simulation of the hardware behaviour.
    
    Args:

    @ibmq_token (string)   :: Token for authentication by IBM to give access
                              to non-pubic hardware.
    @backend_name (string) :: Quantum computer name.
    
    Returns: Backend qiskit object.
    '''
    #FIXME Put import IBMQ here or top?
    print('Enabling IBMQ account using provided token...')
    IBMQ.enable_account(ibmq_token)
    provider = IBMQ.get_provider(hub='ibm-q-cern', group='internal', 
                                project='qvsm4higgs')
    print('Loading IBMQ backend: ' + backend_name)
    if not provider.backends(name = backend_name):
        print('Backend name not found in provider\'s list')
    backend = provider.get_backend(backend_name)
    print('DONE.') #print in same line FIXME
    return backend 


#TODO What is the best place to define the method below
def get_backend_configuration(backend):
    '''
    Gather backend configuration and properties from the calibration data.
    
    Args:
    @backend :: The IBMQBackend object representing a quantum computer.

    Returns:
            @noise_model from the 1-gate, 2-gate (CX) errors, thermal relaxa

    '''
    noise_model = NoiseModel.from_backend(backend)
    coupling_map = backend.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    return noise_model, coupling_map, basis_gates