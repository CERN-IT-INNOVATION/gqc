# Utility methods for the qsvm.

import os
import joblib
import warnings

from .terminal_colors import tcols


def print_accuracies(train_accuracy, valid_accuracy):
    """
    Prints the accuracies of the qsvm.
    @test_accuracy  :: Numpy array of the test data set accuracies.
    @train_accuracy :: Numpy array of the train data set accuracies.
    """
    print(tcols.OKGREEN + f"Training Accuracy   = {train_accuracy}")
    print(f"Validation Accuracy = {valid_accuracy}" + tcols.ENDC)


def create_output_folder(output_folder):
    """
    Creates output folder for the qsvm.
    @output_folder :: Name of the output folder for this particular
                      version of the qsvm.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def save_vqc(model, path):
    """
    Saves the qsvm model to a certain path.
    @model :: qsvm model object.
    @path  :: String of full path to save the model in.
    """
    joblib.dump(model, path)
    print("Trained model saved in: " + path)


def load_vqc(path):
    """
    Load model from pickle file, i.e., deserialisation.
    @path  :: String of full path to load the model from.

    returns :: Joblib object that can be loaded by qiskit.
    """
    return joblib.load(path)


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
    with warnings.catch_warnings(record=True) as w:
        print(tcols.OKGREEN)
        print(vqc.circuit)
        print(tcols.ENDC)
