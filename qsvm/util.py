# Utility methods for the qsvm.

import os
import sys
import joblib
from datetime import datetime

from .terminal_colors import tcols


def print_accuracies(test_accuracy, train_accuracy):
    """
    Prints the accuracies of the qsvm.
    @test_accuracy  :: Numpy array of the test data set accuracies.
    @train_accuracy :: Numpy array of the train data set accuracies.
    """
    print(f'Test Accuracy = {test_accuracy}')
    print(f'Training Accuracy = {train_accuracy}')


def create_output_folder(output_folder):
    """
    Creates output folder for the qsvm.
    @output_folder :: Name of the output folder for this particular
                      version of the qsvm.
    """
    if not os.path.exists('qsvm_models/' + output_folder):
        os.makedirs('qsvm_models/' + output_folder)


def save_qsvm(model, path):
    """
    Saves the qsvm model to a certain path.
    @model :: qsvm model object.
    @path  :: String of full path to save the model in.
    """
    joblib.dump(model, path)
    print(tcols.OKGREEN + "Trained model saved in: " + path + "\n" + tcols.ENDC)


def load_qsvm(path):
    """
    Load model from pickle file, i.e., deserialisation.
    @path  :: String of full path to save the model in.

    returns :: Joblib object that can be loaded by qiskit.
    """
    return joblib.load(path)


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
    original_stdout = sys.stdout
    with open('qsvm_models/' + output_folder + '/train.log', 'a+') as file:
        sys.stdout = file
        print(f'\n---------------------{datetime.now()}----------------------')
        print('QSVM model:',  output_folder)
        print('Autoencoder model:', ae_path)
        print('Data path:', qdata.ae_data.data_folder)
        print(f'ntrain = {len(qdata.ae_data.train_target)}, '
              f'ntest = {len(qdata.ae_data.test_target)}, '
              f'C = {qsvm.C}')
        print(f'Test Accuracy: {test_acc}, Training Accuracy: {train_acc}')
        print('-------------------------------------------\n')
        sys.stdout = original_stdout

    save_qsvm(qsvm, 'qsvm_models/' + output_folder + '/qsvm_model')
