# Utility methods for the qsvm.

"""
Sidenote:
    To save sklearn models joblib package is used. Serialization and
    de-serialization of objects is python-version sensitive.

    Alternatives: As of Python 3.8 and numpy 1.16, pickle protocol 5
    introduced in PEP 574 supports efficient serialization and de-serialization
    for large data buffers natively using the standard library:
         pickle.dump(large_object, fileobj, protocol=5)
"""

import os, sys, joblib
from datetime import datetime

from .terminal_colors import tcols

def print_accuracies(test_accuracy, train_accuracy):
    # Prints the accuracies of the qsvm.
    print(f'Test Accuracy = {test_accuracy}')
    print(f'Training Accuracy = {train_accuracy}')

def create_output_folder(output_folder):
    # Creates output folder for the qsvm.
    if not os.path.exists('qsvm_models/' + output_folder):
        os.makedirs('qsvm_models/' + output_folder)

def save_qsvm(model, path):
    # Saves the qsvm model to a certain path.
    joblib.dump(model, path)
    print(tcols.OKGREEN + "Trained model saved in: " + path + "\n" + tcols.ENDC)

def load_qsvm(path):
    # Load model from pickle file, i.e., deserialisation.
    return joblib.load(path)

def save_model(qdata, qsvm, train_acc, test_acc, output_folder, ae_path):
    # Save the model and a log of useful info regarding the saved model.
    original_stdout = sys.stdout
    with open('qsvm_models/'+ output_folder +'/train.log', 'a+') as file:
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

    save_qsvm(qsvm, 'qsvm_models/' +  output_folder + '/qsvm_model')
