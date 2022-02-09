# Utility methods for the qsvm.

import os
import warnings

from .terminal_colors import tcols


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
