# Main script of the vqc. Imports the data and runs the training of the
# VQC. A plot of the loss function is made using
from time import perf_counter
import numpy as np

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import ADAM

from .vqc import VQC
from .terminal_colors import tcols
from . import qdata as qd
from . import util


seed = 12345
# Ensure same global behaviour.
algorithm_globals.random_seed = seed


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["model_path"],
        train_events=args["train_events"],
        valid_events=args["valid_events"],
        test_events=0,
    )

    train_features = qdata.batchify(qdata.get_latent_space("train"), args["batch_size"])
    train_labels = qdata.batchify(qdata.ae_data.trtarget, args["batch_size"])
    valid_features = qdata.get_latent_space("valid")
    valid_labels = qdata.ae_data.vatarget

    backend = Aer.get_backend("aer_simulator_statevector")
    qinst = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)

    # optimizer = ADAM(maxiter=100, lr=0.001)
    vqc = VQC(
        args["nqubits"],
        args["feature_map"],
        args["ansatz"],
        train_features.shape[2],
        warm_start=False,
        quantum_instance=qinst,
        loss=args["loss"],
        optimizer=None,
    )

    print(tcols.OKCYAN + "Training the VQC..." + tcols.ENDC)
    util.print_model_info(args["model_path"], qdata, vqc)

    train_time_init = perf_counter()
    vqc = train(
        vqc, train_features, train_labels, valid_features, valid_labels, args["epochs"]
    )
    train_time_fina = perf_counter()
    print(f"Training completed in: {train_time_fina-train_time_init:.2e} s")

    util.create_output_folder("trained_vqcs/" + args["output_folder"])
    util.save_vqc(vqc, "trained_vqcs/" + args["output_folder"] + "/model")


def train(vqc, train_features, train_labels, valid_features, valid_labels, epochs):
    """
    Training the vqc.
    @vqc            :: The vqc qiskit object to be trained.
    @train_features :: Numpy array with training data divided into batches.
    @train_labels   :: Numpy array with training target div into batches.
    @epochs         :: Int of the number of epochs this should be trained.
    """
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{epochs}")
        for data_batch, target_batch in zip(train_features, train_labels):
            fit_time_init = perf_counter()
            vqc.fit(data_batch, target_batch)
            fit_time_fina = perf_counter()
            print(f"Fit completed in: {fit_time_fina-fit_time_init:.2e} s")

    return vqc
