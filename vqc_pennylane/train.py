# The VQC training script. Here, the vqc class is instantiated with some
# parameters, the circuit is built, and then it is trained on a data set.
# The hyperparameters of the circuit, the best weights, and a plot of the
# loss function evolution throughout the epochs are saved in a folder
# with the name of a user's choosing..
import os
from time import perf_counter
from typing import Tuple

from .vqc import VQC
from .terminal_colors import tcols
from . import qdata as qd
from . import util
from .vqc_hybrid import VQCHybrid


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["ae_model_path"],
        train_events=args["train_events"],
        valid_events=args["valid_events"],
        seed=args["seed"],
    )
    outdir = "./trained_vqcs/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    model, train_loader, valid_loader = get_data_and_model(qdata, args)

    model.export_hyperparameters(outdir)

    time_the_training(
        model.train_model, train_loader, valid_loader, args["epochs"], 20, outdir
    )
    model.loss_plot(outdir)


def time_the_training(train, *args):
    """Times the training of the VQC.

    Args:
        train (callable): The training method of the VQC.
        *args: Arguments for the train_model callable of the VQC.
    """
    train_time_start = perf_counter()
    train(*args)
    train_time_end = perf_counter()
    print(f"Training completed in: {train_time_end-train_time_start:.2e} s")


def get_data_and_model(qdata_loader, args) -> Tuple:
    """Choose the type of VQC to train. The normal vqc takes the latent space
    data produced by a chosen auto-encoder. The hybrid vqc takes the same
    data that an auto-encoder would take. The correct data is assigned in
    this method. Additionally, these two types of vqc also take different
    initialising hyperparameters; this method takes care of that as well.

    Args:
        qdata_loader (array): Data loader class from qdata.py.
        *args: Dictionary of hyperparameters to give to the vqc, (they can also be a
            subset of this dictionary).

    Returns:
        The instantiated vqc object and the training and validation data to train it on.
    """
    if args["hybrid_training"]:
        vqc_hybrid = VQCHybrid(qdevice="default.qubit", device="cpu", hpars=args)
        return vqc_hybrid, *get_hybrid_training_data(qdata_loader, args)

    vqc = VQC("default.qubit", args)
    util.print_model_info(args["ae_model_path"], qdata_loader, vqc)
    return vqc, *get_nonhybrid_training_data(qdata_loader, args)


def get_nonhybrid_training_data(qdata_loader, args) -> Tuple:
    """Loads the data from pre-trained autoencoder latent space when we have non
    hybrid VQC training.

    Args:
        qdata_loader: Data loader class from qdata.py.
        *args: Dictionary of hyperparameters to give to the vqc, (they can also be a
            subset of this dictionary).

    Returns:
        Training and validation data, in the latent space, for training the non-hybrid
        vqc on.
    """
    train_features = qdata_loader.batchify(
        qdata_loader.get_latent_space("train"), args["batch_size"]
    )
    train_labels = qdata_loader.batchify(
        qdata_loader.ae_data.trtarget, args["batch_size"]
    )
    train_loader = [train_features, train_labels]

    valid_features = qdata_loader.get_latent_space("valid")
    valid_labels = qdata_loader.ae_data.vatarget
    valid_loader = [valid_features, valid_labels]
    return train_loader, valid_loader


def get_hybrid_training_data(qdata_loader, args) -> Tuple:
    """Loads the raw input data for hybrid training.

    Args:
        qdata_loader: Data loader class from qdata.py.
        *args: Dictionary of hyperparameters to give to the vqc, (they can also be a
            subset of this dictionary).

    Returns:
        Training and validation pytorch loaders, loaded on the cpu.
    """
    train_loader = qdata_loader.ae_data.get_loader(
        "train", "cpu", args["batch_size"], True
    )
    valid_loader = qdata_loader.ae_data.get_loader("valid", "cpu", shuffle=True)
    return train_loader, valid_loader
