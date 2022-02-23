# The VQC training script. Here, the vqc class is instantiated with some
# parameters, the circuit is built, and then it is trained on a data set.
# The hyperparameters of the circuit, the best weights, and a plot of the
# loss function evolution throughout the epochs are saved in a folder
# with the name of a user's choosing..
import os
from time import perf_counter
from typing import Tuple

from .vqc import VQC
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

    model = get_model(args)
    model.export_hyperparameters(outdir)
    util.print_model_info(args["ae_model_path"], qdata, model)

    train_loader, valid_loader = get_data(qdata, args)
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


def get_model(args) -> Tuple:
    """Choose the type of VQC to train. The normal vqc takes the latent space
    data produced by a chosen auto-encoder. The hybrid vqc takes the same
    data that an auto-encoder would take.

    Args:
        *args: Dictionary of hyperparameters to give to the vqc, (they can also be a
            subset of this dictionary).

    Returns:
        The instantiated vqc object.
    """
    qdevice = util.get_qdevice(
        args["run_type"],
        wires=args["nqubits"],
        backend_name=args["backend_name"],
        config=args["config"],
    )
    if args["hybrid_training"]:
        vqc_hybrid = VQCHybrid(qdevice, device="cpu", hpars=args)
        return vqc_hybrid

    vqc = VQC(qdevice, args)
    return vqc

def get_data(qdata, args):
    """Load the appropriate data depending on the type of vqc that is used.

    Args:
        qdata (object): Class object with the loaded data.
        *args: Dictionary of hyperparameters to give to the vqc, (they can also be a
            subset of this dictionary).

    Returns:
        The validation and test data tailored to the type of vqc that one
        is testing with this script.
    """
    if args["hybrid_vqc"]:
        return *get_hybrid_test_data(qdata, args)

    return *get_nonhybrid_test_data(qdata, args)

def get_nonhybrid_training_data(qdata, args) -> Tuple:
    """Loads the data from pre-trained autoencoder latent space when we have non
    hybrid VQC training.
    """
    train_features = qdata.batchify(
        qdata.get_latent_space("train"), args["batch_size"]
    )
    train_labels = qdata.batchify(
        qdata.ae_data.trtarget, args["batch_size"]
    )
    train_loader = [train_features, train_labels]

    valid_features = qdata.get_latent_space("valid")
    valid_labels = qdata.ae_data.vatarget
    valid_loader = [valid_features, valid_labels]

    return train_loader, valid_loader


def get_hybrid_training_data(qdata, args) -> Tuple:
    """Loads the raw input data for hybrid training."""
    train_loader = qdata.ae_data.get_loader(
        "train", "cpu", args["batch_size"], True
    )
    valid_loader = qdata.ae_data.get_loader("valid", "cpu", shuffle=True)
    return train_loader, valid_loader
