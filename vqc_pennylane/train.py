# Main script of the vqc. Imports the data and runs the training of the
# VQC. A plot of the loss function is made using
from time import perf_counter
import numpy as np

from .vqc import VQC
from .terminal_colors import tcols
from . import qdata as qd
from . import util

from pennylane.optimize import AdamOptimizer


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["model_path"],
        train_events=args["train_events"],
        valid_events=args["valid_events"],
        test_events=0
    )

    train_features = qdata.batchify(qdata.get_latent_space("train"),
                                    args["batch_size"])
    train_labels = qdata.batchify(qdata.ae_data.trtarget, args["batch_size"])
    train_loader = [train_features, train_labels]

    valid_features = qdata.get_latent_space("valid")
    valid_labels = qdata.ae_data.vatarget
    valid_loader = [valid_features, valid_labels]

    vqc = VQC(args["nqubits"], train_features.shape[2])
    util.print_model_info(args["model_path"], qdata, vqc)

    train_time_start = perf_counter()
    vqc.train_vqc(train_loader, valid_loader, args["epochs"], 20, args["outdir"])
    train_time_end = perf_counter()
