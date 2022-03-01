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

    model = util.get_model(args,args) # FIXME 
    model.export_hyperparameters(outdir)
    util.print_model_info(args["ae_model_path"], qdata, model)

    train_loader, valid_loader = util.get_data(qdata, args, args["hybrid"])
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
