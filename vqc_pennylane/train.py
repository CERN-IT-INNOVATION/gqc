# The VQC training script. Here, the vqc class is instantiated with some
# parameters, the circuit is built, and then it is trained on a data set.
# The hyperparameters of the circuit, the best weights, and a plot of the
# loss function evolution throughout the epochs are saved in a folder
# with the name of a user's choosing..
import os
from time import perf_counter

from . import qdata as qd
from . import util


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

    model = util.get_model(args)
    model.export_hyperparameters(outdir)
    model.export_architecture(outdir)

    train_loader, valid_loader, _ = util.get_data(qdata, args)
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
    print(f"Training completed in: {train_time_end-train_time_start:.2e} s or "
          f"{(train_time_end-train_time_start)/3600:.2e} h.")
