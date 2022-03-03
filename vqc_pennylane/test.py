# The VQC testing script. Here, a vqc is imported and data is passed through it.
# The results are quantified in terms of AUC.
import os
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn import metrics
from pennylane import numpy as np
import re

from . import qdata as qd
from . import util
from .terminal_colors import tcols


def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["ae_model_path"],
        test_events=args["ntest"],
        valid_events=args["nvalid"],
        seed=args["seed"],
    )
    outdir = re.sub("(best_model.pt|best_model.npy)","",args["vqc_path"])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    args = get_hparams_for_testing(args)
    model = util.get_model(args)
    model.load_model(args["vqc_path"])

    _, valid_loader, test_loader = util.get_data(qdata, args)
    x_valid, y_valid = util.split_data_loader(valid_loader)
    x_test, y_test = util.split_data_loader(test_loader)
    print("\n----------------------------------")
    print("VALID LOSS:")
    print(model.compute_loss(x_valid, y_valid))
    print("TEST LOSS:")
    print(model.compute_loss(x_test, y_test))
    print("----------------------------------\n")

    valid_pred = model.predict(x_valid)[-1]
    test_pred = model.predict(x_test)[-1]
    roc_plots(test_pred, y_test, outdir, "roc_plot")


def get_hparams_for_testing(args):
    """Imports the hyperparameters of the vqc at the given path and sets the
    optimiser to none such that no optimiser is loaded (none is needed since no
    training happens within the scope of this script).

    Args:
        args: Dictionary of hyperparameters for the vqc.

    Returns:
        Updated args dictionary with the loaded vqc hyperparameters.
    """

    hyperparams_file = os.path.join(args["vqc_path"], "hyperparameters.json")
    vqc_hyperparams = util.import_hyperparams(hyperparams_file)
    args.update(vqc_hyperparams)
    args.update({"optimiser": "none"})

    return args

def set_plotting_misc():
    """Set the misc settings of the plot such as the axes font size, the title size,
    the legend font size, and so on. These are the standard sizes for publication
    plot... adjust depending on what you need for specific journal.
    """
    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

def roc_plot_misc():
    """Miscellaneous settings for the roc plotting.
    """
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

def make_plots_output_folder(model_path, output_folder):
    """Make the output folder of the plots.

    Args:
        model_path: Path to a trained ae model.
        output_folder: Name of the output folder to save the plots in.

    Returns:
        Full path of the output folder to save the plots in.
    """
    plots_folder = os.path.join(model_path, output_folder)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    return plots_folder

def roc_plots(preds, target, model_path, output_folder):
    """Plot the ROC of the vqc predictions.

    Args:
        preds: Predictions of the vqc for a data set.
        target: Target corresponding to the data.
        model_path: Path to a trained vqc model.
        output_folder: Name of the output folder to save the plots in.
    """
    plots_folder = make_plots_output_folder(model_path, output_folder)
    set_plotting_misc()

    fpr, tpr, mean_auc, std_auc = compute_auc(preds, target)
    fig = plt.figure(figsize=(12, 10))
    roc_plot_misc()

    plt.plot(fpr, tpr, label=f"AUC: {mean_auc:.3f} ± {std_auc:.3f}", color="navy")
    plt.legend()

    fig.savefig(os.path.join(plots_folder, "roc_curve.pdf"))
    plt.close()

    print(tcols.OKCYAN + f"Latent roc plots were saved to {plots_folder}."  + tcols.ENDC)

def compute_auc(preds: np.array, target: np.array) -> Tuple:
    """Split a prediction array into 5, compute the AUC for each, and then calculate
    the mean and stardard deviation of the aucs.

    Args:
        preds: Array of the predictions as computed by the vqc.
        target: Array of the target corresponding to the predicted data..

    Returns:
        The ROC curve coordiantes, the AUC, and the standard deviation on the AUC.
    """
    pred_chunks = np.array_split(preds, 5)
    target_chunks = np.array_split(target, 5)

    aucs = np.array([])
    for prd, trg in zip(pred_chunks, target_chunks):
        fpr, tpr, thresholds = metrics.roc_curve(trg, prd)
        auc = metrics.roc_auc_score(trg, prd)
        aucs = np.append(aucs, auc)

    mean_auc = aucs.mean()
    std_auc = aucs.std()
    fpr, tpr, thresholds = metrics.roc_curve(target, preds)

    return fpr, tpr, mean_auc, std_auc
