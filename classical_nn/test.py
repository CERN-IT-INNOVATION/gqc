# The classical NN testing script. Here, a trained NN is imported 
# and data is passed through it.The results are quantified in terms of ROCs and AUCs.

import os
import sys
import time
sys.path.append("..")
import argparse
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn import metrics
from pennylane import numpy as np

from autoencoders import util as ae_util
from vqc_pennylane import util
from vqc_pennylane import qdata as qd
from vqc_pennylane.terminal_colors import tcols
from neural_network import NeuralNetwork


def main():
    args = get_arguments()
    device = ae_util.define_torch_device()
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],
        args["ae_model_path"],
        test_events=args["ntest"],
        valid_events=args["nvalid"],
        seed=args["seed"],
        kfolds=args["kfolds"],
    )
    args = get_hparams_for_testing(args)
    model = NeuralNetwork(device, args)
    model.load_model(args["nn_model_path"])

    _, valid_loader, test_loader = util.get_hybrid_data(qdata, args)
    x_valid, y_valid = util.split_data_loader(valid_loader)
    x_test, y_test = util.split_data_loader(test_loader)
    print("\n----------------------------------")
    print("VALID LOSS:")
    print(model.compute_loss(x_valid, y_valid))
    print("TEST LOSS:")
    print(model.compute_loss(x_test, y_test))
    print("----------------------------------\n")

    x_valid, y_valid = qdata.get_kfolded_data(datat="valid", latent=False)
    x_test, y_test = qdata.get_kfolded_data(datat="test", latent=False)
    valid_preds = np.array([model.predict(x)[-1] for x in x_valid])
    test_preds = np.array([model.predict(x)[-1] for x in x_test])

    model_dir = os.path.dirname(args["nn_model_path"])
    roc_plots(test_preds, y_test, model_dir, "roc_plot")

    #if args["hybrid"]:
    #    x_test_sig, x_test_bkg = qdata.ae_data.split_sig_bkg(x_test, y_test)
    #    sig = model.predict(x_test_sig)
    #    bkg = model.predict(x_test_bkg)
    #    roc_plots(sig[0], bkg[0], args["model_path"], "latent_roc")
    #    sig_vs_bkg(sig[0], bkg[0], args["vqc_path"], "latent_plots")
    

def get_hparams_for_testing(args: dict) -> dict:
    """Imports the hyperparameters of the NN model at the given path and sets the
    optimiser to none such that no optimiser is loaded (none is needed since no
    training happens within the scope of this script).

    Args:
        args: Dictionary of hyperparameters for the NN classifier.

    Returns: Updated args dictionary with the loaded NN hyperparameters.
    """

    hyperparams_file = os.path.join(
        os.path.dirname(args["nn_model_path"]), "hyperparameters.json"
    )
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
    """Miscellaneous settings for the roc plotting."""
    plt.plot([0, 1], [0, 1], ls="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])


def make_plots_output_folder(model_path:str, output_folder:str):
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

    print(tcols.OKCYAN + f"Latent roc plots were saved to {plots_folder}." + tcols.ENDC)


def compute_auc(preds: np.array, targets: np.array) -> Tuple:
    """Compute the AUC for each prediction array, and then calculate the mean and
    stardard deviation of the aucs.

    Args:
        preds: Array of the predictions as computed by the vqc.
        targets: Array of the targets corresponding to the predicted data.

    Returns:
        The ROC curve coordiantes, the AUC, and the standard deviation on the AUC.
    """
    aucs = np.array([])
    for prd, trg in zip(preds, targets):
        fpr, tpr, thresholds = metrics.roc_curve(trg, prd)
        auc = metrics.roc_auc_score(trg, prd)
        aucs = np.append(aucs, auc)

    mean_auc = aucs.mean()
    std_auc = aucs.std()
    fpr, tpr, thresholds = metrics.roc_curve(targets.flatten(), preds.flatten())

    return fpr, tpr, mean_auc, std_auc


def sig_vs_bkg(data_sig, data_bkg, model_path, output_folder):
    """Plot the overlaid signal vs background given data.

    Args:
        data_sig      :: Numpy array of the signal data.
        data_bkg      :: Numpy array of the background data.
        model_path    :: String of path to a trained ae model.
        output_folder :: Folder where the figures are saved.
    """
    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    for i in range(data_sig.shape[1]):
        xmax = max(np.amax(data_sig[:, i]), np.amax(data_bkg[:, i]))
        xmin = min(np.amin(data_sig[:, i]), np.amin(data_bkg[:, i]))
        fig = plt.figure(figsize=(12, 10))

        hsig, _, _ = plt.hist(
            x=data_sig[:, i],
            density=1,
            range=(xmin, xmax),
            bins=50,
            alpha=0.8,
            histtype="step",
            linewidth=2.5,
            label="Sig",
            color="navy",
        )

        hbkg, _, _ = plt.hist(
            x=data_bkg[:, i],
            density=1,
            range=(xmin, xmax),
            bins=50,
            alpha=0.4,
            histtype="step",
            linewidth=2.5,
            label="Bkg",
            color="gray",
            hatch="xxx",
        )

        plt.legend()
        fig.savefig(plots_folder + "Feature " + str(i) + ".pdf")
        plt.close()

    print(f"Latent plots were saved to {plots_folder}.")

def get_arguments() -> dict:
    """
    Parses command line arguments and gives back a dictionary.
    
    Returns: Dictionary with the arguments
    """
    parser = argparse.ArgumentParser(formatter_class=
                                 argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_folder", type=str,
                        help="The folder where the data is stored on the system..")
    parser.add_argument("--norm", type=str,
                        help="The name of the normalisation that you'll to use.")
    parser.add_argument('--ae_model_path', type=str,
                        help="The path to the Auto-Encoder model.")
    parser.add_argument("--nevents", type=str,
                        help="The number of signal events of the norm file.")
    parser.add_argument('--nn_model_path', type=str,
                    help="The path to the saved NN model (.pt file).")
    parser.add_argument("--nvalid", type=int, default=-1,
                        help="The exact number of valid events used < nevents.")
    parser.add_argument("--ntest", type=int, default=-1,
                        help="The exact number of testing events used < nevents.")
    parser.add_argument("--kfolds", type=int, default=5,
                        help="Number of folds for the test.")                        
    args = parser.parse_args()
    
    seed = 12345
    args = {
        "data_folder": args.data_folder,
        "norm": args.norm,
        "nevents": args.nevents,
        "ae_model_path": args.ae_model_path,
        "nn_model_path": args.nn_model_path,
        "nvalid": args.nvalid,
        "ntest": args.ntest,
        "seed": seed,
        "kfolds": args.kfolds,
    }
    return args

def time_the_training(train: callable, *args):
    """Times the training of the neural network.

    Args:
        train (callable): The training method of the NeuralNetwork class.
        *args: Arguments for the train_model method.
    """
    train_time_start = time.perf_counter()
    train(*args)
    train_time_end = time.perf_counter()
    print(
        tcols.OKCYAN +
        f"Training completed in: {train_time_end-train_time_start:.2e} s or "
        f"{(train_time_end-train_time_start)/3600:.2e} h." + tcols.ENDC
    )

if __name__ == "__main__":
    main()