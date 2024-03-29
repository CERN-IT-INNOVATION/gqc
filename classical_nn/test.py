# The classical NN testing script. Here, a trained NN is imported
# and data is passed through it.The results are quantified in terms of ROCs and AUCs.

import os
import sys

sys.path.append("..")
import time
import argparse
from typing import Tuple, Union

import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.utils import shuffle

from autoencoders import util as ae_util
from vqc_pennylane import util
from vqc_pennylane import qdata as qd
from vqc_pennylane.terminal_colors import tcols
from neural_network import NeuralNetwork


def main(args: dict):
    print(
        tcols.OKCYAN
        + "\n\nTesting the fully connected feed-forward neural network..."
        + tcols.ENDC
    )
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
    valid_preds = np.array([model.predict(x)[-1].squeeze() for x in x_valid])
    test_preds = np.array([model.predict(x)[-1].squeeze() for x in x_test])

    model_dir = os.path.dirname(args["nn_model_path"])
    auc = roc_plots(test_preds, y_test, model_dir, "roc_plot")

    x_test_sig, x_test_bkg = qdata.ae_data.split_sig_bkg(x_test, y_test)
    sig = model.predict(x_test_sig)
    bkg = model.predict(x_test_bkg)
    latent_roc_plot(sig, bkg, model_dir, "latent_plots")
    sig_vs_bkg(sig[0], bkg[0], args["nn_model_path"], "latent_plots")

    return auc


def latent_roc_plot(sig: np.ndarray, bkg: np.ndarray, dir: str, name: str):
    """Creates the data folds and computes the ROC and AUC of the individual features
    in the latent space. Saves the output plots to file.

    Args:
        sig: Array of shape (kfolds, n_test, n_features) the latent feature
             distributions of signal samples.
        bkg: Array of shape (kfolds, n_test, n_features) the latent feature
             distributions of background samples.
        dir: The directory in which to save the plots.
        name: The name of the folder that contains the plots.
    """
    features = np.vstack((sig[0], bkg[0]))
    labels = np.concatenate((np.ones(sig[0].shape[0]), np.zeros(bkg[0].shape[0])))

    features, labels = shuffle(features, labels, random_state=0)
    features_folds = np.array(np.array_split(features, 5))
    labels_folds = np.array(np.array_split(labels, 5))

    print("Computing ROCs of the latent space variables... ", end="")
    for ifeature in range(features_folds.shape[2]):
        roc_plots(
            features_folds[:, :, ifeature],
            labels_folds,
            dir,
            name,
            f"roc_{ifeature}.pdf",
        )
    print(tcols.OKGREEN + "Done." + tcols.ENDC)


def get_hparams_for_testing(args: dict):
    """Imports the hyperparameters of the vqc at the given path and sets the
    optimiser to none such that no optimiser is loaded (none is needed since no
    training happens within the scope of this script).

    Args:
        args: Dictionary of hyperparameters for the vqc.

    Returns:
        Updated args dictionary with the loaded vqc hyperparameters.
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


def make_plots_output_folder(model_path: str, output_folder: str):
    """Make the output folder of the plots.

    Args:
        model_path: Path to a trained model.
        output_folder: Name of the output folder to save the plots in.

    Returns:
        Full path of the output folder to save the plots in.
    """
    plots_folder = os.path.join(model_path, output_folder)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    return plots_folder


def roc_plots(
    scores: np.ndarray,
    target: np.ndarray,
    model_path: str,
    output_folder: str,
    file_name: str = "roc_curve.pdf",
) -> Union[tuple, None]:
    """Plot the ROC of the vqc predictions.

    Args:
        scores: Score predictions of the vqc for a data set of
                shape (kfolds, n_test,).
        target: Target corresponding to the data of shape (kfolds, n_test,).
        model_path: Path to a trained model.
        output_folder: Name of the output folder to save the plots in.

    Returns: The mean the std of the AUC of the model or None if the method is used
             for computing the ROCs and AUC of individual features.
    """
    plots_folder = make_plots_output_folder(model_path, output_folder)
    set_plotting_misc()
    fpr, tpr, mean_auc, std_auc = compute_auc(scores, target, plots_folder)
    fig = plt.figure(figsize=(12, 10))
    roc_plot_misc()

    plt.plot(fpr, tpr, label=f"AUC: {mean_auc:.3f} ± {std_auc:.3f}", color="navy")
    plt.legend()

    fig.savefig(os.path.join(plots_folder, file_name))
    plt.close()

    if file_name == "roc_curve.pdf":  # To not print n_feature times for latent rocs
        print(tcols.OKCYAN + f"ROC plots were saved to {plots_folder}." + tcols.ENDC)
        print(
            "\nMean AUC accross the folds: "
            + tcols.BOLD
            + f"{mean_auc:.3f} ± {std_auc:.3f}"
            + tcols.ENDC
        )
        return mean_auc, std_auc
    return None


def find_nearest(array: np.ndarray, value: float):
    """Finds the index of the nearest value in an array to a given value."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def compute_auc(scores: np.array, targets: np.array, outdir: str) -> Tuple:
    """Compute the AUC for each prediction array, and then calculate the mean and
    stardard deviation of the aucs.

    Args:
        scores: Array of the predictions as computed by the vqc.
        targets: Array of the targets corresponding to the predicted data.

    Returns:
        The ROC curve coordiantes, the AUC, and the standard deviation on the AUC.
    """
    aucs = np.array([])
    fprs_at_tprs = np.array([])
    tpr_baseline = np.linspace(0.025, 0.99, 100)
    fold = 0
    for prd, trg in zip(scores, targets):
        fold += 1
        fpr, tpr, thresholds = metrics.roc_curve(trg, prd)
        fpr_baseline = np.interp(tpr_baseline, tpr, fpr)
        fpr_baseline.astype("float32").tofile(os.path.join(outdir, f"fpr_{fold}.dat"))
        tpr_baseline.astype("float32").tofile(os.path.join(outdir, f"tpr_{fold}.dat"))
        tpr_idx = find_nearest(tpr, 0.8)
        fprs_at_tprs = np.append(fprs_at_tprs, fpr[tpr_idx])
        auc = metrics.roc_auc_score(trg, prd)
        aucs = np.append(aucs, auc)

    fprs_at_tprs.astype("float32").tofile(os.path.join(outdir, "fprs_at_tprs.dat"))
    aucs.astype("float32").tofile(os.path.join(outdir, "aucs.dat"))
    mean_auc = aucs.mean()
    std_auc = aucs.std()
    fpr, tpr, thresholds = metrics.roc_curve(targets.flatten(), scores.flatten())

    return fpr, tpr, mean_auc, std_auc


def sig_vs_bkg(
    data_sig: np.ndarray, data_bkg: np.ndarray, model_path: str, output_folder: str
):
    """Plot the overlaid signal vs background given data.

    Args:
        data_sig: Numpy array of the signal data.
        data_bkg: Numpy array of the background data.
        model_path: String of path to a trained ae model.
        output_folder: Folder where the figures are saved.
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        help="The folder where the data is stored on the system..",
    )
    parser.add_argument(
        "--norm", type=str, help="The name of the normalisation that you'll to use."
    )
    parser.add_argument(
        "--ae_model_path", type=str, help="The path to the Auto-Encoder model."
    )
    parser.add_argument(
        "--nevents", type=str, help="The number of signal events of the norm file."
    )
    parser.add_argument(
        "--nn_model_path", type=str, help="The path to the saved NN model (.pt file)."
    )
    parser.add_argument(
        "--nvalid",
        type=int,
        default=-1,
        help="The exact number of valid events used < nevents.",
    )
    parser.add_argument(
        "--ntest",
        type=int,
        default=-1,
        help="The exact number of testing events used < nevents.",
    )
    parser.add_argument(
        "--kfolds", type=int, default=5, help="Number of folds for the test."
    )
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
        tcols.OKCYAN
        + f"Training completed in: {train_time_end-train_time_start:.2e} s or "
        f"{(train_time_end-train_time_start)/3600:.2e} h." + tcols.ENDC
    )


if __name__ == "__main__":
    args = get_arguments()
    main(args)
