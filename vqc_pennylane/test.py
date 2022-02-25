# The VQC testing script. Here, a vqc is imported and data is passed through it.
# The results are quantified in terms of AUC.
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
        test_events=args["test_events"],
        valid_events=args["valid_events"],
        seed=args["seed"],
    )
    outdir = args["vqc_path"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hyperparams_file = os.path.join(args["vqc_path"], "hyperparameters.json")
    vqc_hyperparams = util.import_hyperparams(hyperparams_file)
    model = get_model(qdata, vqc_hyperparams, args)
    model.load_model(args["vqc_path"])
    util.print_model_info(args["ae_model_path"], qdata, model)

    valid_loader, test_loader = get_data(qdata, args)
    valid_pred = model.predict(valid_loader[0])
    test_pred = model.predict(test_loader[0])

    print("\n----------------------------------")
    print("VALID LOSS:")
    print(model.compute_loss(valid_pred, valid_loader[1]))
    print("TEST LOSS:")
    print(model.compute_loss(test_pred, test_loader[1]))
    print("----------------------------------\n")

    roc_plots(test_pred, test_loader[1], args["ae_model_path"], "vqc_roc_plot")

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
    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    return plots_folder

def roc_plots(data, target, model_path, output_folder):
    """Plot the ROC of a whole data set, for each feature.

    Args:
        data: Data to calculate the AUC on. Can have one or multiple features.
        target: Target corresponding to the data.
        model_path: Path to a trained ae model.
        output_folder: Name of the output folder to save the plots in.
    """
    plots_folder = make_plots_output_folder(model_path, output_folder)
    set_plotting_misc()

    for feature in range(data.shape[1]):
        fpr, tpr, mean_auc, std_auc = compute_auc(data, target, feature)
        fig = plt.figure(figsize=(12, 10))
        roc_plot_misc()

        plt.plot(fpr, tpr, label=f"AUC: {mean_auc:.3f} ± {std_auc:.3f}", color="navy")
        plt.legend()

        fig.savefig(plots_folder + f"Feature {feature}.pdf")
        plt.close()

    print(f"Latent roc plots were saved to {plots_folder}.")

def compute_auc(data, target, feature):
    """Split a data set into 5, compute the AUC for each, and then calculate the mean
    and stardard deviation of these.

    Args:
        data: Numpy array of the whole data (all features).
        target: Numpy array of the target.
        feature: The number of the feature to compute the AUC for.

    Returns:
        The ROC curve coordiantes, the AUC, and the standard deviation on the AUC.
    """
    data, target = shuffle(data, target, random_state=0)
    data_chunks = np.array_split(data, 5)
    target_chunks = np.array_split(target, 5)

    aucs = []
    for dat, trg in zip(data_chunks, target_chunks):
        fpr, tpr, thresholds = metrics.roc_curve(trg, dat[:, feature])
        auc = metrics.roc_auc_score(trg, dat[:, feature])
        aucs.append(auc)

    aucs = np.array(aucs)
    mean_auc = aucs.mean()
    std_auc = aucs.std()
    fpr, tpr, thresholds = metrics.roc_curve(target, data[:, feature])

    return fpr, tpr, mean_auc, std_auc

def get_model(vqc_hyperparams, args) -> Tuple:
    """Choose the type of VQC to train. The normal vqc takes the latent space
    data produced by a chosen auto-encoder. The hybrid vqc takes the same
    data that an auto-encoder would take, since it has an encoder or a full
    auto-encoder attached to it.

    Args:
        vqc_hyperparams (dict): Dictionary of the vqc hyperparameters dictating
            the circuit structure and the training.
        *args: Dictionary of hyperparameters related more loosely to the vqc, such as
            which device to run on.

    Returns:
        An instance of the vqc object with the given specifications (hyperparams).
    """
    qdevice = util.get_qdevice(
        args["run_type"],
        wires=args["nqubits"],
        backend_name=args["backend_name"],
        config=args["config"],
    )

    if args["hybrid_training"]:
        vqc_hybrid = VQCHybrid(qdevice, device="cpu", hpars=vqc_hyperparams)
        return vqc_hybrid

    vqc = VQC(qdevice, vqc_hyperparams)
    return vqc

def get_data(qdata, args):
    """Load the appropriate data depending on the type of vqc that is used.

    Args:
        qdata (object): Class object with the loaded data.
        *args: Dictionary containing specifications relating to the data, such as the
            batch size, whether the data for the hybrid vqc or the normal vqc should be
            loaded, and so on.

    Returns:
        The validation and test data tailored to the type of vqc that one
        is testing with this script.
    """
    if args["hybrid_vqc"]:
        return *get_hybrid_test_data(qdata, args)

    return *get_nonhybrid_test_data(qdata, args)

def get_nonhybrid_test_data(qdata, args) -> Tuple:
    """Loads the data from pre-trained autoencoder latent space when we have non
    hybrid VQC testing.
    """
    valid_features = qdata.get_latent_space("valid")
    valid_labels = qdata.ae_data.vatarget
    valid_loader = [valid_features, valid_labels]

    test_features = qdata.get_latent_space("test"), args["batch_size"]
    test_labels = qdata.ae_data.tetarget, args["batch_size"]
    test_loader = [test_features, test_labels]

    return valid_loader, test_loader

def get_hybrid_test_data(qdata, args) -> Tuple:
    """Loads the raw input data for hybrid testing."""
    valid_loader = qdata.ae_data.get_loader("valid", "cpu", shuffle=True)
    test_loader = qdata.ae_data.get_loader("test", "cpu", args["batch_size"], True)

    return valid_loader, test_loader
