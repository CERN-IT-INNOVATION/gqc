# The VQC testing script. Here, a vqc is imported and data is passed
# through it. The results are quantified in terms of AUC.
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
    outdir = args["vqc_dir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hyperparams_file = os.path.join(args["vqc_dir"], "hyperparameters.json")
    vqc_hyperparams = util.import_hyperparams(hyperparams_file)
    model = get_model(qdata, vqc_hyperparams, args)
    model.load_model(args["vqc_dir"])
    util.print_model_info(args["ae_model_path"], qdata, model)

    valid_loader, test_loader = get_data(qdata, args)

def roc_plots(sig, bkg, model_path, output_folder):
    """Plot the ROC of a whole data set, for each feature, and then save the
    sum of the AUCs of all the features to a text file.

    Args:
        sig           :: Numpy array containing the signal data.
        bkg           :: Numpy array containing the background data.
        model_path    :: String of the path to a trained ae model.
        output_folder :: String of the name to the output folder to save plots.
    """
    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    data = np.vstack((sig, bkg))
    target = np.concatenate((np.ones(sig.shape[0]), np.zeros(bkg.shape[0])))

    auc_sum = 0.0
    for feature in range(data.shape[1]):
        fpr, tpr, mean_auc, std_auc = compute_auc(data, target, feature)
        fig = plt.figure(figsize=(12, 10))
        plt.plot(fpr, tpr, label=f"AUC: {mean_auc:.3f} ± {std_auc:.3f}", color="navy")
        plt.plot([0, 1], [0, 1], ls="--", color="gray")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend()

        auc_sum += mean_auc
        fig.savefig(plots_folder + f"Feature {feature}.pdf")
        plt.close()

    with open(plots_folder + "auc_sum.txt", "w") as auc_sum_file:
        auc_sum_file.write(f"{auc_sum:.3f}")

    print(f"Latent roc plots were saved to {plots_folder}.")

def compute_auc(data, target, feature) -> tuple:
    """
    Split a data set into 5, compute the AUC for each, and then calculate the
    mean and stardard deviation of these.
    @data    :: Numpy array of the whole data (all features).
    @target  :: Numpy array of the target.
    @feature :: The number of the feature to compute the AUC for.

    returns :: The ROC curve coordiantes, the AUC, and the standard deviation
        on the AUC.
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
    auto-encoder attached to it..

    Args:
        vqc_hyperparams (dict): Dictionary of the vqc hyperparameters dictating
            the circuit structure and the training.
        *args: Dictionary of hyperparameters to give to the vqc, (they can also be a
            subset of this dictionary).

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
        *args: Dictionary of hyperparameters to give to the vqc, (they can also be a
            subset of this dictionary).

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
