# Imports two .npy files as constructed in prepare_mlready_data.npy or any two
# .npy files that contain 2D matrices with number of events and features for
# signal and background, respectively.
# An equal number of signal and background events are selected and then
# sklearn is used to normalize the data sets. Finally, the normalised data
# is split into training, validation, and testing subsets.

import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import train_test_split
from sklearn import metrics

from terminal_colors import tcols

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--data_sig",
    type=str,
    required=True,
    help="Path to the .npy file containing sig data.",
)
parser.add_argument(
    "--data_bkg",
    type=str,
    required=True,
    help="Path to the .npy file containing bkg data.",
)
parser.add_argument(
    "--norm_name",
    type=str,
    required=True,
    help="Choose which way to normalise the data."
    "Available options: vanilla (no norm), minmax, "
    "maxabs, std, robust, power, quantile.",
)
parser.add_argument(
    "--maxdata",
    type=int,
    default=-1,
    help="The maximum train samples to use for signal. " "bkg will be equal.",
)
parser.add_argument(
    "--valid_percent",
    type=float,
    default=-1,
    help="Percentage of total data that will make "
    "validation and test data."
    "These two data sets will be equal in size."
    "Example: if you give 0.1 then 10\% of the data will"
    "be validation, 10\% of the data will be test, and 80\%"
    "will be training.",
)


def main():
    print(tcols.HEADER)
    print(f"Normalizing the sig data file: {args.data_sig}")
    print(f"Normalizing the bkg data file: {args.data_bkg}")
    print(tcols.ENDC)

    data_sig = np.load(args.data_sig)[: args.maxdata, :]
    data_bkg = np.load(args.data_bkg)[: args.maxdata, :]

    all_data = np.vstack((data_sig, data_bkg))
    all_targ = np.concatenate((np.ones(args.maxdata), np.zeros(args.maxdata)))

    choose_norm(args.norm_name, all_data, all_targ)


def choose_norm(norm_name, all_data, all_targ):
    """
    Normalise the mlready data.
    @norm_name :: Name of the normalisation applied, chosen by user.
    """
    print(tcols.OKBLUE)
    if norm_name == "vanilla":
        print("\nSaving all (raw) no norm data...")
        split_and_save(all_data, all_targ, f"no_norm_{args.maxdata}")
    elif norm_name == "minmax":
        print("\nApplying minmax normalization...")
        apply_norm(MinMaxScaler(), f"minmax_{args.maxdata:.2e}", all_data, all_targ)
    elif norm_name == "maxabs":
        print("\nApplying maxabs normalization...")
        apply_norm(MaxAbsScaler(), f"maxabs_{args.maxdata:.2e}", all_data, all_targ)
    elif norm_name == "std":
        print("\nApplying standard normalization...")
        apply_norm(StandardScaler(), f"std_{args.maxdata:.2e}", all_data, all_targ)
    elif norm_name == "robust":
        print("\nApplying robust normalization...")
        apply_norm(RobustScaler(), f"robust_{args.maxdata:.2e}", all_data, all_targ)
    elif norm_name == "power":
        print("\nApplying power normalization...")
        apply_norm(PowerTransformer(), f"power_{args.maxdata:.2e}", all_data, all_targ)
    elif norm_name == "quantile":
        print("\nApplying quantile normalization...")
        apply_norm(
            QuantileTransformer(),
            f"quantile_{args.maxdata:.2e}",
            all_data,
            all_targ,
        )
    else:
        print(tcols.ENDC)
        raise TypeError("Specified normalisation type does not exist!!!")

    print(tcols.ENDC)


def apply_norm(norm_method, norm_name, data, target):
    """
    Apply a specific normalization to the whole data set. Save the
    normalized data set into a separate numpy array specifying the
    norm name.
    @norm_method :: The normalization method sklearn object.
    @norm_name   :: String of the normalization name.
    @data        :: 2D numpy array of the features to be normalized.
    @target      :: 1D numpy array with 0 for each bkg event and 1 for
                    the sig.
    """
    if check_norm_exists(norm_name):
        return
    data_norm = norm_method.fit_transform(data)
    plot_roc_auc(data_norm, target, norm_name)
    sig_mask = target == 1
    bkg_mask = target == 0
    plot_sig_bkg(data_norm[sig_mask, :], data_norm[bkg_mask, :], norm_name)
    split_and_save(data_norm, target, norm_name)


def split_and_save(data, target, name):
    """
    Splits a given data set into training, testing, and validation
    samples that then are saved with corresponding names in .npy files.
    @data   :: 2D numpy array of the features to be normalized.
    @target :: 1D numpy array with 0 for each bkg event and 1 for sig.
    @name   :: String containing the name of the normalisation and the
        number of total signal/background events in the data object.

    """
    save_dir = os.path.dirname(args.data_sig) + "/"
    print(tcols.OKBLUE)
    print("Splitting data into training, validation, and testing sets.")
    print(tcols.ENDC)

    x_train, x_valid, y_train, y_valid = train_test_split(
        data,
        target,
        test_size=args.valid_percent,
        shuffle=True,
        stratify=target,
    )
    test_percent = float(x_valid.shape[0] / x_train.shape[0])
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=test_percent, shuffle=True, stratify=y_train
    )

    print(tcols.OKGREEN)
    print("Saving data to: ", save_dir)
    print(tcols.ENDC)
    np.save(save_dir + "x_data_" + name + "_train", x_train)
    np.save(save_dir + "x_data_" + name + "_test", x_test)
    np.save(save_dir + "x_data_" + name + "_valid", x_valid)

    np.save(save_dir + "y_data_" + name + "_train", y_train)
    np.save(save_dir + "y_data_" + name + "_test", y_test)
    np.save(save_dir + "y_data_" + name + "_valid", y_valid)


def plot_sig_bkg(input_sig, input_bkg, norm_name):
    """
    Plot a histogram for the signal normalized data and a histogram
    for the background normalized data and then overlay them.
    @input_sig :: 2D numpy array containing the signal data.
    @input_bkg :: 2D numpy array containing the background data.
    @norm_name :: String name of the normalization that was applied.

    returns :: Saves the plots in a folder where the signal data is
        located.
    """
    save_dir = os.path.dirname(args.data_sig) + "/"
    sig_bkg_plot_dir = save_dir + "sig_bkg_" + norm_name + "_plots/"
    if not os.path.exists(sig_bkg_plot_dir):
        os.makedirs(sig_bkg_plot_dir)

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    for idx in range(input_sig.shape[1]):

        plt.figure(figsize=(12, 10))
        plt.xlim(np.amin(input_bkg[:, idx]), np.amax(input_bkg[:, idx]))
        plt.hist(
            x=input_bkg[:, idx],
            bins=60,
            alpha=0.8,
            histtype="step",
            linewidth=2.5,
            label="Background",
            density=True,
            color="gray",
            hatch="xx",
        )
        plt.hist(
            x=input_sig[:, idx],
            bins=60,
            alpha=0.8,
            histtype="step",
            linewidth=2.5,
            label="Signal",
            density=True,
            color="navy",
        )
        plt.xlabel(varname(idx))
        plt.ylabel("Density")
        plt.gca().set_yscale("log")
        plt.legend()
        plt.savefig(sig_bkg_plot_dir + varname(idx) + ".pdf")
        plt.close()


def plot_roc_auc(data, target, norm_name):
    """
    Compute the roc curve of a given 2D dataset of features.
    @data      :: 2D array, each column is a feature and each row an
                  event.
    @target    :: 1D array, each element is 0 or 1 corresponding to bkg
                  or sig.
    @norm_name :: String of the name of the normalization.

    returns :: Prints and saves the the roc curve along with an indication
        of the AUC on top of it.
    """
    print(f"Plotting the ROC curves for {norm_name} normalization...")
    save_dir = os.path.dirname(args.data_sig) + "/"
    roc_auc_plot_dir = save_dir + "roc_" + norm_name + "_plots/"

    if not os.path.exists(roc_auc_plot_dir):
        os.makedirs(roc_auc_plot_dir)

    plt.rc("xtick", labelsize=23)
    plt.rc("ytick", labelsize=23)
    plt.rc("axes", titlesize=25)
    plt.rc("axes", labelsize=25)
    plt.rc("legend", fontsize=22)

    auc_sum = 0.0
    for feature in range(data.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(target, data[:, feature])
        auc = metrics.roc_auc_score(target, data[:, feature])

        fig = plt.figure(figsize=(12, 10))
        plt.title(varname(feature))
        plt.plot(fpr, tpr, label=f"AUC: {auc}", color="navy")
        plt.plot([0, 1], [0, 1], ls="--", color="gray")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.legend()

        auc_sum += auc
        fig.savefig(roc_auc_plot_dir + varname(feature) + ".png")
        plt.close()

    with open(roc_auc_plot_dir + "auc_sum.txt", "w") as auc_sum_file:
        auc_sum_file.write(f"{auc_sum:.3f}")


def check_norm_exists(norm_name) -> bool:
    """
    Quick helper method that checks if the normalise data files already
    exists and returns 1 if it does but 0 if it does not.
    @norm_name :: The name of the normalisation.
    """
    if glob.glob(os.path.dirname(args.data_sig) + "/*" + norm_name + "*.npy"):
        print(
            f'\033[93mWarning: Files with name ∋ "{norm_name}" '
            "exists so not producing it again.\033[0m"
        )
        return 1
    return 0


def varname(index):
    """
    The variables in the currently analysed arrays follow an order.
    This method gets the name of the variable given its data index.
    @index :: The index for which the name is returned.

    returns :: None if the index is not valid or string of the variable name
        if the given index is valid.
    """
    jet_feats = [
        "$p_t$",
        "$\\eta$",
        "$\\phi$",
        "Energy",
        "$p_x$",
        "$p_y$",
        "$p_z$",
        # "btag",
    ]
    jet_nvars = len(jet_feats)
    num_jets = 7
    met_feats = ["$\\phi$", "$p_t$", "$p_x$", "$p_y$"]
    met_nvars = len(met_feats)
    lep_feats = [
        "$p_t$",
        "$\\eta$",
        "$\\phi$",
        "Energy",
        "$p_x$",
        "$p_y$",
        "$p_z$",
    ]
    lep_nvars = len(lep_feats)

    if index < jet_nvars * num_jets:
        jet = index // jet_nvars + 1
        var = index % jet_nvars
        varstring = "Jet " + str(jet) + " " + jet_feats[var]
        return varstring
    index -= jet_nvars * num_jets

    if index < met_nvars:
        var = index % met_nvars
        varstring = "MET " + met_feats[var]
        return varstring
    index -= met_nvars

    if index < lep_nvars:
        var = index % lep_nvars
        varstring = "Lepton " + lep_feats[var]
        return varstring

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    main()
