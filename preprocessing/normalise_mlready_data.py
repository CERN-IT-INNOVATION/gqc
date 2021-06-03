# Imports an .npy file as constructed in prepare_mlready_data.npy or any
# .npy file that contains a 2D matrix with number of events and features.
# Sklearn is then used to normalize the data sets. Each normalized copy
# is saved as an .npy file. There's also the option of splitting the
# data sets into training and testing subsamples.
import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_sig", type=str, required=True,
    help="Path to the .npy file containing sig data to be normalized.")
parser.add_argument("--data_bkg", type=str, required=True,
    help="Path to the .npy file containing bkg data to be normalized.")
parser.add_argument('--maxdata', type=int, default=-1,
    help='The maximum number of training samples to use.')

def varname(index):
    # Gets the name of what variable is currently considered based on the index
    # in the data.
    jet_feats=["$p_t$","$\\eta$","$\\phi$","Energy","$p_x$","$p_y$","$p_z$",
        "btag"]
    jet_nvars=len(jet_feats); num_jets = 7
    met_feats=["$\\phi$","$p_t$","$p_x$","$p_y$"]
    met_nvars=len(met_feats)
    lep_feats=["$p_t$","$\\eta$","$\\phi$","Energy","$p_x$","$p_y$","$p_z$"]
    lep_nvars=len(lep_feats)

    if (index < jet_nvars * num_jets):
        jet = index // jet_nvars + 1
        var = index % jet_nvars
        varstring = "Jet " + str(jet) + " " + jet_feats[var]
        return varstring
    index -= jet_nvars * num_jets;

    if (index < met_nvars):
        var = index % met_nvars;
        varstring = "MET " + met_feats[var];
        return varstring
    index -= met_nvars;

    if (index < lep_nvars):
        var = index % lep_nvars
        varstring = "Lepton " + lep_feats[var]
        return varstring;

    return None

def split_and_save(data, target, name):
    # Splits a given data set into training, testing, and validation samples
    # that then are saved with corresponding names in .npy files.
    save_dir = os.path.dirname(args.data_sig) + "/"
    print("Splitting data into training, validation, and testing sets.")
    x_train, x_valid, y_train, y_valid = train_test_split(data, target,
        test_size=0.1, shuffle=True)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
        test_size=0.111111, shuffle=True)

    print("Saving data to: ", save_dir)
    np.save(save_dir + "x_data_" + name + "_train", x_train)
    np.save(save_dir + "x_data_" + name + "_test",  x_test)
    np.save(save_dir + "x_data_" + name + "_valid", x_valid)

    np.save(save_dir + "y_data_" + name + "_train", y_train)
    np.save(save_dir + "y_data_" + name + "_test",  y_test)
    np.save(save_dir + "y_data_" + name + "_valid", y_valid)

def plot_sig_bkg(input_sig, input_bkg, norm_name):
    # Plot sig and bkg overlaid for a dataset.
    save_dir = os.path.dirname(args.data_sig) + "/"
    if not os.path.exists(save_dir + 'sig_bkg_' + norm_name + '_plots/'):
        os.makedirs(save_dir + 'sig_bkg_' + norm_name + '_plots/')

    plt.rc('xtick', labelsize=23)
    plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    for idx in range(input_sig.shape[1]):

        plt.figure(figsize=(12,10))
        plt.xlim(np.amin(input_bkg[:,idx]), np.amax(input_bkg[:,idx]))
        plt.hist(x=input_bkg[:,idx], bins=60, alpha=0.8, histtype='step',
            linewidth=2.5, label='Background', density=True, color='gray',
            hatch='xx')
        plt.hist(x=input_sig[:,idx], bins=60, alpha=0.8, histtype='step',
            linewidth=2.5, label='Signal', density=True, color='navy')
        plt.xlabel(varname(idx)); plt.ylabel('Density')
        plt.gca().set_yscale("log")
        plt.legend()
        plt.savefig(save_dir + 'sig_bkg_' + norm_name + '_plots/' +
            varname(idx) + '.pdf')
        plt.close()

def apply_norm(norm_method, norm_name, data, target, maxdata):
    """
    Apply a specific normalization to the whole data set. Save the normalized
    data set into a separate numpy array with a specific name.

    @norm_method :: The normalization method object.
    @norm_name   :: The normalization name.
    @data        :: The numpy array data to be normalized.
    @target      :: The corresponding target array for the data.
    @base_fn     :: The base filename of the raw data.
    """
    norm_name = norm_name + "_{:.2e}".format(maxdata)
    if check_norm_exists(norm_name): return
    data_norm = norm_method.fit_transform(data)
    sig_mask = (target == 1); bkg_mask = (target == 0)
    plot_sig_bkg(data_norm[sig_mask, :], data_norm[bkg_mask, :], norm_name)
    split_and_save(data_norm, target, norm_name)

def normalize_jet(data, jet_nb):
    # Normalize the jet features for a specific jet. For optimal normalization.
    pt = 0 + jet_nb*8; eta = pt+1;
    en = 3 + jet_nb*8; phi = en  ; px = en+1; pz = en+4;
    power_trans = PowerTransformer(method='box-cox')
    data[:,pt][data[:,pt] == 0] = 1e-8
    data[:,en][data[:,en] == 0] = 1e-8
    data[:,pt]  = MinMaxScaler().fit_transform(data[:,pt].reshape(-1, 1))[:,0]
    data[:,en]  = MinMaxScaler().fit_transform(data[:,en].reshape(-1, 1))[:,0]
    data[:,eta:phi] = MaxAbsScaler().fit_transform(data[:,pt+1:en])
    data[:,px:pz]   = MaxAbsScaler().fit_transform(data[:,en+1:en+4])

    return data

def normalize_met(data):
    # Normalize the metadata features. For the optimal normalization.
    phi = 7*8; pt = phi+1; px=phi+2; py=phi+3;
    power_trans = PowerTransformer(method='box-cox')
    data[:,phi] = MaxAbsScaler().fit_transform(data[:,phi].reshape(-1,1))[:,0]
    data[:,pt]  = MinMaxScaler().fit_transform(data[:,pt].reshape(-1,1))[:,0]
    data[:,px]  = MaxAbsScaler().fit_transform(data[:,px].reshape(-1,1))[:,0]
    data[:,py]  = MaxAbsScaler().fit_transform(data[:,py].reshape(-1,1))[:,0]

    return data

def normalize_lep(data):
    # Normalize the lepton features. For the optimal normalization.
    pt = 7*8+4; eta = pt+1; phi= pt+2; en = pt+3
    px = pt+4; py = pt+5; pz = pt+6
    power_trans = PowerTransformer(method='box-cox')
    data[:,pt]  = MinMaxScaler().fit_transform(data[:,pt].reshape(-1,1))[:,0]
    data[:,eta] = MaxAbsScaler().fit_transform(data[:,eta].reshape(-1,1))[:,0]
    data[:,phi] = MaxAbsScaler().fit_transform(data[:,phi].reshape(-1,1))[:,0]
    data[:,en]  = MinMaxScaler().fit_transform(data[:,en].reshape(-1,1))[:,0]
    data[:,px]  = MaxAbsScaler().fit_transform(data[:,px].reshape(-1,1))[:,0]
    data[:,py]  = MaxAbsScaler().fit_transform(data[:,py].reshape(-1,1))[:,0]
    data[:,pz]  = MaxAbsScaler().fit_transform(data[:,pz].reshape(-1,1))[:,0]

    return data

def apply_optimal_norm(data, target, maxdata):
    # Apply normalization specific to the ttHbb data that we are analyzing.
    norm_name = "optimal_norm_{:.2e}".format(maxdata)
    if check_norm_exists(norm_name): return
    data_norm = data
    for jet_nb in range(8): data_norm = normalize_jet(data_norm, jet_nb)
    data_norm = normalize_met(data_norm)
    data_norm = normalize_lep(data_norm)
    sig_mask = (target == 1); bkg_mask = (target == 0)
    plot_sig_bkg(data_norm[sig_mask, :], data_norm[bkg_mask, :], norm_name)
    split_and_save(data_norm, target, norm_name)

def check_norm_exists(norm_name):
    if glob.glob(os.path.dirname(args.data_sig) + '/*' + norm_name + "*.npy"):
        print("\033[93mWarning: Files with name ∋ \"" + norm_name + "\" "
              "exists so not producing it again.\033[0m")
        return 1
    return 0

if __name__ == "__main__":
    args = parser.parse_args()
    print('Normalizing the sig data file: ' + args.data_sig)
    print('Normalizing the bkg data file: ' + args.data_bkg)

    data_sig = np.load(args.data_sig)[:args.maxdata, :]
    data_bkg = np.load(args.data_bkg)[:args.maxdata, :]
    all_data = np.vstack((data_sig, data_bkg))
    all_targ = np.concatenate((np.ones(args.maxdata), np.zeros(args.maxdata)))

    print("\n\033[92mApplying minmax normalization...\033[0m")
    apply_norm(MinMaxScaler(), "minmax_norm", all_data, all_targ, args.maxdata)
    print("\n\033[92mApplying standard normalization...\033[0m")
    apply_norm(StandardScaler(), "std_norm", all_data, all_targ, args.maxdata)
    print("\n\033[92mApplying quantile normalization...\033[0m")
    apply_norm(QuantileTransformer(), "quant_norm", all_data, all_targ,
        args.maxdata)
    print("\n\033[92mApplying quantile gauss normalization...\033[0m")
    apply_norm(QuantileTransformer(output_distribution='normal'),"qgauss_norm",
        all_data, all_targ, args.maxdata)

    print("\n\033[92mApplying optimal normalization...\033[0m")
    apply_optimal_norm(all_data, all_targ, args.maxdata)
