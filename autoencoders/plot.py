# Plot different figures related to the autoencoder such as the latent
# space variables, the ROC curves of the latent space variables, etc.
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.utils import shuffle

from . import util
from . import data


def main(args):
    device = 'cpu'
    model_folder = os.path.dirname(args['model_path'])
    hp_file = os.path.join(model_folder, 'hyperparameters.json')
    hp = util.import_hyperparams(hp_file)

    # Data loading.
    ae_data = data.AE_data(args['data_folder'], args['norm'], args['nevents'])
    test_sig, test_bkg = \
        ae_data.split_sig_bkg(ae_data.test_data, ae_data.test_target)

    # AE model loading.
    model = util.choose_ae_model(hp['ae_type'], device, hp)
    model.load_model(args['model_path'])

    print('\n----------------------------------')
    print("VALID LOSS:")
    print(model.compute_loss(ae_data.valid_data, ae_data.valid_target).item())
    print("TEST LOSS:")
    print(model.compute_loss(ae_data.test_data, ae_data.test_target).item())
    print('----------------------------------\n')

    sig = model.predict(test_sig)
    bkg = model.predict(test_bkg)

    sig_vs_bkg(sig[0], bkg[0], args['model_path'], 'latent_plots')
    roc_plots(sig[0], bkg[0], args['model_path'], 'latent_roc')
    input_reco(test_sig, test_bkg, sig[1], bkg[1], args['model_path'])

    if len(sig) == 3:
        roc_plots(sig[2], bkg[2], args['model_path'], 'classif_roc')


def input_reco(input_sig, input_bkg, recon_sig, recon_bkg, model_path):
    """
    Plots the input data overlaid with the reconstruction data for both sig
    and bkg on the same plot.
    @input_sig  :: Numpy array containing the input signal data.
    @input_bkg  :: Numpy array containing the input background data.
    @recon_sig  :: Numpy array containing the reconstructed signal data.
    @recon_bkg  :: Numpy array containing the reconstructed background data.
    @model_path :: String containing the path to where the model is saved.
    """
    plots_folder = os.path.dirname(model_path) + '/input_vs_reco/'
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    for idx in range(input_sig.shape[1]):
        plt.figure(figsize=(12, 10))

        input_vs_reco(input_bkg[:, idx], recon_bkg[:, idx], idx, 'gray',
                      class_label='Background')
        input_vs_reco(input_sig[:,idx], recon_sig[:,idx], idx, 'navy',
                      class_label='Signal')

        plt.savefig(plots_folder + util.varname(idx) + '.pdf')
        plt.close()

    print(f"Input vs reco plots were saved to {plots_folder}.")


def input_vs_reco(input_data, recon_data, ifeature, color, class_label=''):
    """
    Plots the input against the reconstructed data.
    @input_data  :: Numpy array of the input data.
    @recon_data  :: Numpy array of the reconstructed ae data.
    @ifeature    :: Int the number of the features to be plotted.
    @color       :: String for the color of the two plotted histograms.
    @class_label :: String for either signal or background.
    """
    plt.rc('xtick', labelsize=23)
    plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    prange = (np.amin(input_data, axis=0), np.amax(input_data, axis=0))
    plt.hist(x=input_data, bins=60, range=prange, alpha=0.8, histtype='step',
             linewidth=2.5, label=class_label, density=True, color=color)
    plt.hist(x=recon_data, bins=60, range=prange, alpha=0.8, histtype='step',
             linewidth=2.5, label='Rec. ' + class_label, linestyle='dashed',
             density=True, color=color)

    plt.xlabel(util.varname(ifeature) + ' (normalized)'); plt.ylabel('Density')
    plt.xlim(*prange)
    plt.gca().set_yscale("log")
    plt.legend()


def sig_vs_bkg(data_sig, data_bkg, model_path, output_folder):
    """
    Plots the signal and background histograms of a data set overlaid.
    @data_sig      :: Numpy array of the signal data.
    @data_bkg      :: Numpy array of the background data.
    @model_path    :: String of path to a trained ae model.
    @output_folder :: Folder where the figures are saved.
    """
    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)

    plt.rc('xtick', labelsize=23); plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25); plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    for i in range(data_sig.shape[1]):
        xmax = max(np.amax(data_sig[:,i]),np.amax(data_bkg[:,i]))
        xmin = min(np.amin(data_sig[:,i]),np.amin(data_bkg[:,i]))
        fig = plt.figure(figsize=(12, 10))

        hSig, _, _ = plt.hist(x=data_sig[:, i], density=1,
                              range=(xmin, xmax), bins=50, alpha=0.8,
                              histtype='step', linewidth=2.5, label='Sig',
                              color='navy')
        hBkg, _, _ = plt.hist(x=data_bkg[:, i], density=1,
                              range=(xmin, xmax), bins=50, alpha=0.4,
                              histtype='step', linewidth=2.5, label='Bkg',
                              color='gray', hatch='xxx')
        plt.legend()
        fig.savefig(plots_folder + 'Feature '+ str(i) + '.pdf')
        plt.close()

    print(f"Latent plots were saved to {plots_folder}.")


def compute_auc(data, target, feature) -> tuple(list, list, float, float):
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


def roc_plots(sig, bkg, model_path, output_folder):
    """
    Plot the ROC of a whole data set, for each feature, and then save the
    sum of the AUCs of all the features to a text file.
    @sig           :: Numpy array containing the signal data.
    @bkg           :: Numpy array containing the background data.
    @model_path    :: String of the path to a trained ae model.
    @output_folder :: String of the name to the output folder to save plots.
    """
    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)

    plt.rc('xtick', labelsize=23); plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25); plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    data = np.vstack((sig, bkg))
    target = np.concatenate((np.ones(sig.shape[0]),np.zeros(bkg.shape[0])))

    auc_sum = 0.
    for feature in range(data.shape[1]):
        fpr, tpr, mean_auc, std_auc = compute_auc(data, target, feature)
        fig = plt.figure(figsize=(12, 10))
        plt.plot(fpr, tpr,
                 label=f"AUC: {mean_auc:.3f} ± {std_auc:.3f}", color='navy')
        plt.plot([0, 1], [0, 1], ls="--", color='gray')

        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
        plt.legend()

        auc_sum += mean_auc
        fig.savefig(plots_folder + f"Feature {feature}.pdf")
        plt.close()

    with open(plots_folder + 'auc_sum.txt', 'w') as auc_sum_file:
        auc_sum_file.write(f"{auc_sum:.3f}")

    print(f"Latent roc plots were saved to {plots_folder}.")
