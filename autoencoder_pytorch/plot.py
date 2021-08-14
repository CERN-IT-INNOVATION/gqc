# Plot the different things with respect to a trained autoencoder model
# such as the epochs vs loss plot.
import matplotlib.pyplot as plt
import numpy as np
import argparse, os
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.utils import shuffle

import util
import data
from terminal_colors import tcols

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--norm", type=str,
    help="The name of the normalisation that you'll to use.")
parser.add_argument("--nevents", type=str,
    help="The number of events of the norm file.")
parser.add_argument('--model_path', type=str, required=True,
    help="The path to the saved model.")

def main():
    args   = parser.parse_args()
    device = 'cpu'
    hp     = util.import_hyperparams(args.model_path)

    # Import the data.
    ae_data = data.AE_data(args.data_folder, args.norm, args.nevents)
    test_sig, test_bkg = \
        ae_data.split_sig_bkg(ae_data.test_data, ae_data.test_target)

    # Import the model.
    model = util.choose_ae_model(hp['ae_type'], device, hp)
    model.load_model(args.model_path)

    # Compute loss function results for the test and validation datasets.
    # print('\n----------------------------------')
    # print("VALID LOSS:")
    # print(model.compute_loss(ae_data.valid_data, ae_data.valid_target).item())
    # print("TEST LOSS:")
    # print(model.compute_loss(ae_data.test_data, ae_data.test_target).item())
    # print('----------------------------------\n')

    # Compute the signal and background latent spaces and decoded data.
    model_sig = model.predict(test_sig)
    model_bkg = model.predict(test_bkg)

    if len(model_sig) == 3:
        roc_plots(model_sig[2], model_bkg[2], args.model_path, 'classif_roc')

    sig_vs_bkg(model_sig[0], model_bkg[0], args.model_path, 'latent_plots')
    roc_plots(model_sig[0], model_bkg[0], args.model_path, 'latent_roc')
    input_vs_reco(test_sig,test_bkg,model_sig[1],model_bkg[1],args.model_path)

def input_vs_reco(input_sig, input_bkg, output_sig, output_bkg, model_path):
    # Plot the background and the signal distributions for the input data and
    # the reconstructed data, overlaid.
    plots_folder = os.path.dirname(model_path) + '/input_vs_reco/'
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)

    for idx in range(input_sig.shape[1]):
        plt.figure(figsize=(12,10))

        ratio_plotter(input_bkg[:,idx], output_bkg[:,idx], idx, 'gray',
            class_label='Background')
        ratio_plotter(input_sig[:,idx], output_sig[:,idx], idx, 'chartreuse',
            class_label='Signal')

        plt.savefig(plots_folder + util.varname(idx) + '.png')
        plt.close()

    print(f"Ratio plots were saved to {plots_folder}.")

def sig_vs_bkg(data_sig, data_bkg, model_path, output_folder):
    # Makes the plots of the latent space data produced by the encoder.
    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)

    for i in range(data_sig.shape[1]):
        xmax = max(np.amax(data_sig[:,i]),np.amax(data_bkg[:,i]))
        xmin = min(np.amin(data_sig[:,i]),np.amin(data_bkg[:,i]))

        hSig,_,_ = plt.hist(x=data_sig[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.8, histtype='step',
            linewidth=2.5, label='Sig', color='chartreuse')
        hBkg,_,_ = plt.hist(x=data_bkg[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.4, histtype='step',
            linewidth=2.5,label='Bkg', color='gray', hatch='xxx')

        plt.legend()
        plt.xlabel(f'Latent feature {i}')
        plt.savefig(plots_folder + 'Feature '+ str(i) + '.png')
        plt.close()

    print(f"Latent plots were saved to {plots_folder}.")

def compute_auc(data, target, feature):
    # Divides the full data into chunks, computes auc for each chunk, then
    # computes the mean and the standard devation of these aucs.
    data, target  = shuffle(data, target, random_state=0)
    data_chunks   = np.array_split(data, 5)
    target_chunks = np.array_split(target, 5)

    aucs = []
    for dat, trg in zip(data_chunks, target_chunks):
        fpr, tpr, thresholds = metrics.roc_curve(trg, dat[:, feature])
        auc = metrics.roc_auc_score(trg, dat[:, feature])
        aucs.append(auc)

    aucs = np.array(aucs)
    mean_auc = aucs.mean()
    std_auc  = aucs.std()
    fpr, tpr, thresholds = metrics.roc_curve(target, data[:, feature])

    return fpr, tpr, mean_auc, std_auc

def roc_plots(sig, bkg, model_path, output_folder):
    # Plot roc curves given data and target.
    plots_folder = os.path.dirname(model_path) + "/" + output_folder + "/"
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)

    plt.rc('xtick', labelsize=23); plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25); plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    data   = np.vstack((sig, bkg))
    target = np.concatenate((np.ones(sig.shape[0]),np.zeros(bkg.shape[0])))

    auc_sum = 0.
    for feature in range(data.shape[1]):
        fpr, tpr, mean_auc, std_auc = compute_auc(data, target, feature)
        fig = plt.figure(figsize=(12, 10))
        plt.title(f"Feature {feature}")
        plt.plot(fpr, tpr,
            label=f"AUC: {mean_auc:.3f} ± {std_auc:.3f}", color='chartreuse')
        plt.plot([0, 1], [0, 1], ls="--", color='gray')

        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
        plt.legend()

        auc_sum += mean_auc
        fig.savefig(plots_folder + f"Feature {feature}.png")
        plt.close()

    with open(plots_folder + 'auc_sum.txt', 'w') as auc_sum_file:
        auc_sum_file.write(f"{auc_sum:.3f}")

    print(f"Latent roc plots were saved to {plots_folder}.")

def ratio_plotter(input_data, output_data, ifeature, color, class_label=''):
    # Plots two overlaid histograms.
    plt.rc('xtick', labelsize=23); plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25); plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    prange = (np.amin(input_data, axis=0), np.amax(input_data, axis=0))
    plt.hist(x=input_data, bins=60, range=prange, alpha=0.8, histtype='step',
        linewidth=2.5, label=class_label, density=True, color=color)
    plt.hist(x=output_data, bins=60, range=prange, alpha=0.8, histtype='step',
        linewidth=2.5, label='Rec. ' + class_label, linestyle='dashed',
        density=True, color=color)

    plt.xlabel(util.varname(ifeature) + ' (normalized)'); plt.ylabel('Density')
    plt.xlim(*prange)
    plt.gca().set_yscale("log")
    plt.legend()

if __name__ == '__main__':
    main()
