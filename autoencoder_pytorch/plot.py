# Plot the different things with respect to a trained autoencoder model
# such as the epochs vs loss plot.
import matplotlib.pyplot as plt
import numpy as np
import argparse, os
import torch
import torch.nn as nn
from sklearn import metrics

import util

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_folder", type=str,
    help="The folder where the data is stored on the system..")
parser.add_argument("--valid_file", type=str,
    help="The name of the validation file.")
parser.add_argument("--test_file", type=str,
    help="The name of the test file.")
parser.add_argument('--model_path', type=str, required=True,
    help="The path to the saved model.")

def main():
    args               = parser.parse_args()
    device             = 'cpu'
    ae_type            = "vanilla"
    encoder_activation = nn.Tanh()
    decoder_activation = nn.Tanh()

    # Import the hyperparameters used in training and load the data.
    layers, batch, lr = import_hyperparams(args.model_path)
    valid_target_file = "y" + args.valid_file[1:]
    test_target_file  = "y" + args.test_file[1:]

    valid_data   = np.load(os.path.join(args.data_folder, args.valid_file))
    test_data    = np.load(os.path.join(args.data_folder, args.test_file))
    valid_target = np.load(os.path.join(args.data_folder, valid_target_file))
    test_target  = np.load(os.path.join(args.data_folder, test_target_file))

    test_sig, test_bkg = util.split_sig_bkg(test_data, test_target)

    # Import the model.
    (layers).insert(0, test_data.shape[1])
    model = util.choose_ae_model(ae_type, device, layers, lr,
        encoder_activation, decoder_activation)
    model = util.load_model(model, args.model_path)

    # Compute loss function results for the test and validation datasets.
    print('\n----------------------------------')
    print(f"VALID MSE: {model.compute_loss(valid_data).item()}")
    print(f"TEST MSE: {model.compute_loss(test_data).item()}")
    print('----------------------------------\n')

    # Compute the signal and background latent spaces and decoded data.
    sig_latent, sig_recon = model.predict(test_sig)
    bkg_latent, bkg_recon = model.predict(test_bkg)

    latent_sig_bkg = np.vstack((sig_latent, bkg_latent))
    target_sig_bkg = \
    np.concatenate((np.ones(sig_latent.shape[0]),np.zeros(bkg_latent.shape[0])))

    # Do the plots.
    latent_space_plots(sig_latent, bkg_latent, args.model_path)
    latent_roc_plots(latent_sig_bkg, target_sig_bkg, args.model_path)
    sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg,args.model_path)


def import_hyperparams(model_path):
    # Import the hyperparameters given the path to the model, since the name
    # of the model folder has the layers, batch, and learning rate in it.

    hyperparams = max(model_path.split('/'), key=len)
    layers  = hyperparams[hyperparams.find('L')+1:hyperparams.find('_')]
    layers  = [int(nb) for nb in layers.split(".")]
    batch   = int(hyperparams[hyperparams.find('B')+1:hyperparams.find('_',
        hyperparams.find('B')+1, len(hyperparams))])
    lr      = float(hyperparams[hyperparams.find('Lr')+2:hyperparams.find('_',
        hyperparams.find('Lr')+2, len(hyperparams))])

    print("\nImported model hyperparameters:")
    print("--------------------------------")
    print(f"Layers: {layers}")
    print(f"Batch: {batch}")
    print(f"Learning Rate: {lr}")

    return layers, batch, lr

def sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg, model_path):
    # Plot the background and the signal distributions for the input data and
    # the reconstructed data, overlaid.
    plots_folder = model_path + 'ratio_plots/'
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)

    for idx in range(input_sig.numpy().shape[1]):
        plt.figure(figsize=(12,10))

        ratio_plotter(input_bkg.numpy()[:,idx], output_bkg.numpy()[:,idx], idx,
            'gray', class_label='Background')
        ratio_plotter(input_sig.numpy()[:,idx], output_sig.numpy()[:,idx], idx,
            'navy', class_label='Signal')

        plt.savefig(plots_folder + 'Ratio Plot ' + util.varname(idx) + '.pdf')
        plt.close()

    print(f"\033[92mRatio plots were saved to {plots_folder}.\033[0m")

def latent_space_plots(latent_data_sig, latent_data_bkg, model_path):
    # Makes the plots of the latent space data produced by the encoder.
    storage_folder_path = model_path + 'latent_plots/'
    if not os.path.exists(storage_folder_path): os.makedirs(storage_folder_path)

    for i in range(latent_data_sig.shape[1]):
        xmax = max(np.amax(latent_data_sig[:,i]),np.amax(latent_data_bkg[:,i]))
        xmin = min(np.amin(latent_data_sig[:,i]),np.amin(latent_data_bkg[:,i]))

        hSig,_,_ = plt.hist(x=latent_data_sig[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.8, histtype='step',
            linewidth=2.5, label='Sig', color='navy')
        hBkg,_,_ = plt.hist(x=latent_data_bkg[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.4, histtype='step',
            linewidth=2.5,label='Bkg', color='gray', hatch='xxx')

        plt.legend()
        plt.xlabel(f'Latent feature {i}')
        plt.savefig(storage_folder_path + 'Latent Feature '+ str(i) + '.pdf')
        plt.close()

    print(f"\033[92mLatent plots were saved to {storage_folder_path}.\033[0m")

def latent_roc_plots(data, target, model_path):
    # Plot the roc curves of the latent space distributions.
    plots_folder = model_path + 'latent_roc_plots/'
    if not os.path.exists(plots_folder): os.makedirs(plots_folder)

    plt.rc('xtick', labelsize=23); plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25); plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    auc_sum = 0.
    for feature in range(data.shape[1]):
        fpr, tpr, thresholds = metrics.roc_curve(target, data[:, feature])
        auc = metrics.roc_auc_score(target, data[:, feature])

        fig = plt.figure(figsize=(12, 10))
        plt.title(f"Latent feature {feature}")
        plt.plot(fpr, tpr, label=f"AUC: {auc}", color='navy')
        plt.plot([0, 1], [0, 1], ls="--", color='gray')

        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
        plt.legend()

        auc_sum += auc
        fig.savefig(plots_folder + f"Latent feature {feature}.png")
        plt.close()

    with open(plots_folder + 'auc_sum.txt', 'w') as auc_sum_file:
        auc_sum_file.write(f"{auc_sum:.3f}")

    print(f"\033[92mLatent roc plots were saved to {plots_folder}.\033[0m")

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

def loss_plot(loss_train, loss_valid, min_valid, epochs, outdir):
    # Plots the loss for each epoch for the training and validation data.

    plt.plot(list(range(epochs)), loss_train, color="gray",
        label="Training Loss (last batch)")
    plt.plot(list(range(epochs)), loss_valid, color="navy",
        label="Validation Loss (1 per epoch)")
    plt.xlabel("Epochs"); plt.ylabel("Loss")

    plt.title(f"min={min_valid:.6f}")

    plt.legend()
    plt.savefig(outdir + "loss_epochs.pdf"); plt.close()

    print(f"\033[92mLoss vs epochs plot saved to {outdir}.\033[0m")

if __name__ == '__main__':
    main()
