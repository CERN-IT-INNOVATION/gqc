import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn
import os

from sklearn.preprocessing import MinMaxScaler

from model_vasilis import AE
import util
from util import compute_model

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--validation_file", type=str,
    help="The path to the validation data.")
parser.add_argument("--testing_file", type=str,
    help="The path to the test data.")
parser.add_argument("--testing_target", type=str,
    help="The path to the test target numpy file.")
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layers structure.')
parser.add_argument('--lr', type=float,
    help='The learning rate of the model.')
parser.add_argument('--batch', type=int, default=64,
    help='The batch size.')
parser.add_argument('--maxdata_train', type=int, default=1150000,
    help='The maximum number of training samples to use.')
parser.add_argument('--model_path', type=str, required=True,
    help='The path to the saved model.')

def main():
    args = parser.parse_args()
    device = 'cpu'

    # Import data. For plotting, split test data into s and b.

    max_data = int(0.1*args.maxdata_train/0.8)
    valid_data   = np.load(args.validation_file)[:max_data,:]
    test_data    = np.load(args.testing_file)[:max_data,:]
    valid_loader = util.to_pytorch_data(valid_data, device, None, False)
    test_loader  = util.to_pytorch_data(test_data, device, None, False)

    test_data   = np.load(args.testing_file)
    test_target = np.load(args.testing_target)
    data_sig, data_bkg = util.split_sig_bkg(test_data, test_target, max_data)

    # If want exact data that Vasilis uses, uncomment this one line  and
    # comment previous three lines.
    # data_sig, data_bkg = vasilis_reprod_data()

    test_loader_sig = util.to_pytorch_data(data_sig, device, None, False)
    test_loader_bkg = util.to_pytorch_data(data_bkg, device, None, False)

    # Import the model.
    (args.layers).insert(0, test_data.shape[1])
    model = util.load_model(AE, args.layers, args.lr, args.model_path, device)
    criterion = model.criterion()

    # Compute criterion scores for data.
    score_file = open(args.model_path + "score_file.txt", "w")
    compute_score(model, test_loader, criterion, "TEST MSE: ", score_file)
    compute_score(model, valid_loader, criterion, "VALID MSE: ", score_file)
    score_file.close()

    # Load the signal and background latent spaces.
    output_sig, sig_latent, input_sig = compute_model(model, test_loader_sig)
    output_bkg, bkg_latent, input_bkg = compute_model(model, test_loader_bkg)
    sig_latent = sig_latent.numpy()
    bkg_latent = bkg_latent.numpy()

    # Do the plots.
    latent_space_plots(sig_latent, bkg_latent, args.model_path)
    sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg,args.model_path)

    # Load sig/bkg data, but this time with batch_size.
    loader_sig = util.to_pytorch_data(data_sig, device, args.batch, True)
    loader_bkg = util.to_pytorch_data(data_bkg, device, args.batch, True)

    feature_nb = test_data.shape[1]
    sig_batch_loss = util.mean_batch_loss(loader_sig, model, feature_nb,
        criterion, device)
    bkg_batch_loss = util.mean_batch_loss(loader_bkg, model, feature_nb,
        criterion, device)

    mean_losses = [sig_batch_loss, bkg_batch_loss]
    batch_loss_plots(mean_losses, len(test_data), args.model_path)

def vasilis_reprod_data():
    # Reproduces the way Vasilis normalizes data such that it leads to
    # identical plots.

    data_sig = np.load('/work/deodagiu/qml_data/input_ae/x_data_sig.npy')
    data_bkg = np.load('/work/deodagiu/qml_data/input_ae/x_data_bkg.npy')

    ntot = 360000
    data_sig = data_sig[:360000, :]
    data_bkg = data_bkg[:360000, :]
    all_data = np.vstack((data_sig, data_bkg))
    norm_data = MinMaxScaler().fit_transform(all_data)
    data_sig = norm_data[:360000, :]
    data_bkg = norm_data[360000:, :]

    ntrain, nvalid, ntest = int(0.8*ntot), int(0.1*ntot), int(0.1*ntot)
    data_sig = data_sig[ntrain+nvalid:ntot, :]
    data_bkg = data_bkg[ntrain+nvalid:ntot, :]

    return data_sig, data_bkg

def compute_score(model, loader, criterion, prepend_str, score_file):
    """
    Compute the score for a certain model and data.

    @model       :: The pytorch model object.
    @loader      :: The pytorch data loader object for the data.
    @criterion   :: The criterion object for the loss function.
    @prepend_str :: Descriptive string to prepend to output.

    @returns :: Writes to screen and to file the descriptive string and the
        computed loss function score.
    """
    output, latent, inp = compute_model(model, loader)
    print(prepend_str, criterion(output,inp).item())
    score_file.write(prepend_str + "{}\n".format(criterion(output,inp).item()))


def ratio_plotter(input_data, output_data, ifeature, color, class_label=''):
    # Plots the overlaid input and output data to see how good the
    # reconstruction really is.
    plt.rc('xtick', labelsize=23)
    plt.rc('ytick', labelsize=23)
    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)
    plt.rc('legend', fontsize=22)

    plt.hist(x=input_data, bins=60, range=(0,1), alpha=0.8, histtype='step',
        linewidth=2.5, label=class_label, density=True, color=color)
    # plt.hist(x=output_data, bins=60, range=(0,1), alpha=0.8, histtype='step',
    #     linewidth=2.5, linestyle='dashed',
    #     label='Rec. ' + class_label, density=True, color=color)
    plt.xlabel(util.varname(ifeature) + ' (normalized)')
    plt.ylabel('Density')
    plt.xlim(0, 1)
    plt.gca().set_yscale("log")
    plt.legend()

def latent_space_plots(latent_data_sig, latent_data_bkg, model_path):
    # Makes the plots of the latent space data for each of the latent space
    # features, which should be 8 for the vasilis_model, for example.
    if not os.path.exists(model_path + 'latent_plots/'):
        os.makedirs(model_path + 'latent_plots/')
    for i in range(latent_data_sig.shape[1]):
        xmax = max(np.amax(latent_data_sig[:,i]),np.amax(latent_data_bkg[:,i]))
        xmin = min(np.amin(latent_data_sig[:,i]),np.amin(latent_data_bkg[:,i]))
        hSig,_,_ = plt.hist(x=latent_data_sig[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.6, histtype='step',
            linewidth=2.5, label='Sig')
        hBkg,_,_ = plt.hist(x=latent_data_bkg[:,i], density=1,
            range = (xmin,xmax), bins=50, alpha=0.6, histtype='step',
            linewidth=2.5,label='Bkg')

        plt.legend()
        plt.xlabel(f'Latent feature {i}')
        plt.savefig(model_path + 'latent_plots/' + 'latent_plot'+
            str(i) + '.pdf')
        plt.clf()

def sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg, model_path):
    # Plot the background and the signal distributions, overlaid.
    if not os.path.exists(model_path + 'ratio_plots/'):
        os.makedirs(model_path + 'ratio_plots/')
    for idx in range(input_sig.numpy().shape[1]):
        plt.figure(figsize=(12,10))

        ratio_plotter(input_bkg.numpy()[:,idx], output_bkg.numpy()[:,idx], idx,
            'gray', class_label='Background',)
        ratio_plotter(input_sig.numpy()[:,idx], output_sig.numpy()[:,idx], idx,
            'navy',class_label='Signal')

        plt.savefig(model_path + 'ratio_plots/' + 'Ratio Plot ' +
            util.varname(idx) + '.pdf')
        plt.clf()

def diagnosis_plots(loss_train, loss_valid, min_valid, nodes, batch_size, lr,
    epochs, outdir):
    # Quick plots to see if what we trained is of any good.
    plt.plot(list(range(epochs)), loss_train,
        label='Training Loss (last batch)')
    plt.plot(list(range(epochs)), loss_valid,
        label='Validation Loss (1 per epoch)')
    plt.ylabel("MSE")
    plt.xlabel("epochs")
    plt.title("B=" + str(batch_size) + ", lr=" + str(lr) + ", L=" +
        str(nodes) + ', min={:.6f}'.format(min_valid))
    plt.legend()
    plt.savefig(outdir + 'loss_epochs.pdf')

def batch_loss_plots(losses_sig_bkg, n_test, model_path):
    """
    Makes plots of the mean batch loss for the signal and background samples,
    overlays them and saves the final result.

    @losses_sig_bkg :: Array with two elements: mean batch losses array for
        the signal sample and the same for the bkg sample.
    @n_test         :: Number of test data events.
    @model_path     :: The path to the saved model.
    """
    colors = ['b', 'r']
    label  = ["Test on Sig.", "Test on Bkg."]
    for idx in range(len(losses_sig_bkg)):
        plt.hist(losses_sig_bkg[idx], bins=20, density=0, color=colors[idx],
            alpha=0.5, ec='black', label=label[idx])
        plt.ylabel('Entries/Bin')
        plt.xlabel('MSE per Batch')
        plt.title('MSE per batch, Ntest={}.'.format(n_test))

    plt.legend()
    plt.savefig(model_path + 'test_loss_hist.pdf', dpi=300)

if __name__ == '__main__':
    main()
