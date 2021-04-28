import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import torch.nn as nn

from model_vasilis import AE
import util

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
parser.add_argument('--batch', type=int, default=64,
    help='The batch size.')
parser.add_argument('--model', type=str, required=True,
    help='The path to the saved model.')


def main():
    # Parse the arguments only if this is ran as a script.
    args = parser.parse_args()
    # Define torch device.
    device = util.define_torch_device()
    # Import all the useful data. For plotting, split test data into s and b.
    valid_data   = np.load(args.validation_file)
    test_data    = np.load(args.testing_file)
    valid_loader = util.to_pytorch_data(valid_data, None, False)
    test_loader  = util.to_pytorch_data(test_data, None, False)

    test_target   = np.load(args.testing_target)
    test_data_sig, test_data_bkg = split_sig_bkg(test_data, test_target)
    test_loader_sig = util.to_pytorch_data(test_data_sig, None, False)
    test_loader_bkg = util.to_pytorch_data(test_data_bkg, None, False)

    (args.layers).insert(0, test_data.shape[1])

    # Import the model... NB add a more general way of importing ae models.
    model = load_model(AE, args.layers, args.model, device)
    criterion = nn.MSELoss(reduction= 'mean')

    # Calculate the MSE and MAPE for test and validation data.
    test_output, test_latent, test_input = compute_model(model, test_loader)
    print('TEST sample MSE:', criterion(test_output, test_input).item())
    valid_output, valid_latent, valid_input = compute_model(model,valid_loader)
    print('VALID sample MSE:', criterion(valid_output, valid_input).item())
    test_output, test_latent, test_input = compute_model(model, test_loader)
    print('TEST sample MAPE:', mape(test_output, test_input).item())

    # Load the signal and background latent spaces.
    output_sig, sig_latent, input_sig = compute_model(model, test_loader_sig)
    output_bkg, bkg_latent,_input_bkg = compute_model(model, test_loader_bkg)
    sig_latent = sig_latent.numpy()
    bkg_latent = bkg_latent.numpy()

    # Do the plots.
    latent_space_plots(sig_latent, bkg_latent, args.model)
    sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg, args.model)

    # Load sig/bkg data, but this time with batch_size.
    loader_sig = util.to_pytorch_data(test_data_sig, args.batch, True)
    loader_bkg = util.to_pytorch_data(test_data_bkg, args.batch, True)

    sig_bloss = util.mean_batch_loss(loader_sig, test_data.shape[1], device)
    bkg_bloss = util.mean_batch_loss(loader_bkg, test_data.shape[1], device)

    mean_losses = [sig_batch_loss, bkg_batch_loss]
    batch_loss_plots(mean_losses, len(test_data), args.model)

def ratio_plotter(input_data, output_data, ifeature, class_label=''):
    # Plots the overlaid input and output data to see how good the
    # reconstruction really is.
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', titlesize=22)
    plt.rc('axes', labelsize=22)
    plt.rc('legend', fontsize=22)

    plt.hist(x=input_data, bins=60, range=(0,1), alpha=0.8, histtype='step',
        linewidth=2.5, label=class_label, density=True)
    plt.hist(x=output_data, bins=60, range = (0,1), alpha=0.8, histtype='step',
        linewidth = 2.5, label='Rec. ' + class_label, density=True)
    plt.xlabel(util.varname(ifeature) + ' (normalized)')
    plt.ylabel('Density')
    plt.xlim(0, 0.4)
    plt.legend()

def latent_space_plots(latent_data_sig, latent_data_bkg, model_path):
    # Makes the plots of the latent space data for each of the latent space
    # features, which should be 8 for the vasilis_model, for example.
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
        plt.savefig(model_path + 'latent_plot' + str(i) + '.png')
        plt.clf()

def sig_bkg_plots(input_sig, input_bkg, output_sig, output_bkg, model_path):
    # Plot the background and the signal distributions, overlaid.
    # FIXME: Not displayed properly. Values are off and seems to be bias still
    # between sig and bkg for the MSE distributions.

    for idx in range(input_sig.numpy().shape[1]):
        plt.figure(figsize=(12,10))

        ratio_plotter(input_bkg.numpy()[:,idx], output_bkg.numpy()[:,idx], idx,
            class_label='Background')
        ratio_plotter(input_sig.numpy()[:,idx], output_sig.numpy()[:,idx], idx,
            class_label='Signal')

        plt.savefig(model_path + 'ratio_plotter' + util.varname(idx) + '.pdf')
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
    plt.savefig(outdir + 'loss_epochs.png')

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
        plt.hist(losses_sig_bkg[idx], bins=20, density=0, color=color[idx],
            alpha=0.5, ec='black', label=label[idx])
        plt.ylabel('Entries/Bin')
        plt.xlabel('MSE per Batch')
        plt.title('MSE per batch, Ntest={}.'.format(n_test))

    plt.legend()
    plt.savefig(model_path + 'test_loss_hist.png', dpi=300)

def mape(output, target, epsilon=1e-4):
    # Another type of criterion. Did not work too well.
    # Very sensitive to epsilon, which is very bad.
    loss = torch.mean(torch.abs((output - target)/(target + epsilon)))
    return loss

if __name__ == '__main__':
    main()
