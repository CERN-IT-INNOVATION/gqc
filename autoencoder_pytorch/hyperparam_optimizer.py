# Uses the optuna library to optimize the hyperparameters for a given AE.
import argparse, os, time

import optuna
from optuna.trial import TrialState
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim

import util

default_layers = [64, 52, 44, 32, 24, 16]
parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_file", type=str,
    help="The path to the training data.")
parser.add_argument("--valid_file", type=str,
    help="The path to the validation data.")
parser.add_argument('--lr', type=float, nargs=2,
    help='The learning rate range [min max].')
parser.add_argument('--layers', type=int, default=default_layers, nargs='+',
    help='The layers structure.')
parser.add_argument('--batch', type=int, nargs="+",
    help='The batch options, e.g., [128 256 512].')
parser.add_argument('--epochs', type=int,
    help='The number of training epochs.')

def optuna_train(train_loader, valid_loader, model, epochs, trial):
    # Training method for the autoencoder that was defined above.
    print('\033[96mTraining the AE model...\033[0m')

    for epoch in range(epochs):
        model.train()

        model.train_all_batches(train_loader)
        model.valid(valid_loader, outdir)

        model.print_losses(epoch, epochs)

        trial.report(model.all_valid_loss[epoch], epoch)
        if trial.should_prune():         raise optuna.TrialPruned()
        if model.best_valid_loss > 0.09: raise optuna.TrialPruned()

    return model.best_valid_loss

def optuna_objective(trial):
    """
    Wrapper of the normal training such that it agrees with what optuna
    is trying to do.
    """
    args               = parser.parse_args()
    device             = util.define_torch_device()
    ae_type            = "vanilla"
    encoder_activation = nn.Tanh()
    decoder_activation = nn.Tanh()

    # Define parameters to be optimized by optuna.
    lr    = trial.suggest_loguniform('lr', *args.lr)
    batch = trial.suggest_categorical('batch', args.batch)

    # Load the data, both input and target.
    train_data = np.load(os.path.join(args.data_folder, args.train_file))
    valid_data = np.load(os.path.join(args.data_folder, args.valid_file))
    train_target_file = "y" + args.train_file[1:]
    valid_target_file = "y" + args.valid_file[1:]
    train_target = np.load(os.path.join(args.data_folder, train_target_file))
    valid_target = np.load(os.path.join(args.data_folder, valid_target_file))

    train_loader = \
        util.to_pytorch_data(train_data, train_target, device, batch, True)
    valid_loader = \
        util.to_pytorch_data(valid_data, valid_target, device, None, True)

    print("\n----------------")
    print("\033[92mData loading complete:\033[0m")
    print(f"Training data size: {train_data.shape[0]:.2e}")
    print(f"Validation data size: {valid_data.shape[0]:.2e}")
    print("----------------\n")

    # Define the model and prepare the output folder.
    nfeatures = train_data.shape[1]
    (args.layers).insert(0, nfeatures)

    model = util.choose_ae_model(ae_type, device, args.layers, lr,
        encoder_activation, decoder_activation)

    min_valid = \
        optuna_train(train_loader, valid_loader, model, args.epochs, trial)

    return min_valid

if __name__ == '__main__':
    # Start the optuna study.
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize',
        pruner=optuna.pruners.HyperbandPruner())

    study.optimize(optuna_objective, n_trials=50)

    comp_trials= study.trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(comp_trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("\n\nParams: ")
    for key, value in trial.params.items(): print("{}: {}".format(key, value))
