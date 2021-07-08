# Uses the optuna library to optimize the hyperparameters for a given AE.
import argparse, os, time

import optuna
from optuna.trial import TrialState
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim

import ae_vanilla
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
parser.add_argument('--maxdata_train', type=int, default=-1,
    help='The maximum number of training samples to use.')


def optuna_train(train_loader, valid_loader, model, epochs, trial):
    # Training method for the autoencoder that was defined above.
    print('Training the model...')
    loss_training = []; loss_validation = []; min_valid = 99999
    optimizer = model.optimizer()

    for epoch in range(epochs):
        model.train()
        for i, batch_feats in enumerate(train_loader):
            train_loss = ae_vanilla.eval_train(model, batch_feats, optimizer)

        valid_loss, min_valid = \
        ae_vanilla.eval_valid(model, valid_loader, min_valid, None)

        loss_validation.append(valid_loss)
        loss_training.append(train_loss.item())

        trial.report(valid_loss, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        if min_valid > 0.05: raise optuna.TrialPruned()

        print("Epoch : {}/{}, Training loss (last batch) = {:.8f}".
               format(epoch + 1, epochs, train_loss.item()))
        print("Epoch : {}/{}, Validation loss = {:.8f}".
               format(epoch + 1, epochs, valid_loss))

    return loss_training, loss_validation, min_valid

def optuna_objective(trial):
    """
    Wrapper of the normal training such that it agrees with what optuna
    is trying to do.
    """
    args   = parser.parse_args()
    device = util.define_torch_device()

    # Define optuna parameters.
    lr     = trial.suggest_loguniform('lr', *args.lr)
    batch  = trial.suggest_categorical('batch', args.batch)

    # Load the data.
    train_data = np.load(args.train_file)
    valid_data = np.load(args.valid_file)
    print("\n----------------")
    print("Training data size: {:.2e}".format(train_data.shape[0]))
    print("Validation data size: {:.2e}".format(valid_data.shape[0]))
    print("----------------\n")
    train_loader = util.to_pytorch_data(train_data, device, batch, True)
    valid_loader = util.to_pytorch_data(valid_data, device, None, True)

    (args.layers).insert(0, len(train_loader.dataset[1]))

    # Define model.
    (args.layers).insert(0, np.load(args.train_file).shape[1])
    model = ae_vanilla.AE(nodes=args.layers, lr=lr, device=device,
        en_activ=nn.Tanh(), dec_activ=nn.Tanh()).to(device)

    # Train model.
    start_time = time.time()
    loss_train, loss_valid, min_valid = optuna_train(train_loader,
        valid_loader, model, args.epochs, trial)
    end_time = time.time()
    train_time = (end_time - start_time)/60

    print("Training time: {:.2e} mins.".format(train_time))

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
