# Uses the optuna library to optimize the hyperparameters for a given AE.
import argparse, os, time

import optuna
from optuna.trial import TrialState
import joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim

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
parser.add_argument("--aetype", type=str,
    help="The type of autoencoder that you will use, i.e., vanilla etc..")
parser.add_argument('--lr', type=float, nargs=2,
    help='The learning rate range [min max].')
parser.add_argument('--batch', type=int, nargs="+",
    help='The batch options, e.g., [128 256 512].')
parser.add_argument('--epochs', type=int,
    help='The number of training epochs.')

def optuna_train(train_loader, valid_loader, model, epochs, trial):
    # Training method for the autoencoder that was defined above.
    print(tcols.OKCYAN+"Training the AE model to be optimized..." +tcols.ENDC)
    model.instantiate_adam_optimizer()
    model.network_summary(); model.optimizer_summary()

    valid_losses = []
    for epoch in range(epochs):
        model.train()

        train_loss = model.train_all_batches(train_loader)
        valid_loss = model.valid(valid_loader, None)

        if model.hp["ae_type"] in ["classifier", "classvqc"]:
            model.all_recon_loss.append(valid_loss[1].item())
            model.all_class_loss.append(valid_loss[2].item())

        if model.hp["ae_type"] in ["sinkclass"]:
            model.all_recon_loss.append(valid_loss[1].item())
            model.all_laten_loss.append(valid_loss[2].item())
            model.all_class_loss.append(valid_loss[3].item())

        if model.hp["ae_type"] in ["variational", "sinkhorn"]:
            model.all_recon_loss.append(valid_loss[1].item())

        if model.early_stopping(): return model.best_valid_loss
        model.print_losses(epoch, epochs, train_loss, valid_loss)

        trial.report(train_loss.item(), epoch)
        if trial.should_prune(): raise optuna.TrialPruned()

    if model.hp["ae_type"] in ["classifier", "classvqc", "sinkclass"]:
        return min(model.all_class_loss)

    if model.hp["ae_type"] in ["variational", "sinkhorn"]:
        return min(model.all_recon_loss)

    return model.best_valid_loss

def optuna_objective(trial):
    """
    Wrapper of the normal training such that it agrees with what optuna
    is trying to do.
    """
    args   = parser.parse_args()
    device = util.define_torch_device()
    vqc_specs   = [["zzfm", 0, 4], ["2local", 0, 20, 4, "linear"],
                   ["zzfm", 4, 8], ["2local", 20, 40, 4, "linear"]]
    hyperparams   = {
        "lr"           : args.lr,
        "ae_layers"    : [64, 52, 44, 32, 24, 16],
        "class_layers" : [32, 64, 128, 64, 32, 16, 8, 1],
        "enc_activ"    : 'nn.Tanh()',
        "dec_activ"    : 'nn.Tanh()',
        "vqc_specs"    : vqc_specs,
        "loss_weight"  : 1,
        "weight_sink"  : 1,
        "adam_betas"   : (0.9, 0.999),
    }
    # Define parameters to be optimized by optuna.
    lr             = trial.suggest_loguniform('lr', *args.lr)
    loss_weight    = trial.suggest_uniform('loss_weight', 0, 1)
    weight_sink    = trial.suggest_uniform('weight_sink', 0, 1)
    batch          = trial.suggest_categorical('batch', args.batch)
    hyperparams.update({"lr": lr, "loss_weight": loss_weight})

    # Load the data.
    ae_data = data.AE_data(args.data_folder, args.norm, args.nevents)
    train_loader = ae_data.get_loader("train", device, batch, True)
    valid_loader = ae_data.get_loader("valid", device, None, True)

    # Define the model and prepare the output folder.
    (hyperparams['ae_layers']).insert(0, ae_data.nfeats)
    model = util.choose_ae_model(args.aetype, device, hyperparams)

    min_valid = \
        optuna_train(train_loader, valid_loader, model, args.epochs, trial)

    return min_valid

if __name__ == '__main__':
    # Start the optuna study.
    print('\n')
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="loss-weight-optimizer",
        sampler=sampler, direction='minimize',
        pruner=optuna.pruners.HyperbandPruner())

    study.optimize(optuna_objective, n_trials=200)

    comp_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of complete trials: ", len(comp_trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("\n\nParams: ")
    for key, value in trial.params.items(): print("{}: {}".format(key, value))
