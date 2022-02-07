import optuna
from optuna.trial import TrialState

from . import util
from . import data
from .terminal_colors import tcols


def main(args):
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    name = args["study_name"]

    study = optuna.create_study(
        study_name=name, sampler=sampler, direction="minimize", pruner=pruner
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args["ntri"])

    trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of complete trials: ", len(trials))

    print("Best trial: ")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("\n\nParams: ")
    for key, value in trial.params.items():
        print("{}: {}".format(key, value))


def optuna_train(train_loader, valid_loader, model, epochs, trial, woptim) -> float:
    """
    Training the autoencoder in a way that is compatible with optuna.
    @train_loader :: Pytorch loader object containing training data.
    @train_loader :: Pytorch loader object containing validation data.
    @model        :: The ae model to be trained.
    @epochs       :: The number of epochs to train the model for.
    @trial        :: The optuna trial object, used in pruning.

    returns :: The best loss depending on various factors, such as if
        the weights in the loss are optimised, type of ae that is
        optimised, etc.
    """
    print(tcols.OKCYAN + "Training the AE model to be optimized..." + tcols.ENDC)
    model.instantiate_adam_optimizer()
    model.network_summary()
    model.optimizer_summary()

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

        if model.early_stopping():
            return model.best_valid_loss
        model.print_losses(epoch, epochs, train_loss, valid_loss)

        trial.report(train_loss.item(), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if woptim:
        if model.hp["ae_type"] in ["classifier", "classvqc", "sinkclass"]:
            return min(model.all_class_loss)

        if model.hp["ae_type"] in ["variational", "sinkhorn"]:
            return min(model.all_recon_loss)

    return model.best_valid_loss


def objective(trial, args) -> float:
    """
    Wrapper of the normal training such that it agrees with what optuna
    is trying to do. The data and model are loaded and the hyperparameter
    ranges to be explored by optuna are set.
    @trial   :: Optuna trial object.
    @args    :: Dictionary with all the hyperparameters of the considered ae.

    returns :: The minimum validation loss.
    """
    device = util.define_torch_device()
    # Define parameters to be optimized by optuna.
    lr = trial.suggest_loguniform("lr", *args["lr"])
    loss_weight = trial.suggest_uniform("loss_weight", 1, 1)
    weight_sink = trial.suggest_uniform("weight_sink", 1, 1)
    batch = trial.suggest_categorical("batch", args["batch"])
    args.update({"lr": lr, "loss_weight": loss_weight, "weight_sink": weight_sink})

    # Load the data.
    ae_data = data.AE_data(args["data_folder"], args["norm"], args["nevents"])
    train_loader = ae_data.get_loader("train", device, batch, True)
    valid_loader = ae_data.get_loader("valid", device, None, True)

    # Define the model and prepare the output folder.
    (args["ae_layers"]).insert(0, ae_data.nfeats)
    model = util.choose_ae_model(args["aetype"], device, args)

    min_valid = optuna_train(
        train_loader, valid_loader, model, args["epochs"], trial, args["woptim"]
    )

    return min_valid
