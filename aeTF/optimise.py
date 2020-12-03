import autoTF
import optuna


def objective(trial):
	factor = trial.suggest_uniform('factor', 1e-2, 10);
	nlayers = trial.suggest_int('nlayers', 6, 10);
	error = autoTF.compute(nlayers, factor = factor)
	return error

study = optuna.create_study(direction = 'minimize')
study.optimize(objective,n_trials=200)



