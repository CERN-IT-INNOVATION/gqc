from time import perf_counter
from .terminal_colors import tcols
import numpy as np
from . import qdata as qd
from . import plot
from . import util

def main(args):
    qdata = qd.qdata(
        args["data_folder"],
        args["norm"],
        args["nevents"],                    
        args["model_path"],
        test_events=args["n_test"],
        kfolds=5,
    )
    print(f'\n----------------\n Loading QSVM model: {args["qsvm_model"]}')
    qsvm = util.load_qsvm(args["qsvm_model"]+'model')
    
    #TODO specify output folder 
    print(tcols.OKCYAN +
          "\n\nComputing accuracies on kfolded test data..." +
          tcols.ENDC)
          
    #TODO with kernel.evaluate()
    test_folds = qdata.get_kfolded_data("test")
    print(test_folds.shape)
    scores = compute_model_scores(qsvm, test_folds, args["qsvm_model"])
    print(tcols.OKCYAN + "\n\nPlotting and saving ROC figure..." + tcols.ENDC)
    
    #plot.roc_plot(scores, qdata, args["output_folder"], args["display_name"])


def compute_model_scores(model, data_folds, output_folder) -> np.ndarray:
    """
    Computing the model scores on all the test data folds to construct
    performance metrics of the model, e.g., ROC curve and AUC.

    @model         :: The qsvm model to compute the score for.
    @data_folds    :: Numpy array of kfolded data.
    @output_folder :: The folder where the results are saved.

    returns :: Array of the qsvm scores obtained.
    """
    scores_time_init = perf_counter()
    model_scores = np.array(
        [model.decision_function(fold) for fold in data_folds]
    )
    scores_time_fina = perf_counter()
    print(f"Completed in: {scores_time_fina-scores_time_init:2.2e} s")

    path = "models/" + output_folder + "/y_score_list.npy"

    print("Saving model scores array in: " + path)
    np.save(path, model_scores)

    return model_scores