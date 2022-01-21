# Does the plotting of the qvsm.
import matplotlib.pyplot as plt
import numpy as np

from sklearn import metrics
from .terminal_colors import tcols


def roc_plot(scores, qdata_loader, output_folder, model_name):
    """
    Plot the ROC of a given vqc model, given kfolded data.
    Also calculate the AUC of the respective ROC and display it.
    @model_dictionary :: Python dicitionary of the model and and object.
    @qdata_loader     :: Data class that contains the kfolded data.
    @output_folder    :: Name of the folder where the plot will be saved.
    """

    f1 = plt.figure(1, figsize=(10, 10))
    plt.rc("xtick", labelsize=20)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=20)  # fontsize of the tick labels
    plt.rc("axes", titlesize=22)  # fontsize of the axes title
    plt.rc("axes", labelsize=22)  # fontsize of the x and y labels
    plt.rc("legend", fontsize=22)  # legend fontsize
    plt.rc("figure", titlesize=22)  # fontsize of the figure title

    y_scores = scores
    auc = np.array(
        [
            metrics.roc_auc_score(qdata_loader.ae_data.tetarget, y_score)
            for y_score in y_scores
        ]
    )
    auc_mean, auc_std = np.mean(auc), np.std(auc)
    print(f"AUC's: {auc}")
    print(tcols.OKGREEN +
          f"AUC (mean) = {auc_mean} +/- {auc_std}" +
          tcols.ENDC)
    y_scores_flat = y_scores.flatten()
    fpr, tpr, _ = metrics.roc_curve(
        np.tile(qdata_loader.ae_data.tetarget, qdata_loader.kfolds),
        y_scores_flat,
    )
    plt.plot(
        fpr,
        tpr,
        label=model_name
        + fr": AUC = {auc_mean:.3f} $\pm$ \
        {auc_std:.3f}",
    )

    plt.title(
        r"$N^{train}$"
        + f"={qdata_loader.ntrain},"
        + r" $N^{test}$"
        + f"={qdata_loader.ntest} ($x 5$)",
        loc="left",
    )
    plt.xlabel("Background Efficiency (FPR)")
    plt.ylabel("Signal Efficiency (TPR)")

    x = np.linspace(0, 1, num=50)
    plt.plot(x, x, "--", color="k", label="Random Classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.tight_layout()
    plt.legend()
    f1.savefig("qsvm_models/" + output_folder + "/roc_plot.pdf")
    plt.close()
