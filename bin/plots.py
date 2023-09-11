# Plot latent space representation for the vanilla autoencoder and gqCompression model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def main():
    ae_path = "/data/vabelis/ae_qml/bin/trained_aes/vanilla_ae_nobtag/latent_plots"
    hybrid_ae_path = "/data/vabelis/ae_qml/bin/trained_vqcs/hybrid_ae_20k_4q_r2_67to16_c0.7/latent_plots"
    ae_lat_sig = np.load(ae_path + "/sig.npy")
    ae_lat_bkg = np.load(ae_path + "/bkg.npy")
    hybrid_ae_lat_sig = np.load(hybrid_ae_path + "/sig.npy")
    hybrid_ae_lat_bkg = np.load(hybrid_ae_path + "/bkg.npy")
    ae_lat_pairplot = pairplot(ae_lat_sig, ae_lat_bkg, feat_i=7, feat_j=10, filename="ae_lat_2d.pdf")
    hybrid_ae_pairplot = pairplot(hybrid_ae_lat_sig, hybrid_ae_lat_bkg, feat_i=7, feat_j=10, bbox=(0.09, 0.99), filename="gqC_lat_2d_50dpi.pdf")



def prepare_pairplot_data(lat_sig: np.ndarray, lat_bkg: np.ndarray, i: int = 0, j: int = 1) -> sns.PairGrid:
    """Loads the signal and background datasets and concatenates them into a single dataframe."""
    data_sig = pd.DataFrame({rf"$z_{{{i}}}$": lat_sig[:, i], rf"$z_{{{j}}}$": lat_sig[:, j], " ": "Sig."})
    data_bkg = pd.DataFrame({rf"$z_{{{i}}}$": lat_bkg[:, i], rf"$z_{{{j}}}$": lat_bkg[:, j], " ": "Bkg."})
    return pd.concat([data_sig, data_bkg])


def pairplot(data_sig: np.ndarray, data_bkg: np.ndarray, feat_i: int, feat_j: int, filename: str, bbox=(0.125, 0.99)):
    """Constructs the 2x2 pair plot of 2 chosen features. The figure contains the
    1d and 2d feature distributions.
    """
    data_feat = prepare_pairplot_data(data_sig, data_bkg, i=feat_i, j=feat_j)
    plot = sns.pairplot(data_feat, hue=" ", 
                        palette={"Sig.": "#648FFF", "Bkg.": "#DC267F"}, 
                        diag_kind="hist",
                        plot_kws=dict(marker="o", linewidth=1, alpha=0.25),
                        diag_kws=dict(alpha=0.45, fill=True, stat="density", element="step", bins=60),
                )
    
    for ax in plot.axes.flatten():
        ax.tick_params(labelsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)

    handles = plot._legend_data.values()
    labels = plot._legend_data.keys()
    #sns.move_legend(plot, "upper left", bbox_to_anchor=bbox)
    plot._legend.remove()
    plot.figure.legend(handles=handles, labels=labels, handletextpad=0.01, loc="upper left", bbox_to_anchor=bbox, frameon=False)
    plot.savefig(filename, dpi=50)
    
    return plot


if __name__ == "__main__":
    main()