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
    file_suffix = "lat_2d_fulldata_kde_fill_alpha65.pdf"
    out_folder = "latent_comparison/"
    ae_lat_pairplot = pairplot(ae_lat_sig, ae_lat_bkg, feat_i=7, feat_j=10, 
                               bbox=(0.135, 0.99),
                               filename=out_folder + "ae_" + file_suffix,
                               math_cord_bottom=(0.16, 0.135),
                               math_cord_top=(0.52, 0.575)
                               )
    hybrid_ae_pairplot = pairplot(hybrid_ae_lat_sig, hybrid_ae_lat_bkg, feat_i=7, 
                                  feat_j=10, bbox=(0.10, 0.99), 
                                  filename=out_folder + "gqC_" + file_suffix,
                                  math_cord_bottom=(0.13, 0.135),
                                  math_cord_top=(0.495, 0.58)
                                  )


def prepare_pairplot_data(lat_sig: np.ndarray, lat_bkg: np.ndarray, i: int = 0, j: int = 1) -> sns.PairGrid:
    """Loads the signal and background datasets and concatenates them into a single dataframe."""
    data_sig = pd.DataFrame({rf"$z_{{{i}}}$": lat_sig[:, i], rf"$z_{{{j}}}$": lat_sig[:, j], " ": " Sig."})
    data_bkg = pd.DataFrame({rf"$z_{{{i}}}$": lat_bkg[:, i], rf"$z_{{{j}}}$": lat_bkg[:, j], " ": " Bkg."})
    return pd.concat([data_sig, data_bkg])


def pairplot(data_sig: np.ndarray, data_bkg: np.ndarray, feat_i: int, feat_j: int, filename: str, 
             bbox=(0.125, 0.99), 
             math_cord_bottom: tuple = (0.3, 0.42), math_cord_top: tuple = (0.65, 0.9)):
    """Constructs the 2x2 pair plot of 2 chosen features. The figure contains the
    1d and 2d feature distributions.
    """
    data_feat = prepare_pairplot_data(data_sig, data_bkg, i=feat_i, j=feat_j)
    plot = sns.pairplot(data_feat, hue=" ", 
                        palette={" Sig.": "#648FFF", " Bkg.": "#DC267F"}, 
                        diag_kind="hist",
                        kind="kde",
                        plot_kws=dict(alpha=0.65, fill=True),
                        #plot_kws=dict(marker="o", linewidth=1, alpha=0.25),
                        diag_kws=dict(alpha=0.45, fill=True, stat="density", element="step", bins=35),
                        grid_kws=dict(diag_sharey=False),
                        #corner=True
                )
    
    for ax in plot.axes.flatten():
        if ax is not None:
            ax.tick_params(labelsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=18, math_fontfamily='cm')
        ax.set_ylabel(ax.get_ylabel(), fontsize=18, math_fontfamily='cm')

    plot.fig.text(math_cord_bottom[0], math_cord_bottom[1], rf"$\mathcal{{P}}(z_{{{feat_i}}}, z_{{{feat_j}}})$", 
                  ha ='left', fontsize = 15, math_fontfamily='cm')
    plot.fig.text(math_cord_top[0], math_cord_top[1], rf"$\mathcal{{P}}(z_{{{feat_j}}}, z_{{{feat_i}}})$", 
                  ha ='left', fontsize = 15, math_fontfamily='cm')

    handles = plot._legend_data.values()
    labels = plot._legend_data.keys()
    sns.move_legend(plot, "upper left", bbox_to_anchor=bbox)
    plot._legend.remove()
    plot.figure.legend(handles=handles, labels=labels, handletextpad=0.01, loc="upper left", bbox_to_anchor=bbox, frameon=False)
    plot.savefig(filename)
    
    return plot


if __name__ == "__main__":
    main()
