# Reads in one signal and one background .h5 file containing the simulated
# signal and background data of the studied ttHbb process.
# Selects the interesting variables, applies selection cuts on the raw data,
# formats it, and casts it into a flat 2D array that is saved in the
# x_data_ray.npy file.

import argparse, os, glob, itertools
import pandas as pd
import numpy as np

# Disable the pandas warning when working on a copy. Re-enable if you care
# about what you do to a dataframe copy making it back to original.
pd.options.mode.chained_assignment = None  # default='warn'


parser = argparse.ArgumentParser()
parser.add_argument("--sig_file", type=str, required=True, action="store",
    help="The name of the signal .h5 file (including the extension).")
parser.add_argument("--bkg_file", type=str, required=True, action="store",
    help="The name of the background .h5 file (including the extension).")
parser.add_argument("--outdir", type=str, action="store",
    default="./ml_ready_unnormalised_data/",
    help="The output directory.")

# Variables of choice and selection cuts.
jet_feats = ["pt", "eta", "phi", "en", "px", "py", "pz", "btag"]
lep_feats = ["pt","eta","phi","en","px","py","pz"]
met_feats = ["phi", "pt", "px", "py"]
njets = 7
nleps = 1
selection = "nleps == 1 & (nbtags >= 2) & (njets >= 4)"

def main():
    data_sig = read_single_file(args.sig_file, "1000000")
    data_bkg = read_single_file(args.bkg_file, "1000000")
    print("Data has been loaded!")
    if not os.path.exists(args.outdir): os.makedirs(args.outdir)

    sig, y_sig = make_flat_numpy_array(data_sig)
    bkg, y_bkg = make_flat_numpy_array(data_bkg, False)

    np.save(os.path.join(args.outdir, "x_data_sig"), sig)
    np.save(os.path.join(args.outdir, "x_data_bkg"), bkg)

def read_single_file(path, chunksize):
    """
    Load a single .h5 file with a certain chunksize.
    @path       :: String path to the data.
    @chunksize  :: String of the number of rows imported at once into RAM.

    returns :: The loaded Pandas data frame object.
    """
    print(f"Loading h5 file {path}...")
    return pd.read_hdf(path, chunksize=chunksize)

def map_jet_btag_values(data):
    """
    Map the jet btag from being 0-10 to being 0 when there are no jets
    and being one when there are any number of jets.
    @data :: Pandas data frame object containing the data.

    returns :: The modified pandas data frame object.
    """
    for idx in range(10):
        data["jets_btag_" + str(idx)] = (data['jets_btag_{0}'.format(idx)]>1)
        data["jets_btag_" + str(idx)] = \
            data["jets_btag_" + str(idx)].astype(float)

    return data

def jet_formatting(data, flats):
    """
    Formatting the jets features.
    @data  :: Pandas data frame object containing the data.
    @flats :: Array containing the values of the different data features in 2D.

    returns :: The updated flats array.
    """
    print('Formatting jets...')

    onejet = list(range(njets)); number_jet_feats = len(jet_feats)
    jet_col = ["jets_%s_%d"%(feat,jet) for jet in onejet for feat in jet_feats]
    jetsa = data[jet_col].values
    if flats[0].size == 0:   flats[0] = jetsa
    else:                    flats[0] = np.concatenate((flats[0], jetsa))
    jetsa = jetsa.reshape(-1, njets,number_jet_feats)
    print('Jet formatting done. Shape of the proc jets array: ', jetsa.shape)

    return flats

def lep_formatting(data, flats):
    """
    Formatting the lepton features.
    @data  :: Pandas data frame object containing the data.
    @flats :: Array containing the values of the different data features in 2D.

    returns :: The updated flats array.
    """
    print('Formatting leptons...')

    number_lep_feats = len(lep_feats)
    lepsa = data[["leps_%s_%d" % (feat,lep) for lep in range(nleps)
        for feat in lep_feats]].values
    if flats[2].size == 0:   flats[2] = lepsa
    else:                    flats[2] = np.concatenate((flats[2], lepsa))
    lepsa = lepsa.reshape(-1, nleps, number_lep_feats)
    print('Lepton formatting done. Shape of the processed lept array:', lepsa.shape)

    return flats

def met_formatting(data, flats):
    """
    Formatting the meta features.
    @data  :: Pandas data frame object containing the data.
    @flats :: Array containing the values of the different data features in 2D.

    returns :: The updated flats array.
    """
    print('Formatting missing energy features...')

    data["met_px"] = data["met_" + met_feats[1]] * \
        np.cos(data["met_"+met_feats[0]])
    data["met_py"] = data["met_" + met_feats[1]] * \
        np.sin(data["met_"+met_feats[0]])
    meta = data[["met_%s" % feat for feat in met_feats]].values
    if flats[1].size == 0:  flats[1] = meta
    else:                   flats[1] = np.concatenate((flats[1], meta))
    print('Missing energy formatting done. Shape of the processed met array:', meta.shape)

    return flats

def make_flat_features(flats, is_signal=True):
    """
    Make a 2D flat array of all the selected features, with a row per event
    and the columns being each features (variable).
    @flats     :: 3D array, with each class of features being a 2D array.
    @is_signal :: Bool of whether we are dealing with sig or bkg events.

    returns :: A 2D matrix, containing rows for each event and columns for
        each feature selected to be part of the final data set.
        Also returns a 1D array filled with 0s for each bkg events and 1s for
        each signal event. The position of the 0 or 1 matches the row number
        in the 2D array mentioned earlier.
    """
    print('\n-----------\nMaking flat features...')
    flat_array = np.hstack(flats)

    nevents = flat_array.shape[0]
    if is_signal: y = np.ones(nevents)
    else:         y = np.zeros(nevents)

    return flat_array, y

def make_flat_numpy_array(data, is_signal=True):
    """
    Take the loaded .h5 dataset and save the features of interest chunk by
    chunk.

    @data      :: The .h5 imported data.
    @is_signal :: Bool of whether the data is signal (True) or background.

    @returns   :: A 2D python list containing the events (rows) and features
        (columns) pre-selected and formatted.
    """
    chunk_nb = 0; flats = [np.array([]), np.array([]), np.array([])]
    for chunk in data:
        chunk_nb += 1
        print("\n----------\nProcessing chunk number {0}.\n".format(chunk_nb))
        chunk = map_jet_btag_values(chunk)

        if selection is not None: chunk = chunk.query(selection)
        if jet_feats is not None: flats = jet_formatting(chunk, flats)
        if lep_feats is not None: flats = lep_formatting(chunk, flats)
        if met_feats is not None: flats = met_formatting(chunk, flats)

    return make_flat_features(flats, is_signal)

if __name__ == "__main__":
    args = parser.parse_args()
    main()
