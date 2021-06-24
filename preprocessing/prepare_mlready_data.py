# Reads in one signal and one background .h5 file containing the simulated
# signal and background data of a studied pp process. Selects the interesting
# variables, applies selection cuts on the raw data, formats it, and casts it
# into a flat 2D array that is saved in the x_data_ray.npy file.

# Caution, you need to either have a lot of ram or a lot of swap memory to run
# this script. Pandas is not the best tool for working with large datasets.
# This should be changed in the future, but it is good enough for now.
import argparse, os, glob, itertools
import pandas as pd
import numpy as np
import data_features_definitions

# Disable the pandas warning when working on a copy. Re-enable if you care
# about what you do to a dataframe copy making it back to original.
pd.options.mode.chained_assignment = None  # default='warn'


parser = argparse.ArgumentParser()
parser.add_argument("--infile_prefix", type=str, required=True, action="store",
    help="The prefix of .h5 files (which should have Sig.h5 or Bkg.h5 after).")
parser.add_argument("--outdir", type=str, action="store", default="data/",
    help="The output directory.")
parser.add_argument("--datatype", type=str, required=True,
    choices=["cms_0l","cms_1l","cms_2l","delphes_1l","delphes_2l","delphes_had"
             "mass_1l", "class_2016_1l", "class_2016_2l", "cms_2017_1l",
             "cms_2017_2l"], help="Choose where the data comes from..")
args = parser.parse_args()

def main():
    features = data_features_definitions.opts()
    features.update(data_features_definitions.choose_data_type(args.datatype))

    # Specify which variables are interesting for the imported data and specify
    # the selection criteria.
    globals().update(features)
    globals()['selection'] = "nleps == 1 & (nbtags >= 2) & (njets >= 4)"
    globals()['njets'] = 7

    # Load .h5 data. Create the output directory.
    data_sig = load_files(args.infile_prefix + "Sig.h5")
    data_bkg = load_files(args.infile_prefix + "Bkg.h5")
    if not os.path.exists(args.outdir): os.makedirs(args.outdir)

    # Preprocess the data and create flat arrays.
    sig, y_sig = make_flat_numpy_array(data_sig)
    bkg, y_bkg = make_flat_numpy_array(data_bkg, False)

    # Save to .npy arrays.
    np.save(os.path.join(args.outdir, "x_data_sig"), sig)
    np.save(os.path.join(args.outdir, "x_data_bkg"), bkg)

def read_single_file(path):
    # Load a single .h5 file with chunksize 10000.
    print("Loading hdf file {0}...".format(path))
    return pd.read_hdf(path, chunksize="1000000")

def load_files(path):
    # Load a single file specified by path or load all .h5 files in the folder
    # specified by path.
    if path.endswith(".h5"): return read_single_file(path)

    file_paths = sorted(glob.glob(path + '/data*.h5'))
    for path in file_paths:
        if file_paths.index(path) == 0: data = read_single_file(path)
        else: data = itertools.chain(data, read_single_file(path))

    print("All data is loaded!")

    return data

def map_jet_values(data):
    # Map the jet btag from being 0-10 to being 0 when there are no jets
    # and being one when there are any jets.
    for idx in range(10):
        data["jets_btag_" + str(idx)] = (data['jets_btag_{0}'.format(idx)]>1)
        data["jets_btag_" + str(idx)] = \
            data["jets_btag_" + str(idx)].astype(float)

    return data

def jet_formatting(data, flats):
    """
    Formatting the jets features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, 2D flatly.

    @returns :: The updated flats array.
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

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, 2D flatly.

    @returns :: The updated flats array.
    """
    print('Formatting leptons...')

    number_lep_feats = len(lep_feats)
    lepsa = data[["leps_%s_%d" % (feat,lep) for lep in range(nleps)
        for feat in lep_feats]].values
    if flats[2].size == 0:   flats[2] = lepsa
    else:                    flats[2] = np.concatenate((flats[2], lepsa))
    lepsa = lepsa.reshape(-1, nleps, number_lep_feats)
    print('Lepton formatting done. Shape of the proc lept array:', lepsa.shape)

    return flats

def met_formatting(data, flats):
    """
    Formatting the meta features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting metadata features...')

    data["met_px"] = data["met_" + met_feats[1]] * \
        np.cos(data["met_"+met_feats[0]])
    data["met_py"] = data["met_" + met_feats[1]] * \
        np.sin(data["met_"+met_feats[0]])
    meta = data[["met_%s" % feat for feat in met_feats]].values
    if flats[1].size == 0:  flats[1] = meta
    else:                   flats[1] = np.concatenate((flats[1], meta))
    print('Metadata formatting done. Shape of the proc met array:', meta.shape)

    return flats

def make_flat_features(flats, is_signal=True):
    """
    Make the flats object constructed earlier into a flat 2D array, and generate
    vector of 1s or 0s depending on the type of data (signal or bkg).

    @flats     :: The flats array as constructed in make_flat_numpy_array.
    @is_signal :: Bool of whether we are dealing with signal or background.

    @returns :: A 2D matrix, containing rows as many as the number of events
        and columns as many as features.
    """
    print('\n-----------\nMaking flat features...')
    flat_array = np.hstack(flats)

    nevents = flat_array.shape[0]
    if is_signal: y = np.ones(nevents)
    else:         y = np.zeros(nevents)

    return flat_array, y

def make_flat_numpy_array(data, is_signal=True):
    """
    Take the loaded .h5 datasets and save the important features chunk by
    chunk. The flats 2D array elements are as follows:
    [0] - The jets features.
    [2] - The lepton features.
    [1] - The meta features.

    @data      :: The .h5 imported data.
    @is_signal :: Bool of whether the data is signal (True) or background.

    @returns   :: A python list containing the feature arrays (flats).
    """
    chunk_nb = 0; flats = [np.array([]), np.array([]), np.array([])]
    for chunk in data:
        chunk_nb += 1
        print("\n----------\nProcessing chunk number {0}.\n".format(chunk_nb))
        chunk = map_jet_values(chunk)

        if selection is not None: chunk = chunk.query(selection)
        if jet_feats is not None: flats = jet_formatting(chunk, flats)
        if lep_feats is not None: flats = lep_formatting(chunk, flats)
        if met_feats is not None: flats = met_formatting(chunk, flats)

    return make_flat_features(flats, is_signal)

if __name__ == "__main__":
    main()
