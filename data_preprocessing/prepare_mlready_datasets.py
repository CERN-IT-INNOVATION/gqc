# Takes the .h5 data files and formats them depending on what the data contains
import argparse, os, glob, itertools
import pandas as pd
import numpy as np
import data_features_definitions


parser = argparse.ArgumentParser()
parser.add_argument("--infile", type=str, required=True, action="store",
    help="The input folder/data file.")
parser.add_argument("--outdir", type=str, action="store",
    default="data/", help="The output directory.")
parser.add_argument("--datatype", type=str, required=True,
    choices=["cms_0l","cms_1l","cms_2l","delphes_1l","delphes_2l","delphes_had"
             "mass_1l", "class_2016_1l", "class_2016_2l", "cms_2017_1l",
             "cms_2017_2l"], help="Choose where the data comes from..")
args = parser.parse_args()

def main():
    # This code imports the data, formats it and stores it in h5 files.
    # An output folder with the given name in outdir will be created.
    # Additional selection cuts differing from standard ones are added.
    features = data_features_definitions.opts()
    features.update(data_features_definitions.choose_data_type(args.datatype))
    globals().update(features)
    selection = "nleps == 1 & (nbtags >= 2) & (njets >= 4)"; njets = 7

    # Load .h5 data. Create the output directory.
    data_sig = load_files(args.infile + "Sig.h5")
    data_bkg = load_files(args.infile + "Bkg.h5")
    os.makedirs(args.outdir)

    # Preprocess the data and create flat arrays.
    sig, y_sig = make_flat_numpy_array(data_sig)
    bkg, y_bkg = make_flat_numpy_array(data_bkg, False)

    # Prepare for training.
    x = np.concatenate((sig, bkg), axis=0)
    y = np.concatenate((y_sig, y_bkg), axis=0)

    # Save to .npy arrays.
    np.save(os.path.join(args.outdir, "x_data_raw"), x)
    np.save(os.path.join(args.outdir, "y_data_raw"), y)

def read_single_file(path):
    # Load a single .h5 file with chunksize 10000.
    print("Loading hdf file {0}...".format(path))
    return pd.read_hdf(path, chunksize="1000000")

def load_files(path):
    """
    Load a single file specified by path or load all .h5 files in the folder
    specified by path.
    """
    if path.endswith(".h5"): return read_single_file(path)

    file_paths = sorted(glob.glob(path + '/data*.h5'))
    for path in file_paths:
        if file_paths.index(path) == 0: data = read_single_file(path)
        else: data = itertools.chain(data, read_single_file(path))

    print("All data is loaded!")

    return data

def map_jets_values(data):
    # Map the jet values from being 0-10 to being 0 when there are no jets
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
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting jets...')

    jetsa = None
    onejet = list(range(njets)); number_jet_feats = len(jet_feats)
    jet_col = ["jets_%s_%d"%(feat,jet) for jet in onejet for feat in jet_feats]
    jetsa = data[jet_col].values
    flats.append(jetsa); jetsa = jetsa.reshape(-1, njets, number_jet_feats)
    print('Jet formatting done.', jetsa.shape)

    return flats

def lep_formatting(data, flats):
    """
    Formatting the lepton features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting leptons...')

    lepsa = None
    number_lep_feats = len(lep_feats)
    lepsa = data[["leps_%s_%d" % (feat,lep) for lep in range(nleps)
        for feat in lep_feats]].values
    flats.append(lepsa)
    lepsa = lepsa.reshape(-1, nleps, number_lep_feats)
    print('Lepton formatting done.', lepsa.shape)

    return flats

def met_formatting(data, flats):
    """
    Formatting the meta features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting metadata features...')

    meta = None
    data["met_px"] = data["met_" + met_feats[1]] * \
        np.cos(data["met_"+met_feats[0]])
    data["met_py"] = data["met_" + met_feats[1]] * \
        np.sin(data["met_"+met_feats[0]])
    meta = data[["met_%s" % feat for feat in met_feats]].values
    flats.append(meta)
    print('Metadata formatting done.', meta.shape)

    return flats

def make_flat_features(flats, is_signal=True):
    # Make the flats object constructed earlier into a flat array, and generate
    # vector of 1s or 0s depending on the type of data (signal or bkg).
    print('Making flat features...')
    flat_array = np.hstack(flats)

    nevents = flat_array.shape[0]
    if is_signal: y = np.ones(nevents)
    else:         y = np.zeros(nevents)

    return flat_array, y

def make_flat_numpy_array(data, is_signal=True):
    # Take the loaded .h5 data file and make a flat numpy array out of it
    # containing only the interesting features, processed such that they are
    # ready for the machine learning training.

    chunk_nb = 0; flats_array = []; y = []
    for chunk in data:
        chunk_nb += 1
        print("\n------------\nProcessing chunk number {0}".format(chunk_nb))
        chunk = map_jets_values(chunk)
        if selection is not None: chunk = chunk.query(selection)

        flats = []
        if jet_feats is not None: flats = jet_formatting(chunk, flats)
        if lep_feats is not None: flats = lep_formatting(chunk, flats)
        if met_feats is not None: flats = met_formatting(chunk, flats)

    return make_flat_features(flats, is_signal)

if __name__ == "__main__":
    main()
