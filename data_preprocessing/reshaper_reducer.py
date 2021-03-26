# Reshape events to have a single row feature vector per event to feed the AE.
# The output is saved in dir 'input_ae'. There is a 'reduce events' step where
# cuts are applied to reduce the number of events. There is a 'reshape_events'
# step where the feature arrays are flattened to prepare them to be fed into
# the auto encoder.
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--nevents", type=int, default=int(3.6e5),
    help="Number of samples to be generated. Train + valid. + test.")
parser.add_argument("--fileFlag", type=str, default='',
    help="Output file flag.")
parser.add_argument("--noCart", action='store_true',
    help='Enable to eliminate px,py,pz.')
args = parser.parse_args()

def main():
    sig_data = import_data('sig_npy_sample')
    bkg_data = import_data('bkg_npy_sample')

    print('------Generating Training & Testing dataset------')

    sig_data = elim_cfeats(*sig_data) if args.noCart else sig_data
    bkg_data = elim_cfeats(*bkg_data) if args.noCart else bkg_data

    number_events = args.nevents
    sig_feats = reduce_events(sig_data, number_events)
    bkg_feats = reduce_events(bkg_data, number_events)

    sig_feats_reshaped = reshape_events(sig_feats)
    bkg_feats_reshaped = reshape_events(bkg_feats)

    features = ["JETS", "MET", "LEPS"]
    sig_feats_out = prepare_output(features, sig_feats_reashaped)
    bkg_feats_out = prepare_output(features, bkg_feats_reashaped)

    outdir = 'input_ae/'
    if not(os.path.exists(outdir)): os.mkdir(outdir)
    np.save(outdir + "/raw_sig" + args.fileFlag + ".npy", sig_feats_out)
    np.save(outdir + "/raw_bkg" + args.fileFlag + ".npy", bkg_feats_out)

def import_data(infolder):
    """
    Imports the .npy data from a given input folder. The last 3 jets are
    discarded since they mostly contain initial+final state radiation.

    @returns :: Arrays containing the different types of data associated
                with the studied events.
    """
    print('Input folder:', infolder)

    jets = np.load(infolder + "jets.npy"); jets = jets[:, :-3, :]
    met  = np.load(infolder + 'met.npy')
    leps = np.load(infolder + 'leps.npy')
    evdesc = np.load(infolder + 'evdesc.npy')

    return [jets, met, leps, evdesc]

def elim_cfeats(jets, met, leps, evdesc):
    """
    Eliminate the cartezian features of the data, namely px, py, and pz.

    @jets :: Vector containing the jet information.
    @met  :: Vector containing the met information
    @leps :: Vector containing the lepton information.

    @returns :: The simplified vectors.
    """
    jetElim = [4, 5, 6]; metElim = [2, 3]; lepsElim = [4, 5, 6]
    print('jets:', jets.shape, '\nmet:', met.shape, '\nleps:', leps.shape)
    print('\nRemoving px,py,pz:\n')

    jets = np.delete(jets, jetElim, axis=2)
    met  = np.delete(met[:], metElim, axis=1)
    leps = np.delete(leps, lepsElim, axis=2)
    print('jets:', jets.shape, '\nmet:', met.shape, '\nleps:', leps.shape)

    return [jets, met, leps, evdesc]

def reduce_events(data, nsamples):
    """
    Reduce event size and apply event selection cuts. Use evdesc to apply the
    cuts demanding at least 4 showers and at least 2 btagged showers.
    BUT we want to keep same number of signal and background events for
    training after the cuts.

    @infolder   :: The input file in .npy format.
    @nsamples   :: The number of samples that one wants in the reduced data.

    @returns :: Ntuple with the reduced data vectors, excluding the event
                descrpition information.
    """
    try: jets, met, leps, evdesc = data
    except: TypeError("Invalid data object!")

    ind = np.logical_and(evdesc[:, 2] >= 2, evdesc[:, 1] >= 4)
    redJet = jets[ind]; redMet = met[ind]; redLeps = leps[ind]

    print('\nApplying EVENT SELECTION: n_events = {} ---> {}\n'
        .format(jets.shape[0], redJet.shape[0]))

    print('Map Jet btag = {0,1,...,7} (btagger efficiency stages) --> \
          {0,1} (btaged or not btaged)')
    redJet[:, :, -1] = np.where(redJet[:, :, -1] > 1, 1, 0)

    print('Requested number of samples (events): {}'.format(nsamples))
    return (redJet[:nsamples], redMet[:nsamples], redLeps[:nsamples])


def reshape_events(features):
    """
    Reshape the feature arrays contained in the tuple 'features'.

    @features :: Tuple containing the included features.

    @returns :: Tuple of the same size, but now only containing 1D arrays.
    """
    if not isinstance(features, tuple): raise TypeError("Need tuple for resh.")
    reshaped_features = []
    for feature in features:
        reshaped_features.append(feature.reshape(feature.shape[0], -1))
    return reshaped_features

def prepare_output(features, feats_reshaped):
    # Display the reshaped features and prepare them for saving to file.
    for i, feature in enumerate(feats_reshaped):
        print(features[i] + ":")
        print("Reshaped: {} ---> {} ".format(feat_in[i].shape, ifeat.shape))
        feat_out = np.hstack(feaths_reshaped)
        print("Finalized data vectors to AE: (events,features) = {}".
            format(feat_out.shape))

if __name__ == "__main__":
    main()
