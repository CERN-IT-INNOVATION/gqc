# Reshape events to have a single row feature vector per event to feed the AE
# From format.py to map array indeces to the variable names:
'''
#Variables: 
        jet_feats = ["pt","eta","phi","en","px","py","pz","btag"],
        njets = 10,
        met_feats = ["phi","pt","px","py"],
        nleps = 1,
        lep_feats = ["pt","eta","phi","en","px","py","pz"],
        evdesc_feats = ["nleps", "njets", "nbtags", "nMatch_wq", "nMatch_tb", "nMatch_hb"],#Don't fit in AE!!
'''
import numpy as np
import os
import argparse


def reduce_events(infile, nend, outdir=None, nstart=0, elimCart=False):
    # Reduce event size and apply event selection cuts
    # Using evdesc info apply phase space realistic cuts:
    # BUT we want to keep same number of signal and background events for training after the cuts.
    jets = np.load(infile+"jets.npy")
    # 1. discard last 3 jets they are mostly initial+final state radiation
    jets = jets[:, :-3, :]
    met = np.load(infile+'met.npy')
    leps = np.load(infile+'leps.npy')
    evdesc = np.load(infile+'evdesc.npy')
    print('Input file:', infile)

    # Eliminate px,py,pz features:
    if elimCart:
        jetElim = [4, 5, 6]
        metElim = [2, 3]
        lepsElim = [4, 5, 6]
        print('jets:', jets.shape)
        print('met:', met.shape)
        print('leps:', leps.shape)
        print('\nRemoving px,py,pz:\n')
        jets = np.delete(jets, jetElim, axis=2)
        met = np.delete(met[:], metElim, axis=1)
        leps = np.delete(leps, lepsElim, axis=2)
        print('jets:', jets.shape)
        print('met:', met.shape)
        print('leps:', leps.shape)

    # Save indices for the cuts and apply numpy [indices] to all arrays
    ind = np.logical_and(evdesc[:, 2] >= 2, evdesc[:, 1] >= 4)

    # event selection:
    redJet = jets[ind]
    redMet = met[ind]
    redLeps = leps[ind]

    print(
        '\nApplying EVENT SELECTION: n_events = {} ---> {}\n'.format(jets.shape[0], redJet.shape[0]))
    print('Requested number of samples (events): {}'.format(nend-nstart))

    return (redJet[nstart:nend], redMet[nstart:nend], redLeps[nstart:nend])


def reshape_events(a):  # single array or tuple(needed below for hstacking) of arrays as input
    # using -1 here instead of shape[1]*shape[2] e.g. 3d Jet array -> 2d Jet array
    # becayse it also includes automatically the reshaping of the non 3d arrays i.e. 2d MET
    if isinstance(a, tuple):
        i_r = []
        for i in a:
            i_r.append(i.reshape(i.shape[0], -1))
        return i_r
    else:
        a_r = a.reshape(a.shape[0], -1)
        return a_r


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # include defaults in -h
    parser.add_argument("--nevents", type=int, default=int(3.6e5),
                        help="number of samples to be generate. train+valid.+test")
    parser.add_argument("--fileFlag", type=str, default='',
                        help="Output file flag")
    parser.add_argument("--noCart", action='store_true',
                        help='Enable to eliminate px,py,pz')
    args = parser.parse_args()

    print('------Generating Training & Testing dataset------')
    samples = ['sig', 'bkg']
    n = args.nevents
    for isample in samples:
        print("\n_____"+isample+" samples_____")

        (jets, met, leps) = reduce_events(infile=isample +
                                          '_npy_sample/', nend=n, elimCart=args.noCart)
        feat_in = (jets, met, leps)

        # Map Jet btag = {0,1,...,7} --> {0,1}
        # Transform to boolean
        print(
            'Map Jet btag = {0,1,...,7} (btager efficiency stages) --> {0,1} (btaged or not btaged)')
        print('btag>1 --> 1 ; else --> 0')
        jets[:, :, -1] = np.where(jets[:, :, -1] > 1, 1, 0)
        feat_resh = reshape_events(feat_in)
        features = ["JETS", "MET", "LEPS"]

        for i, ifeat in enumerate(feat_resh):
            print(features[i]+":")
            print(
                "Reshaping: {} ---> {} ".format(feat_in[i].shape, ifeat.shape))

        feat_out = np.hstack(feat_resh)
        print("Finalized data vectors to AE: (events,features) = {}".format(
            feat_out.shape))
        outdir = 'input_ae/'
        if not(os.path.exists(outdir)):
            os.mkdir(outdir)
        np.save(outdir+"/raw_"+isample+args.fileFlag+".npy", feat_out)
