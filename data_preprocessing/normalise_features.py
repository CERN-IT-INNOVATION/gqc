# Normalise all features to [0,1] to see how it affects Auto-Encoder learning
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
import argparse

# Simultaneous trsf for sig and bkg:
# Using numpy broadcasting without for loops for efficiency (loops run in C and not in Python)


def normalizerMinMax(bkgfile, sigfile, outfile, fileflag):
    bkg, sig = np.load(bkgfile), np.load(sigfile)
    maxBkg, maxSig = np.amax(bkg, axis=0), np.amax(sig, axis=0)
    minBkg, minSig = np.amin(sig, axis=0), np.amin(bkg, axis=0)

    # Find global max/min for each feature for sig AND bkg
    # -->Correct normalization to retain the shapes of the pdfs
    maxTot = np.amax(np.vstack((maxBkg, maxSig)), axis=0)
    minTot = np.amin(np.vstack((minBkg, minSig)), axis=0)

    bkgNorm = (bkg-minTot)/(maxTot-minTot)
    sigNorm = (sig-minTot)/(maxTot-minTot)
    np.save(outfile+'Bkg'+fileflag+'.npy', bkgNorm)
    np.save(outfile+'Sig'+fileflag+'.npy', sigNorm)
    return bkgNorm, sigNorm


def normalizerMeanStd(bkgfile, sigfile, outfile, fileflag):
    bkg, sig = np.load(bkgfile), np.load(sigfile)
    if len(bkg) != len(sig):
        raise Exception('len(bkg) != len(sig) !')
    ntrain = int(0.8*len(bkg))
    meanBkg, meanSig = np.mean(
        bkg[:ntrain], axis=0), np.mean(sig[:ntrain], axis=0)
    mean = np.mean(np.vstack((meanBkg, meanSig)), axis=0)
    stdBkg, stdSig = np.std(bkg[:ntrain], axis=0), np.std(sig[:ntrain], axis=0)
    std = np.mean(np.vstack((stdBkg, stdSig)), axis=0)
    bkgNorm = (bkg-mean)/std
    sigNorm = (sig-mean)/std
    np.save(outfile+'BkgMeanSigma'+fileflag+'.npy', bkgNorm)
    np.save(outfile+'SigMeanSigma'+fileflag+'.npy', sigNorm)
    return bkgNorm, sigNorm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # include defaults in -h
    infileBkg, infileSig = 'input_ae/raw_bkg7.2e5.npy', 'input_ae/raw_sig7.2e5.npy'
    infiles = (infileBkg, infileSig)
    parser.add_argument("--infiles", type=str,
                        default=infiles, nargs=2, help="path to files")
    parser.add_argument('--fileFlag', type=str, default='',
                        help='fileFlag to concatenate to outputFiles')
    args = parser.parse_args()

    infileBkg, infileSig = args.infiles
    print('Normalizing: '+infileBkg+' & '+infileSig)
    outfile = "input_ae/trainingTestingData"
    bkgNorm, sigNorm = normalizerMinMax(
        infileBkg, infileSig, outfile, args.fileFlag)
    bkgNormStd, sigNormStd = normalizerMeanStd(
        infileBkg, infileSig, outfile, args.fileFlag)

    print('Output: ', outfile)
