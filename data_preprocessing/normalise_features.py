# Use different feature normalisations to see how it affects the autoencoder.

import numpy as np
import argparse
from sklearn import preprocessing

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--infiles", type=str, default=infiles, nargs=2,
    help="Path to the files.")
parser.add_argument('--outfile', type=str, default='',
    help='Path to output file.')
parser.add_argument('--fileFlag', type=str, default='',
    help='Fileflag to concatenate to outputFiles.')
args = parser.parse_args()

if __name__ == "__main__":
    infile_bkg, infile_sig = args.infiles
    print('Normalizing: ' + infile_bkg + ' and ' + infile_sig)
    scaler = preprocessing.StandardScaler().fit(infile_sig)

    outfile = args.outfile
    print('Output: ', outfile)
