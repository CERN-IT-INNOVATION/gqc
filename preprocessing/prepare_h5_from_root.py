# Takes the .root files as input and formats them. Outputs .h5 files that are
# then taken by format.py and formatted further.

# TO BE TESTED BY VASILIS ON THE ROOT FILES BEFORE MERGING!
from __future__ import print_function
import pandas as pd
import numpy as np
import glob, argparse, os
import ROOT
import root_numpy

parser = argparse.ArgumentParser()
parser.add_argument( "--input", type=str,
    required=True, action="store", nargs='+',
    help="Input data_folder or list of files.")
parser.add_argument("--output", type=str,
    default="data.h5", action="store",
    help="The output data file.")
args = parser.parse_args()
output_folder = 'preprocessed_data'

def main():
    data_frame = load_data(args.input); print(data_frame.columns)

    for ilep in range(2):  construct_four_momentum(data_frame, 'leps', ilep)
    for ijet in range(10): construct_four_momentum(data_frame, 'jets', ijet)

    if not "nbtags" in list(data_frame): data_frame['nbtags'] = sum(
        data_frame['jets_btag_{0}'.format(i)]>1 for i in range(10))
    recompute_bbnMatch(data_frame)

    for ilep in range(2):  construct_four_momentum(data_frame, 'leps', ilep)
    for ijet in range(10): construct_four_momentum(data_frame, 'jets', ijet)

    print("Saving {0} to {1}".format(data_frame.shape, args.output))
    print(list(data_frame.columns))
    data_frame.to_hdf(args.output, key='data_frame', format='t', mode='w')


def check_single_folder(data_folder):
    if len(data_folder) == 1 and os.path.isdir(data_folder[0]):
        return 1

    return 0

def load_data(data_folder):
    """
    Loads the data from .h5 format files produced by format.py.

    @data_folder :: String with path to the folder where the data is.

    @returns :: Pandas data frame object with all the files.
    """
    if check_single_folder(data_folder):
        print("Loading files from data folder {0}.".format(data_folder[0]))
        data_files = sorted(glob.glob(data_folder[0] + '/*flat*.root'))
    elif isinstance(data_folder, list): data_files = list(data_folder)
    else: raise TypeError("Given data folder is not correct!")

    print("Data files: ")
    for file in data_files: print(file)
    data_frame= pd.DataFrame(root_numpy.root2array(data_files,treename="tree"))

    return data_frame

def construct_four_momentum(data, ptype, idx):
    """
    Constructs the 4-momentum from the given data and stores it in the same
    data frame. Works with pointers so you don't need to return anything.

    @data  :: Imported data frame object.
    @ptype :: Str type of particle we are dealing with, lepton or jet (quarks).
    @idx   :: The identifier index of the particle.
    """
    idx  = "" if idx is None else "_%d" % idx
    pt   = data['%s_pt%s'   % (ptype, idx)]
    eta  = data['%s_eta%s'  % (ptype, idx)]
    phi  = data['%s_phi%s'  % (ptype, idx)]
    mass = data['%s_mass%s' % (ptype, idx)]
    data["%s_px%s" % (ptype, idx)] = pt * np.cos(phi)
    data["%s_py%s" % (ptype, idx)] = pt * np.sin(phi)
    data["%s_pz%s" % (ptype, idx)] = pt * np.sinh(eta)
    data["%s_en%s" % (ptype, idx)] = np.sqrt(mass**2 +
        (1 + np.sinh(eta)**2) * pt**2)

def construct_invariant_mass(data, ptype1, idx1, ptype2, idx2):
    """
    Constructs the invariant mass from the given data and stores it in the same
    data frame. Works with pointers so you don't need to return anything.

    @data  :: Imported data frame object.
    @ptype :: Str type of particle we are dealing with, lepton or jet (quarks).
    @idx   :: The identifier index of the particle.
    """
    im = ""

    if idx1 is None: idx1 = ""
    else: idx1 = "_%d" % idx1; im += idx1
    if idx2 is None: idx2 = ""
    else:
        im  += "%d" % idx2 if im.startswith("_") else "_%d" % idx2
        idx2 = "_%d" % idx2

    px = data["%s_px%s" % (ptype1,idx1)] + data["%s_px%s" % (ptype2,idx2)]
    py = data["%s_py%s" % (ptype1,idx1)] + data["%s_py%s" % (ptype2,idx2)]
    pz = data["%s_pz%s" % (ptype1,idx1)] + data["%s_pz%s" % (ptype2,idx2)]
    en = data["%s_en%s" % (ptype1,idx1)] + data["%s_en%s" % (ptype2,idx2)]
    data["%s_%s_m2%s" % (ptype1,ptype2,im)] = en*en - px*px - py*py - pz*pz

def recompute_bbnMatch(data_frame):
    # Recompute the bjet tag flags to go from 0 to 10 depending on how confi
    # the algorithm is that this is actually a bjet.
    if "bb_nMatch" in list(data_frame):
        data_frame.drop(['bb_nMatch'], axis=1)
        print("Deleted pre-defined bb_nMatch.")
    data_frame['bb_nMatch'] = sum(data_frame['jets_matchFlag_{0}'.format(i)]
        for i in range(10))
    print("Re-computed bb_nMatch.")

if __name__ == "__main__":
    main()
