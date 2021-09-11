# Takes the .root files as input and formats them, outputting .h5 files.

from __future__ import print_function
import pandas as pd
import numpy as np
import glob, argparse, os
import ROOT
import root_numpy

parser = argparse.ArgumentParser()
parser.add_argument( "--input", type=str,
    required=True, action="store", nargs='+',
    help="Input data file, folder, or list of files (root trees).")
parser.add_argument("--output", type=str,
    default="data.h5", action="store",
    help="The output data .h5 file name (include the extension).")

def main():
    data_frame = load_data(args.input)

    if not "nbtags" in list(data_frame): data_frame['nbtags'] = sum(
        data_frame['jets_btag_{0}'.format(i)]>1 for i in range(10))
    recompute_bbnMatch(data_frame)

    for ilep in range(2):  construct_four_momentum(data_frame, 'leps', ilep)
    for ijet in range(10): construct_four_momentum(data_frame, 'jets', ijet)

    print(f"Saving {data_frame.shape} to {args.output}")
    print(list(data_frame.columns))
    data_frame.to_hdf(args.output, key='data_frame', format='t', mode='w')

def check_single_folder(data_folder):
    """
    Checks if a folder was given as input.
    @data_folder :: String with the data folder name/path.

    returns :: True if it is a folder or false if not.
    """
    if len(data_folder) == 1 and os.path.isdir(data_folder[0]): return 1
    return 0

def load_data(data_string):
    """
    Loads the root files specified by the user.
    @data_string   :: String with path to the folder where the data is or a
        list of data file name(s).

    returns :: Pandas data frame object with all the root files.
    """
    if check_single_folder(data_string):
        print(f"Loading files from data folder {data[0]}.")
        data_files = sorted(glob.glob(data_string[0] + '/*flat*.root'))
    elif isinstance(data, list): data_files = list(data_string)
    else: raise TypeError("Given data string is not correct!")

    print("Data files: "); for file in data_files: print(file)
    data_frame = pd.DataFrame(root_numpy.root2array(data_files,treename="tree"))

    return data_frame

def construct_four_momentum(data, ptype, idx):
    """
    Constructs the 4-momentum from the given data and stores it in the same
    data frame.
    @data  :: Pandas data frame object containing loaded root data.
    @ptype :: Str type of particle we are dealing with, lepton or jet (quarks).
    @idx   :: The number of the lepton/jet.
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

def recompute_bbnMatch(data):
    """
    Recompute the bjet tag flags to go from 0 to 10 depending on how
    confident the algorithm is that this is actually a bjet.

    @data  :: Pandas data frame object containing loaded root data.
    """
    if "bb_nMatch" in list(data): data.drop(['bb_nMatch'], axis=1)

    data['bb_nMatch'] = \
        sum(data['jets_matchFlag_{0}'.format(i)] for i in range(10))

if __name__ == "__main__":
    args = parser.parse_args()
    main()
