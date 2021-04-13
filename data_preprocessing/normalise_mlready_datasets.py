# Imports an .npy file as constructed in prepare_mlready_data.npy or any
# .npy file that contains a 2D matrix with number of events and features.
# Sklearn is then used to normalize the data sets. Each normalized copy
# is saved as an .npy file.
import argparse
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import minmax_scale

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--infile", type=str, required=True,
    help="Path to the .npy file containing data to be normalized.")
args = parser.parse_args()


if __name__ == "__main__":
    infile = args.infile; print('Normalizing: ' + infile)

    data = np.load(infile); data.shape
    data_minmax = MinMaxScaler().fit_transform(data)

    base_filename = os.path.splitext(args.infile)[0]
    np.save(base_filename + "_minmax_norm", data_minmax)
