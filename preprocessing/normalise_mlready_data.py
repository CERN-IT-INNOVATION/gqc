# Imports an .npy file as constructed in prepare_mlready_data.npy or any
# .npy file that contains a 2D matrix with number of events and features.
# Sklearn is then used to normalize the data sets. Each normalized copy
# is saved as an .npy file. There's also the option of splitting the
# data sets into training and testing subsamples.
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

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--infile_data", type=str, required=True,
    help="Path to the .npy file containing data to be normalized.")
parser.add_argument("--infile_target", type=str, required=True,
    help="Path to the .npy file containing target associated with data.")

args = parser.parse_args()

def split_and_save(data, target, name):
    # Splits a given data set into training, testing, and validation samples
    # that then are saved with corresponding names in .npy files.
    print("Splitting data into training and testing, then saving...")
    x_train, x_test, y_train, y_test = train_test_split(data, target,
        test_size=0.1, shuffle=True)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,
        test_size=0.111111, shuffle=True)

    base_filename_y = os.path.splitext(args.infile_target)[0]

    np.save(name + "_train", x_train)
    np.save(name + "_test", x_test)
    np.save(name + "_valid", x_vali)
    np.save(base_filename_y + "_train", y_train)
    np.save(base_filename_y + "_test", y_test)
    np.save(base_filename_y + "_valid", y_valid)

def apply_normalization(norm_method, norm_name, data, target, base_fn):

    data_minmax = norm_method().fit_transform(data)
    np.save(base_filename + "_" + norm_name, data_minmax)
    split_and_save(data_minmax, target, base_filename + "_minmax_norm")
if __name__ == "__main__":
    print('Normalizing the data file: ' + args.infile_data)
    print('Target imported: ' + args.infile_target)

    data   = np.load(args.infile_data)
    target = np.load(args.infile_target)
    base_filename = os.path.splitext(args.infile_data)[0].replace('_raw', '')

    print("\nApplying MinMax normalization...")
    apply_normalization(MinMaxScaler, "minmax_norm", data, target, base_fn)
    print("\nApplying Standard normalization...")
    apply_normalization(StandardScaler, "std_norm", data, target, base_fn)
