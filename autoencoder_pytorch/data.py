# Handling of all the data needs of the autencoder.
import torch
import torch.nn as nn

import numpy as np
import os, warnings, time

from terminal_colors import tcols


class AE_data():
    def __init__(self, data_folder, norm_name, nevents,
        train_events=-1, valid_events=-1, test_events=-1):

        self.norm_name   = norm_name
        self.nevents     = nevents
        self.data_folder = data_folder

        self.train_data = self.get_numpy_data("train")[:train_events, :]
        self.valid_data = self.get_numpy_data("valid")[:valid_events, :]
        self.test_data  = self.get_numpy_data("test")[:test_events, :]

        self.train_target = self.get_numpy_target("train")[:train_events, :]
        self.valid_target = self.get_numpy_target("valid")[:valid_events, :]
        self.test_target  = self.get_numpy_target("test")[:test_events, :]

        self.nfeats = self.train_data.shape[1]

        self.success_message()

    def get_data_file(self, data_type):
        return "x_data_" + self.norm_name + "_norm_" + self.nevents + "_" + \
            data_type + ".npy"

    def get_target_file(self, data_type):
        return "y_data_" + self.norm_name + "_norm_" + self.nevents + "_" + \
            data_type + ".npy"

    def get_numpy_data(self, data_type):
        data = []
        path = os.path.join(self.data_folder, self.get_data_file(data_type))
        try: data = np.load(path)
        except: print(tcols.WARNING + data_type + " data file not found!" +
                      tcols.ENDC)

        return data

    def get_numpy_target(self, data_type):
        data = []
        path = os.path.join(self.data_folder, self.get_target_file(data_type))
        try: data = np.load(path)
        except: print(tcols.WARNING + data_type + " data file not found!" +
                      tcols.ENDC)

        return data

    def success_message(self):
        print("\n----------------")
        print(tcols.OKGREEN + "AE data loading complete:" + tcols.ENDC)
        print(f"Training data size: {self.train_data.shape[0]:.2e}")
        print(f"Validation data size: {self.valid_data.shape[0]:.2e}")
        print(f"Validation data size: {self.test_data.shape[0]:.2e}")
        print("----------------\n")

    def get_pytorch_dataset(self, data_type):
        switcher = {
            'train': lambda: self.make_set(self.train_data, self.train_target),
            'valid': lambda: self.make_set(self.valid_data, self.valid_target),
            'test':  lambda: self.make_set(self.test_data,  self.test_target)
        }
        dataset = switcher.get(data_type, lambda: None)()
        if dataset is None:
            raise TypeError("Dataset must be train, valid, or test!!")

        return dataset

    @staticmethod
    def make_set(data, target):
        data    = torch.Tensor(data)
        target  = torch.Tensor(target)
        return torch.utils.data.TensorDataset(data, target)

    def get_loader(self, data_type, device, batch_size=None, shuffle=True):
        """
        Convert numpy arrays of training/validation/testing data into pytroch
        objects ready to be used in training the autoencoder.

        @device     :: String if the training is done on cpu or gpu.
        @batch_size :: Int of the batch size used in training.
        @shuffle    :: Bool of whether to shuffle the data or not.

        @returns :: Pytorch objects to be passed to the autoencoder for training.
        """
        dataset = self.get_pytorch_dataset(data_type)
        if batch_size is None: batch_size = len(dataset)

        if device == 'cpu':
            pytorch_loader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size, shuffle=shuffle)
        else:
            pytorch_loader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size, shuffle=shuffle, pin_memory=True)

        return pytorch_loader

    @staticmethod
    def split_sig_bkg(data, target):
        # Split dataset into signal and background samples using the target.
        # The target is supposed to be 1 for every signal and 0 for every bkg.
        sig_mask = (target == 1); bkg_mask = (target == 0)
        data_sig = data[sig_mask, :]
        data_bkg = data[bkg_mask, :]

        return data_sig, data_bkg
