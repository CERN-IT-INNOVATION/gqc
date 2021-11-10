# Loads the data and an autoencoder model. The original data is passed
# through the AE and the latent space is fed to the qsvm network.
import sys
import os
import numpy as np

from .terminal_colors import tcols
from autoencoders import data as aedata
from autoencoders import util as aeutil
sys.path.append("..")


class qdata:
    def __init__(self, data_folder, norm_name, nevents, model_path,
                 train_events=-1, valid_events=-1, test_events=-1, kfolds=0):

        device = 'cpu'
        model_folder = os.path.dirname(model_path)
        hp_file = os.path.join(model_folder, 'hyperparameters.json')
        hp = aeutil.import_hyperparams(hp_file)

        print(tcols.OKCYAN + "\nLoading training data:" + tcols.ENDC)
        self.ae_data = aedata.AE_data(data_folder, norm_name, nevents,
                                      train_events, valid_events, test_events)
        self.model = aeutil.choose_ae_model(hp['ae_type'], device, hp)
        self.model.load_model(model_path)

        self.ntrain = self.ae_data.train_data.shape[0]
        self.nvalid = self.ae_data.valid_data.shape[0]
        self.ntest = self.ae_data.test_data.shape[0]

        print(tcols.OKCYAN + "Loading k-folded validation data:" + tcols.ENDC)
        self.kfolds = kfolds
        self.ae_kfold_data = \
            aedata.AE_data(data_folder, norm_name, nevents, 0,
                           kfolds*valid_events, kfolds*test_events)

    def get_latent_space(self, datat) -> np.ndarray:
        """
        Get the latent space depending on the data set you want.
        @datat :: String of the data type.

        returns :: Output of the ae depending on the given data type.
        """
        if datat == 'train':
            return self.model.predict(self.ae_data.train_data)[0]
        if datat == 'valid':
            return self.model.predict(self.ae_data.valid_data)[0]
        if datat == 'test':
            return self.model.predict(self.ae_data.test_data)[0]

        raise TypeError("Given data type does not exist!")

    def get_kfold_latent_space(self, datat) -> np.ndarray:
        """
        Get the kfolded latent space for validation or testing data.
        @datat :: String of the data type.

        returns :: The kfolded output of the ae depending on the data.
        """
        if datat == 'valid':
            return self.model.predict(self.ae_kfold_data.valid_data)[0]
        if datat == 'test':
            return self.model.predict(self.ae_kfold_data.test_data)[0]

        raise TypeError("Given data type does not exist!")

    def fold(self, data, target, events_per_kfold) -> np.ndarray:
        """
        Fold the data, given a number of events you want per fold.
        All data that is not folded is then discarded.
        @data   :: Numpy array of the data to be folded.
        @target :: Numpy array of the target corresponding to the data.
        @events_per_kfold :: The number of events wanted per fold.

        returns :: Folded data set with a certain number of events
            per fold.
        """
        data_sig, data_bkg = self.ae_data.split_sig_bkg(data, target)
        data_sig = data_sig.reshape(-1, int(events_per_kfold/2),
                                    data_sig.shape[1])
        data_bkg = data_bkg.reshape(-1, int(events_per_kfold/2),
                                    data_bkg.shape[1])

        return np.concatenate((data_sig, data_bkg), axis=1)

    def get_kfolded_data(self, datat) -> np.ndarray:
        """
        Get the kfolded data for either the validation or testing data.
        @datat :: String of the data type.

        returns :: Folded data set with a certain number of events
            pre fold.
        """
        if datat == 'valid':
            return self.fold(self.get_kfold_latent_space(datat),
                             self.ae_kfold_data.valid_target, self.nvalid)
        if datat == 'test':
            return self.fold(self.get_kfold_latent_space(datat),
                             self.ae_kfold_data.test_target, self.ntest)

        raise TypeError("Given data type does not exist!")
