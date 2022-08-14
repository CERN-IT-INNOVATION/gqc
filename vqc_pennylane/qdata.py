# Loads the data and an autoencoder model. The original data is passed
# through the AE and the latent space is fed to the qsvm network.
import sys
import os
import numpy as np

sys.path.append("..")
from .terminal_colors import tcols
from autoencoders import data as aedata
from autoencoders import util as aeutil


class qdata:
    """
    Data loader class. qdata is used to load the train/validation/test datasets
    for the quantum ML model training given a pre-trained Auto-Encoder model
    that reduces the number of features of the initial dataset.
    """

    def __init__(
        self,
        data_folder,
        norm_name,
        nevents,
        model_path,
        train_events=0,
        valid_events=0,
        test_events=0,
        kfolds=0,
        seed=None,
    ):
        """
        Args:
            data_folder (str): Path to the input data of the Auto-Encoder.
            norm_name (str): Specify the normalisation of the input data
                             e.g., minmax, maxabs etc.
            nevents (float): Number of signal data samples in the input data file.
                             Conventionally, we encode this number in the file
                             name, e.g., nevents = 7.20e+05.
            model_path (str): Path to the save PyTorch Auto-Encoder model.
            train_events (int): Number of desired train events to be loaded by
                                qdata.
            valid_events (int): Number of desired validation events to be loaded
                                by qdata.
            test_events (int): Number of desired test events to be loaded by
                                qdata.
            kfolds (int): Number of folds (i.e. statistiaclly independent datasets)
                          to use for validation/testing of the trained QML models.
            seed (int): Seed for the shufling of the train/test/validation and
                        k-folds datasets.
        """

        device = "cpu"
        model_folder = os.path.dirname(model_path)
        hp_file = os.path.join(model_folder, "hyperparameters.json")
        hp = aeutil.import_hyperparams(hp_file)

        print(tcols.OKCYAN + "\nLoading data:" + tcols.ENDC)
        self.ae_data = aedata.AE_data(
            data_folder,
            norm_name,
            nevents,
            train_events,
            valid_events,
            test_events,
            seed,
        )
        self.model = aeutil.choose_ae_model(hp["ae_type"], device, hp)
        self.model.load_model(model_path)

        self.ntrain = self.ae_data.trdata.shape[0]
        self.nvalid = self.ae_data.vadata.shape[0]
        self.ntest = self.ae_data.tedata.shape[0]
        self.seed = seed

        if kfolds > 0:
            print(tcols.OKCYAN + "Loading k-folded valid data:" + tcols.ENDC)
            self.kfolds = kfolds
            self.ae_kfold_data = aedata.AE_data(
                data_folder,
                norm_name,
                nevents,
                0,
                kfolds * valid_events,
                kfolds * test_events,
                seed,
            )

    def get_latent_space(self, datat) -> np.ndarray:
        """
        Get the latent space depending on the data set you want.
        @datat :: String of the data type.

        returns :: Output of the ae depending on the given data type.
        """
        if datat == "train":
            return self.model.predict(self.ae_data.trdata)[0]
        if datat == "valid":
            return self.model.predict(self.ae_data.vadata)[0]
        if datat == "test":
            return self.model.predict(self.ae_data.tedata)[0]

        raise TypeError("Given data type does not exist!")

    def fold(
        self, data: np.array, target: np.array, events_per_kfold: int, latent: bool
    ):
        """
        Fold the data, given a number of events you want per fold.
        All data that is not folded is then discarded.

        For the case of kfold=n and kfold=m, we should not expect any of the
        folds to be the same between each other. That is, the input @data
        contains self.ntest samples which are then split (create the folds),
        concatenated and shuffled again. Hence, we should not expect identical
        folds between these two different cases, even for the same self.ntest.

        Args:
            data: 2D data to be folded (already shuffled once).
            target: 1D target data corresponding to the @data.
            events_per_kfold: The number of events wanted per fold.
            latent: Whether the data should be passed through an ae (True) or not.

        Returns:
            Folded data set with a certain number of events per fold.
        """
        data_sig, data_bkg = self.ae_data.split_sig_bkg(data, target)
        data_sig = data_sig.reshape(-1, int(events_per_kfold / 2), data_sig.shape[1])
        data_bkg = data_bkg.reshape(-1, int(events_per_kfold / 2), data_bkg.shape[1])
        data = np.concatenate((data_sig, data_bkg), axis=1)
        target = np.array(
            [
                np.concatenate(
                    (
                        np.ones(int(events_per_kfold / 2)),
                        np.zeros(int(events_per_kfold / 2)),
                    )
                )
                for kfold in range(self.kfolds)
            ]
        )
        shuffling = np.random.RandomState(seed=self.seed).permutation(events_per_kfold)

        data = data[:, shuffling]
        target = target[:, shuffling]
        if not latent:
            return data, target

        data = [self.model.predict(kfold)[0] for kfold in data]
        return data, target

    def get_kfolded_data(self, datat: str, latent: bool):
        """Get the kfolded data for either the validation or testing data. Choose
        whether this data should be passed through an autoencoder or not.

        Args:
            datat: The data type, i.e., either 'valid' or 'test'.
            latent: Whether the data should be passed through an ae (True) or not.

        Returns:
            Folded data set with a certain number of events per fold.
        """
        if datat == "valid":
            return self.fold(
                self.ae_kfold_data.vadata,
                self.ae_kfold_data.vatarget,
                self.nvalid,
                latent,
            )
        if datat == "test":
            return self.fold(
                self.ae_kfold_data.tedata,
                self.ae_kfold_data.tetarget,
                self.ntest,
                latent,
            )

        raise TypeError("Given data type does not exist!")

    @staticmethod
    def batchify(data, batch_size):
        """
        Reshape the training data into an array of arrays, the sub arrays
        containing the amount of events that are contained in a batch.
        @data :: Array of data to be split.
        @batch_size :: Int of the batch size.
        """
        num_splits = np.ceil(data.shape[0] / batch_size)
        return np.array_split(data, num_splits)

    @staticmethod
    def to_onehot(target):
        """
        Reshape the target that such that it follows onehot encoding.
        @target :: Numpy array with target data.
        """
        onehot_target = np.zeros((target.size, int(target.max() + 1)))
        onehot_target[np.arange(target.size), target.astype(int)] = 1

        return onehot_target
