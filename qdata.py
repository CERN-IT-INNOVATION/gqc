# Loads the data that is subsequently fed to the quantum ml networks.
import numpy as np
import ae_pytorch as ae
from ae_pytorch.terminal_colors import tcols

class qdata:
    def __init__(self, data_folder, norm_name, nevents, model_path,
        train_events=-1, valid_events=-1, test_events=-1):

        hp           = ae.util.import_hyperparams(model_path)
        self.ae_data = ae.data.AE_data(data_folder, norm_name, nevents,
            train_events, valid_events, test_events)
        self.model   = ae.util.choose_ae_model(hp['ae_type'], device, hp)
        self.model.load_model(model_path)

        self.ntrain = self.ae_data.train_data.shape[0]
        self.nvalid = self.ae_data.valid_data.shape[0]
        self.ntest  = self.ae_data.test_data.shape[0]

    def get_latent_space(self, datat):
        if datat == 'train': return self.model.predict(self.ae_data.train_data)
        if datat == 'valid': return self.model.predict(self.ae_data.valid_data)
        if datat == 'test':  return self.model.predict(self.ae_data.test_data)

        raise TypeError("Given data type does not exist!")

    def fold(self, data, target, kfolds, events_per_kfold):
        data_sig, data_bkg = self.ae_data.split_sig_bkg(data, target)
        data_sig = data_sig.reshape(-1, self.ae_data.nfeats, events_per_kfold)
        data_bkg = data_bkg.reshape(-1, self.ae_data.nfeats, events_per_kfold)

        return np.concatenate((data_sig[:, :, :kfolds],
            data_bkg[:, :, :kfolds]), axis=1)

    def get_kfolded_data(self, datat, kfolds, nevt):
        if datat == 'valid':
            return self.fold(self.valid_data, self.valid_target, kfolds, nevt)
        if datat == 'test':
            return self.fold(self.test_data, self.test_target, kfolds, nevt)

        raise TypeError("Given data type does not exist!")
