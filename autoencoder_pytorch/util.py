# Utility methods for dealing with all the different autoencoder things.
import torch
import numpy as np

class tensor_data(torch.utils.data.Dataset):
    # I do not know how to define this exactly.
    def __init__(self,x):
        # x is the numpy array dataset #TODO float casting
        # self.x_labels = x_labels No labels here, we want autoencoder
        self.x = torch.Tensor(x)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index]


def to_pytorch_data(data, batch_size=None, shuffle):
    """
    Convert training and validation data into pytorch ready data objects.

    @data       :: The data we want to convert.
    @batch_size :: The batch sizeto import the data with.
    @shuffle    :: Bool of wheter to shuffle the data or not.

    @returns :: Pytorch ready data object.
    """
    if batch_size is None: batch_size = data.shape[0]
    data = tensor_data(data)
    pytorch_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)

    return pytorch_loader

def split_sig_bkg(data, target):
    # Split dataset into signal and background samples using the target data.
    # The target is supposed to be 1 for every signal and 0 for every bkg.
    sig_mask   = (target == 1); bkg_mask = (target == 0)
    zipped_sig = zip(target, sig_mask)
    zipped_bkg = zip(target, bkg_mask)
    data_sig = [[num for num, b in zip(lst, mask) if b]
        for lst, mask in zipped_sig]
    data_bkg = [[num for num, b in zip(lst, mask) if b]
        for lst, mask in zipped_bkg]

    return data_sig, data_bkg

def load_model(model_module, layers, model_path):
    """
    Loads a model that was trained previously.

    @model_module :: The class of the model imported a the top.
    @layers       :: The layer structure of the model.
    @model_path   :: The path to where the trained model was saved.

    @returns :: The pytorch model object as trained in the training file.
    """
    model = model_module(node_number=layers).to(device)
    model.load_state_dict(torch.load(model_path + 'best_model.pt',
        map_location=torch.device('cpu')))
    model.eval()

    return model

@torch.no_grad()
def compute_model(model, data_loader):
    """
    Computes the output of an autoencoder trained model.

    @model       :: The name of the model we are using.
    @data_loader :: Pytorch data loader object.

    @returns :: The model output, the latent space output, and the input data.
    """
    data_iter = iter(data_loader)
    input_data = data_iter.next()
    model_output, latent_output = model(input_data.float())

    return model_output, latent_output, input_data

def mean_batch_loss(data_loader, feature_size, device):
    # Computes the mean batch loss for a sample and model, for each batch.
    mean_loss_batch = []
    for idx, batch_features in enumerate(data_loader):
        batch_features = batch_features.view(-1, feature_size).to(device)
        output,_ = model(batch_features)
        loss = criterion(output, batch_features)
        mean_loss_batch.append(loss.item())

    return mean_loss_batch

def prepare_output(model_nodes, batch_size, learning_rate):
    # Prepare the naming of training outputs. Do not print output size.
    layersTag = '.'.join(str(inode) for inode in model_nodes[1:])
    filetag = 'L' + layersTag + '_B' + str(batch_size) + '_Lr{:.0e}'.\
        format(learning_rate) + "_" + fileFlag

    outdir = './trained_models/' + filetag + '/'
    if not(os.path.exists(outdir)): os.mkdir(outdir)

    return filetag, outdir

def save_MSE_log(filetag, train_time, min_valid):
    # Save MSE log to check best model:
    with open('trained_models/mse_log.txt','a+') as mse_log:
        log_entry = filetag + f': Training time = {train_time:.2f} min, Min. Validation loss = {min_valid:.6f}\n'
        mseLog.write(logEntry)
