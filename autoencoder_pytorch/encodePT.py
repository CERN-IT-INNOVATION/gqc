# Legacy code that is to be used in the classfiers module. Make sure to
# move it when performing a full cleanup of the code.
import numpy as np
import torch
from autoencoder_pytorch.model_vasilis import AE
from autoencoder_pytorch.util import


def encode_array(data, saved_model, layers):
    """
    Not sure what the details of this method are. Talk to Vasilis.
    """
    data_loader = torch.utils.data.DataLoader(tensorData(data),
        batch_size=data.shape[0], shuffle = False)
    device ='cpu'
    model = AE(node_number = layers).to(device)
    model.load_state_dict(torch.load(saved_model + 'best_model.pt',
        map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        data_iter = iter(data_loader)
        input_data = data_iter.next().to(device)
        _, output = model(input_data.float())

        return output.cpu().numpy()

def encode(data, saved_model, layers):
    """
    Again, not sure what it does. Have not seen it used anywhere.
    """
    if (isinstance(data, dict)):
        for x in data:
            x = torch.Tensor(x)
            data[x] = encode_array(data[x], saved_model, layers)
        return data

    return encode_array(torch.Tensor(data),saved_model,layers)
