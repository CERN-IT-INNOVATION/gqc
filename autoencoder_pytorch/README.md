TODO: Maybe softmax as output activation? To automatize the sum to 1 of amplitudes of the Qcircuit
TODO: define train method here, instead of separate function in train.py
TODO: Load numpy arrays as data


TIPS for PyTorch (1.7.0):

- When a tensor is created it is automatically loaded on cpu. If gpu is desired to(device) needs to be cast on every tensor.
- cpu->gpu loading requires copying the tensor elements which can take time the benefit of using gpu comes after loading during
  matrix/tensor parallelized manipulations
- A custom model class inheriting from nn.Module is "a list" of tensors (called also parameters of the model) tha are trainable, i.e. gradients
  are computed by pytorch. If one defines a custom layer/tensor which should be a trainable part of the model on should call a nn.Parameter
  wrapper to make it part of the parameter list of the model.
- When calling model.to(device) all the parameters of the model ara loaded to the device (if one does not call nn.Parameter on the custom tensors
  they will not be loaded to the device, potentially causing problems)
Example:
self.Min = nn.Parameter(torch.Tensor([Min]), requires_grad=False)
model = Model()
model.Min
print(model) -> Whatever is printed is part of the model parameters (nn.Parameter object)
