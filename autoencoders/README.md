# Autoencoder
The autoencoder framework used for feature reduction. There are different types of autoencoder architectures that are used, each with their own file following the naming convention `ae_\[architecture name\].py`. The `train.py` file is used to pick whatever architecture the user desires, instantiates it, and then trains the architecture on the data the user provided it with. All the hyperparameters of the AE and of the training are also set in the same file.

The workflow of the auto-encoder framework is depicted in the following diagram.

![Autoencoder Workflow](ae_workflow.png)

For more details, please see the code and comments therein.

