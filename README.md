[![Email: patrick](https://img.shields.io/badge/email-podagiu%40student.ethz.ch-blue?style=flat-square&logo=minutemailer)](mailto:podagiu@student.ethz.ch)
[![Email: vasilis](https://img.shields.io/badge/email-vasileios.belis%40cern.ch-blue?style=flat-square&logo=minutemailer)](mailto:vasileios.belis@cern.ch)
[![Python: version](https://img.shields.io/badge/python-3.9%20%7C%203.10-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![License: version](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](https://github.com/QML-HEP/ae_qml/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)

# AE QML

### Using autoencoders to train quantum machine learning models for Higgs boson identification.


## What is it?

Near-term Intermediate Size Quantum (NISQ) devices can only provide a limited
number of bits for Quantum Machine Learning (QML) algorithms to run on. Feature
reduction methods need to be applied to larger datasets if they are to be
processed by QML algorithms. For example, the Quantum Support Vector
Machine (QSVM) and the Variational Quantum Circuit (VQC) implemented in this
work take an input that has a maximum dimensionality of 16. However, the
studied data set has a dimensionality of 67. Thus, different types of
autoencoders are implemented to reduce the dimensionality of our
High Energy Physics (HEP) data set (ttHbb). The latent spaces of these
autoencoders are then given to the QSVM and VQC to train on and their
performance is benchmarked.


## Installing Dependencies

We strongly recommend using `conda` to install the dependencies for this repo.
If you have conda, creating a virtual environment with all the packages
used in our code is done easily, by running the command:

```
conda env create -f ae_qml_conda.yml
```

Afterwards, you can activate the environment with

```
conda activate ae_qml
```

Once inside the environment, you should be able to run the code smoothly.

If you do not want to use `conda`, here is a list of the packages you
would need to install:

* numpy
* pandas
* tables
* matplotlib
* scikit-learn
* torch
* torchvision
* torchaudio
* torchinfo
* cudatoolkit
* cudatoolkit-dev (adds the nvcc compiler to cudatoolkit)
* pykeops
* geomloss
* qiskit
* qiskit-machine-learning

The pykeops package is required to run the Sinkhorn auto-encoder. However,
it is a tricky package to manage, so make sure that you have a gcc and a g++
compiler in your path that is compatible with the version of cuda you are
running. We recommend using conda for exactly this reason, since conda
sets certain environment variables such that everything is configured correctly
and pykeops can compile using cuda.

If you encounter any bugs, please contact us at the email addresses listed
on this repository.

## Running the Code

The data preprocessing scripts are ran from inside the preprocessing folder.
These scripts were customised for the specific data set that the authors are
using. For access to this data, please contact us.

The preprocessing scripts produce normalised numpy arrays saved to three
different files for training, validation, and testing.

The scripts to launch the autoencoder training on the data are in the bin
folder. *Look for the `run.snip` files to see the basic run cases for the*
*code and customise from there*.
