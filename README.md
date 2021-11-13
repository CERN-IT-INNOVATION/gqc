[![Email: patrick](https://img.shields.io/badge/email-podagiu%40student.ethz.ch-blue?style=flat-square&logo=minutemailer)](mailto:podagiu@student.ethz.ch)
[![Email: vasilis](https://img.shields.io/badge/email-vasileios.belis%40cern.ch-blue?style=flat-square&logo=minutemailer)](mailto:vasilis.belis@cern.ch)
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

This code base has a relatively large number of dependencies due to its
complexity and variety of methods.

There are two alternatives for installing the dependencies necessary to run
this code: either through pipenv or through running `setup.py`. We recommend
using pipenv.


#### Pipenv

Install pipenv using pip. Make sure that the python version you are using
is python 3.9.

```
pip install pipenv
```

Then you should go to the ae_qml directory and run

```
pipenv sync
pipenv shell
```
The first line will create a virtual environment where all the required
packages are installed (with the correct versions) and the second line will
activate this environment.

Now you should be ready to run the code!

---

#### Setup file

*Warning:* This will install all the dependencies in the current environment
that you are in. Use with caution, i.e., if you do not create a virtual env
this will be installed directly inside the standard python libs.

To install using the setup.py file, just run

```
python setup.py install
```

Now you should be ready to run the code!

## Running the Code

The data preprocessing scripts are ran from inside the preprocessing folder.
These scripts were customised for the specific data set that the authors are
using. For access to this data, please contact us.

The preprocessing scripts produce normalised numpy arrays saved to three
different files for training, validation, and testing.

The scripts to launch the autoencoder training on the data are in the bin
folder. *Look for the `run.snip` files to see the basic run cases for the*
*code and customise from there*.
