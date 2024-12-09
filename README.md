[![Email: patrick](https://img.shields.io/badge/email-podagiu%40student.ethz.ch-blue?style=flat-square&logo=minutemailer)](mailto:podagiu@student.ethz.ch)
[![Email: vasilis](https://img.shields.io/badge/email-vasileios.belis%40cern.ch-blue?style=flat-square&logo=minutemailer)](mailto:vasileios.belis@cern.ch)
[![Python: version](https://img.shields.io/badge/python-3.8-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![License: version](https://img.shields.io/badge/license-MIT-purple?style=flat-square)](https://github.com/QML-HEP/ae_qml/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-black?style=flat-square&logo=black)](https://github.com/psf/black)

# Guided Quantum Compression for Higgs Identification

Many data sets are too complex for currently available quantum computers. Consequently, quantum machine learning applications conventionally resort to dimensionality reduction algorithms, e.g., auto-encoders, before passing data through the quantum models. 
We show that using a classical auto-encoder as an independent preprocessing step can significantly decrease the classification performance of a quantum machine learning algorithm. To ameliorate this issue, we design an architecture that unifies the preprocessing and quantum classification algorithms into a single trainable model: the guided quantum compression model. The utility of this model is demonstrated by using it to identify the Higgs boson in proton-proton collisions at the LHC, where the conventional approach proves ineffective. Conversely, the guided quantum compression model excels at solving this classification problem, achieving a good accuracy. Additionally, the model developed herein shows better performance compared to the classical benchmark when using only low-level kinematic features.

This repository represents the source code of the following paper [Guided quantum compression for high dimensional data classification](https://iopscience.iop.org/article/10.1088/2632-2153/ad5fdd)

If you plan to use or take part of the code, please cite the usage:
```
@article{Belis_2024,
   title={Guided quantum compression for high dimensional data classification},
   volume={5},
   ISSN={2632-2153},
   url={http://dx.doi.org/10.1088/2632-2153/ad5fdd},
   DOI={10.1088/2632-2153/ad5fdd},
   number={3},
   journal={Machine Learning: Science and Technology},
   publisher={IOP Publishing},
   author={Belis, Vasilis and Odagiu, Patrick and Grossi, Michele and Reiter, Florentin and Dissertori, Günther and Vallecorsa, Sofia},
   year={2024},
   month=jul, pages={035010} }
```




## Installing Dependencies

We strongly recommend using `conda` to install the dependencies for this repo.
If you have 'conda', go into the folder with the code you want to run, then create
an environment from the .yml file in that folder. Activate the environment.
Now you can run the code! Go to the *Running the code section.* for further instructions.

If you do not want to use `conda`, here is a list of the packages you
would need to install:

**Pre-processing**
* numpy
* pandas
* pytables
* matplotlib
* scikit-learn

**Auto-encoders**
* numpy
* matplotlib
* scikit-learn
* pytorch (follow instruction [here](https://pytorch.org))
* torchinfo
* pykeops
  * g++ compiler version >= 7
  * cudatoolkit version >= 10  
* geomloss

**Pennylane VQC**
* numpy
* matplotlib
* scikit-learn
* pytorch (follow instruction [here](https://pytorch.org))
* torchinfo
* pykeops
  * g++ compiler version >= 7
  * cudatoolkit version >= 10  
* geomloss
* pennylane
* pennylane-qiskit
* pennylane-lightning[gpu]
  * NVidia cuQuantum SDK 

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
