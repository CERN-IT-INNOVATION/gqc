## Documentation

Quantum classifiers for Higgs searches:

### Data pre-processing
- Raw data from ROOT trees (ntuples) .root -> .h5 file (format.py)
- .h5 file -> .npy with event selection (data\_prep.py)
- Extra event selection, concatination of arrays and reducing the number of samples/events (e.g. n\_btag>=2) (reshaper\_reducer.py)
- Normalize the dataset for the Autoencoder (normalize.py)

