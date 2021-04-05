# Data Pre-processing:

Here we have all the relevant scripts needed for the pre-processing of the raw data. With these modules we achieve the pipeline that takes .root ntuples (TTree flattened files) and at the end outputs .npy arrays of the data (sig & bkg) that are used for the Auto-encoder and classifier model training. 

In this pipeline we apply event and object selection criteria and at the end we normalize (min-max or mean-sigma) the samples. (For now the Autoencoder is able to learn only with min-max method)

The flow is as follows:

1. Raw data from ROOT trees (ntuples) .root -> .h5 file (format.py)
2. .h5 file -> .npy with event selection (data\_prep.py) #TODO: specify more
3. Extra event selection, concatination of arrays and reducing the number of samples/events (e.g. n\_btag>=2) (reshaper\_reducer.py) #TODO: specify more
4. Normalize the dataset for the Autoencoder (normalize.py) #TODO: specify more