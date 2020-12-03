import numpy as np

bkg = np.load("trainingTestingDataBkg.npy")
sig = np.load("trainingTestingDataSig.npy")

nval = int(min(bkg.shape[0], sig.shape[0]))
limval1 = int(np.floor(nval * 10 / 12))
limval2 = int(np.floor(nval * 11 / 12))

train_bkg = bkg[0:limval1]
train_sig = sig[0:limval1]

vali_bkg = bkg[limval1:limval2]
vali_sig = sig[limval1:limval2]

test_bkg = bkg[limval2:nval]
test_sig = sig[limval2:nval]

train = np.vstack((train_bkg, train_sig))
vali = np.vstack((vali_bkg, vali_sig))
test = np.vstack((test_bkg, test_sig))


