from compare import *
from data import *
from tensorflow.keras.models import load_model

from sklearn import svm

model = "best"

ae = load_model("out/" + model + "/model")
#X_tr = np.array(ae.encoder(vali_bkg));
#Y_tr = np.array(ae.encoder(vali_sig));
X_tr = np.array(vali_bkg);
Y_tr = np.array(vali_sig);
train = np.vstack((X_tr, Y_tr));
labels = ['x'] * X_tr.shape[0] + ['y'] * Y_tr.shape[0]

#X = np.array(ae.encoder(test_bkg)).tolist();
#Y = np.array(ae.encoder(test_sig)).tolist();
X = np.array(test_bkg).tolist();
Y = np.array(test_sig).tolist();



cls = svm.SVC()
cls.fit(train, labels);

resX = cls.predict(X)
resY = cls.predict(Y)
print("Success ratio bkg: ", sum(resX == 'x') / (sum(resX == 'x') + sum(resX == 'y')))
print("Success ratio sig: ", sum(resY == 'y') / (sum(resY == 'x') + sum(resY == 'y')))

