# ML classifiers, to act on the latent space of the autoencoder and produce
# a loss, by calculating the binary cross-entropy. This can then be used
# to optimize for separation between background and signal in the latent
# space of the autoencoder.

import torch
from torch import nn, optim
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier


class SVM():
    # A Standard Support Vector Machine implementation.
    def __init__(self):
        self.clf = SGDClassifier(loss="log", alpha=10.0, verbose=1, warm_start=True)

    def fit(self, x_batch, y_batch):

        self.clf.partial_fit(x, y, classes=np.unique(y))
        print("\033[92mBatch processed in SVM.\033[0m")

    def predict(self, x):

        y_pred = self.clf.predict_proba(x)
        return y_pred[:,1]


class BDT():
    # A Standard Boosted Decision Tree implementation.
    def __init__(self):
        self.clf = GradientBoostingClassifier(n_estimators=1, verbose=1,
            max_depth=2, warm_start=True)

    def fit(self, x, y):

        self.clf.fit(x, y)
        self.clf.n_estimators += 1
        print("\033[92mBatch processed in BDT.\033[0m")

    def predict(self,x):

        y_pred = self.clf.predict_proba(x)
        return y_pred[:,1]


class FFWD(nn.Module):
    # FFWD DNN implementation.
    def __init__(self, device, layers, loss_func=nn.BCELoss(), lr=2e-3,
        lr_decay=0.04, epochs=85, batch_size=128):

        super(FFWD, self).__init__()

        self.device         = device
        self.loss_func      = loss_func
        self.lr             = lr
        self.lr_decay       = lr_decay
        self.epochs         = epochs
        self.batch_size     = batch_size

        dnn_layers = self.construct_dnn(layers)
        self.ffwd  = nn.Sequential(*dnn_layers)

    @staticmethod
    def construct_dnn(layers):

        dnn_layers = []
        dnn_layers.append(nn.BatchNorm1d(layers[0]))

        for idx in range(len(layers)):
            dnn_layers.append(nn.Linear(layers[idx], layers[idx+1]))
            if idx == len(layers) - 2: dnn_layers.append(nn.Sigmoid()); break

            dnn_layers.append(nn.BatchNorm1d(layers[idx+1]))
            dnn_layers.append(nn.Dropout(0.5))
            dnn_layers.append(nn.LeakyReLU(0.2))

        return dnn_layers

    def forward(self, x):
        x = self.ffwd(x)
        return x

    def train(self, optimizer, data):

        self.ffwd.train()
        for epoch in range(1, self.epochs + 1):
            loss_epoch = 0
            for data_batch in data:
                x, y = data_batch
                x = x.to(self.device); y = y.to(self.device)

                optimizer.zero_grad()
                dnn_output = self.ffwd(x)
                loss_value = self.loss_func(dnn_output, y)
                loss_epoch += loss_value.item()

                loss_value.backward()
                optimizer.step()
                print("\033[92mBatch processed in DNN.\033[0m")

        return loss_epoch/len(data)

    def fit(self, x_train, y_train):

        tensor_x    = torch.Tensor(x_train)
        tensor_y    = torch.Tensor(y_train)
        dataset     = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=self.batch_size, shuffle=True)

        ffwd      = self.ffwd.to(self.device)
        optimizer = optim.Adagrad(ffwd.parameters(), lr=self.lr,
            lr_decay=self.lr_decay)

        print("\033[92mTraining the DNN...\033[0m")
        loss = self.train(optimizer, data_loader)

        return loss

    @torch.no_grad()
    def predict(self, x_test):
        x_test = torch.from_numpy(x_test).to(self.device)
        pred = self.ffwd(x_test.float())
        pred = pred.cpu().numpy()

        return pred
