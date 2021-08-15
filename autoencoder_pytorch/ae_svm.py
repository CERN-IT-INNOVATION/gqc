# SVM autencoder. Different from the vanilla one since it has a
# SVM attached to the latent space, that does the classification
# for each batch latent space and outputs the binary cross-entropy loss
# that is then used to optimize the autoencoder as a whole.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .ae_classifier import AE_classifier
from .terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE_svm(AE_classifier):
    def __init__(self, device, layers, lr, en_activ, dec_activ, loss_weight):

        super(AE_classifier, self).__init__(device, layers, lr,
            en_activ, dec_activ)

        self.class_loss_function = self.hinge_loss
        self.loss_weight         = loss_weight

        self.class_layers = [layers[-1] ,1]
        self.classifier   = nn.Sequential(nn.Linear(*self.class_layers))
        self = self.to(device)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    @staticmethod
    def hinge_loss(output, y):
        return torch.mean(torch.clamp(1 - output * y, min=0))

    def train_autoencoder(self, train_loader, valid_loader, epochs, outdir):

        self.network_summary(); self.optimizer_summary()
        print(tcols.OKCYAN + "Training the SVM AE model..." + tcols.ENDC)
        all_train_loss = []; all_valid_loss = []

        for epoch in range(epochs):
            self.train()

            train_loss   = self.train_all_batches(train_loader)
            valid_losses = self.valid(valid_loader,outdir)

            if self.early_stopping():
                return all_train_loss, all_valid_loss, self.best_valid_loss

            all_train_loss.append(train_loss.item())
            all_valid_loss.append(valid_losses[0].item())

            self.print_losses(epoch, epochs, train_loss, valid_losses)

        return all_train_loss, all_valid_loss, self.best_valid_loss

