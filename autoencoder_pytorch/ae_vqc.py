# Just like the classifier ae, but with a vqc attached that does the
# classification.

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.templates import AmplitudeEmbedding
from pennylane.templates import AngleEmbedding

import matplotlib.pyplot as plt


from ae_classifier import AE_classifier
import vqc_forms
from terminal_colors import tcols

seed = 100
torch.manual_seed(seed)

# Diagnosis tools. Enable in case you need.
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class AE_vqc(AE_classifier):
    def __init__(self, device = 'cpu', hparams = {}):

        super().__init__(device, hparams)
        del self.class_layers; del self.classifier
        self.qcirc_default_specs = \
            [["zzfm", 0, 4], ["2local", 0, 20, 4, "linear"],
             ["zzfm", 4, 8], ["2local", 20, 40, 4, "linear"]]

        new_hp = {
            "ae_type"     : "classvqc",
            "nqbits"      : 4,
            "vqc_specs"   : self.qcirc_default_specs,
            "measurement" : "first",
        }

        self.hp.update(new_hp)
        self.hp.update((k, hparams[k]) for k in self.hp.keys() & hparams.keys())
        self.hp.pop("class_layers")

        self.nparams = self.getnparams(self.hp['vqc_specs'])
        self.qdevice = qml.device("default.qubit", wires=self.hp["nqbits"])
        self.vqc_circuit = \
            qml.qnode(self.qdevice, interface="torch")(self.construct_vqc)
        self.wshape  = {"theta": self.nparams}

        self.classifier = qml.qnn.TorchLayer(self.vqc_circuit, self.wshape)

    @staticmethod
    def getnparams(specs):
        nparams = 0
        for spec in specs:
            if spec[0] == '2local' or spec[0] == 'tree' or spec[0] == 'step':
                nparams += spec[2] - spec[1]

        return nparams

    def construct_vqc(self, inputs, theta):

        state_0 = [[1], [0]]
        state_all = [[1]],
        y = state_0 * np.conj(state_0).T

        for idx in range(0, len(self.hp["vqc_specs"])):
            self.add_vqc_layer(self.hp["vqc_specs"][idx], self.hp['nqbits'],
                inputs, theta)

        get_meas_type = {
            "first": lambda: self.meas_first_qbit(y),
            "all3":  lambda: self.meas_all_3_qbits(y),
            "all4":  lambda: self.meas_all_4_qbits(y),
        }
        measurement = get_meas_type.get(self.hp["measurement"], lambda: None)()
        if measurement is None: raise TypeError("Undefined measurement!")

        return measurement

    @staticmethod
    def add_vqc_layer(spec, nqbits, inputs, theta):
        vform_name    = spec[0]
        nfrom         = int(spec[1])
        nto           = int(spec[2])
        layer_nparams = nto - nfrom

        implement_vqc_layer = {
            "zzfm"   : lambda: vqc_forms.zzfm(nqbits, inputs[nfrom:nto]),
            "zzfm2"  : lambda: vqc_forms.zzfm(nqbits, inputs[nfrom:nto],
                        scaled=True),
            "angle"  : lambda: AngleEmbedding(features=inputs[nfrom:nto],
                        wires=range(nqbits)),
            "angle2" : lambda: AngleEmbedding(features=np.pi*inputs[nfrom:nto],
                        wires=range(nqbits)),
            "amp"    : lambda: AmplitudeEmbedding(features=inputs[nfrom:nto],
                        wires=range(nqbits), normalize=True),
            "2local" : lambda: vqc_forms.twolocal(nqbits, theta[nfrom:nto],
                        reps=int(spec[3]), entanglement=spec[4]),
            "tree"   : lambda: vqc_forms.treevf(nqbits, theta[nfrom:nto],
                        reps=int(spec[3])),
            "step"   : lambda: vqc_forms.stepc(nqbits, theta[nfrom:nto],
                        reps=int(spec[3]))
        }
        if vform_name in implement_vqc_layer.keys():
            implement_vqc_layer.get(vform_name)()
        else: raise TypeError("Undefined VQC Template!")

    @staticmethod
    def meas_first_qbit(y):
        return qml.expval(qml.Hermitian(y, wires = [0]))

    @staticmethod
    def meas_all_4_qbits(y):
        return qml.expval(qml.Hermitian(y, wires=[0]) @
            qml.Hermitian(y, wires = [1]) @ qml.Hermitian(y,wires=[2]) @
            qml.Hermitian(y,wires = [3]))

    @staticmethod
    def meas_all_3_qbits(y):
        return qml.expval(qml.Hermitian(y, wires=[0]) @
            qml.Hermitian(y, wires = [1]) @ qml.Hermitian(y, wires=[2]))


    @torch.no_grad()
    def predict(self, x_data):
        # Compute the prediction of the autoencoder, given input np array x.
        x_data = torch.from_numpy(x_data).to(self.device)
        self.eval()
        latent, classif, recon = self.forward(x_data.float())

        latent  = latent.cpu().numpy()
        classif = classif.cpu().numpy().reshape((-1,1))
        recon   = recon.cpu().numpy()
        return latent, recon, classif
