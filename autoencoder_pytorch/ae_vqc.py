# Just like the classifier ae, but with a vqc attached that does the
# classification. 

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from pennylane.templates import AmplitudeEmbedding
from pennylane.templates import AngleEmbedding

import matplotlib.pyplot as plt


from ae_vanilla import AE_classifier
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
        new_hp = {
            "ae_type"     : "classvqc",
            "nqbits"      : 4,
            "vqc_specs"   : qcirc_specs,
            "measurement" : "first",
        }
        self.hp.update(new_hp)
        self.hp.update((k, hparams[k]) for k in self.hp.keys() & hparams.keys())

        self.nparams = 0 # To be filled in construct_vqc method.
        self.qdevice = qml.device("default.qubit", wires=self.hp["nqubits"])
        self.vqc_cl  = self.construct_vqc
        self.wshape  = {"theta": self.nparams}

        self.classifier = qml.qnn.TorchLayer(self.vqc_cl, wshape, output_dim=1)

    @qml.qnode(self.qdevice, interface="torch")
    def construct_vqc(self, inputs, theta):

        state_0 = [[1], [0]]
        state_all = [[1]],
        y = state_0 * np.conj(state_0).T

        for idx in range(1, len(self.hp["vqc_specs"])):
            self.nparams += self.add_vqc_layer(self.hp["vqc_specs"][idx],
                self.nqubits, inputs, theta)

        get_meas_type = {
            "first": lambda: self.meas_first_qbit(y),
            "all3":  lambda: self.meas_all_3_qbits(y),
            "all4":  lambda: self.meas_all_4_qbits(y),
        }
        measurement = get_meas_type.get(self.hp["measurement"], lambda: None)()
        if measurement is None: raise TypeError("Undefined measurement!")

        return measurement

    @staticmethod
    def add_vqc_layer(spec, nqubits, inputs, theta):
        vform_name    = spec[0]
        nfrom         = int(spec[1])
        nto           = int(spec[2])
        layer_nparams = nto - nfrom

        implement_vqc_form = {
            "zzfm"   : lambda: vqc_forms.zzfm(nqubits, inputs[nfrom:nto])
            "zzfm2"  : lambda: vqc_forms.zzfm(nqubits, inputs[nfrom:nto],
                        scaled = True)
            "angle"  : lambda: AngleEmbedding(features=inputs[nfrom:nto],
                        wires = range(nqubits))
            "angle2" : lambda: AngleEmbedding(features=np.pi*inputs[nfrom:nto],
                        wires = range(nqubits))
            "amp"    : lambda: AmplitudeEmbedding(features=inputs[nfrom:nto],
                        wires = range(nqubits), normalize = True)
            "2local" : lambda: vqc_forms.twolocal(nqubits, theta[nfrom:nto],
                        reps=int(spec[3]), entanglement=spec[4])
            "tree"   : lambda: vqc_forms.treevf(nqubits, theta[nfrom:nto],
                        reps=int(spec[3]))
            "step"   : lambda: vqc_forms.stepc(nqubits, theta[nfrom:nto],
                        reps=int(spec[3]))
        }
        if vform_name in implement_vqc_layer.keys():
            implement_vqc_layer.get(vform_name)()
        else: raise TypeError("Undefined VQC Template!")

        return layer_nparams

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
