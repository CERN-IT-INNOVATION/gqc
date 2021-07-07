from vqctf.train import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ntrain = 1000
epochs = 80
learning_rate = 0.005
batch_size = 50


spec = [4,
		["elu", 67],
		["elu", 32],
		["elu", 16],
		["elu", 8],
        ["zzfm", 0, 4],
        ["2local", 0, 20, 4, "linear"],
        ["zzfm", 4, 8],
        ["2local", 20, 40, 4, "linear"]
]

encoder = list(range(67)) 

name = __file__[:-3]
model = train(epochs, learning_rate, batch_size, spec, ntrain, encoder, name)
