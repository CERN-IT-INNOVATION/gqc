from vqctf.train import *

ntrain = 15
epochs = 1
learning_rate = 0.005
batch_size = 50


spec = [4, ["zzfm", 0, 4], ["2local", 0, 20, 4], ["zzfm", 4, 8], ["2local", 20, 40, 4]]
encoder = [1, 4, 5, 3, 7, 0, 11, 45]

name = __file__[:-3]
model = train(epochs, learning_rate, batch_size, spec, ntrain, encoder, name)





