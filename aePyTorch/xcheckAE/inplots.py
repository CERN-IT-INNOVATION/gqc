import numpy as np
import matplotlib.pyplot as plt
#Unormalised vars:
bkg = np.load('/data/vabelis/disk/sample_preprocessing/input_ae/raw_bkg.npy')
sig = np.load('/data/vabelis/disk/sample_preprocessing/input_ae/raw_sig.npy')

for i in range(bkg.shape[1]):
    hSig,_,_ = plt.hist(x=sig[:,i],density=1,bins=60,alpha=0.6,histtype='step',linewidth=2.5,label='Sig')
    hBkg,_,_ = plt.hist(x=bkg[:,i],density=1,bins=60,alpha=0.6,histtype='step',linewidth=2.5,label='Bkg')
    plt.xlabel(f'feature {i}')
    plt.ylabel('Entries/Bin')
    plt.legend()
    plt.savefig(f'inputPlots/plot{i}.png')
    plt.clf()

