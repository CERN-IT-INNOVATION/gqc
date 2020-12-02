import numpy as np
import matplotlib.pyplot as plt
import os
from varname import *
#Unormalised vars:
bkg = np.load('/data/vabelis/disk/sample_preprocessing/input_ae/raw_bkg.npy')
sig = np.load('/data/vabelis/disk/sample_preprocessing/input_ae/raw_sig.npy')
outdir = 'pdfsIn'
if not(os.path.exists(outdir)):
	os.mkdir(outdir)
for i in range(bkg.shape[1]):
    hSig,_,_ = plt.hist(x=sig[:,i],density=1,bins=60,alpha=0.6,histtype='step',linewidth=2.5,label='Sig')
    hBkg,_,_ = plt.hist(x=bkg[:,i],density=1,bins=60,alpha=0.6,histtype='step',linewidth=2.5,label='Bkg')
    plt.xlabel(f'feature {i}')
    plt.ylabel('Entries/Bin')
    plt.legend()
    plt.savefig(outdir+'/plot'+varname(i)+'.png')
    plt.clf()
