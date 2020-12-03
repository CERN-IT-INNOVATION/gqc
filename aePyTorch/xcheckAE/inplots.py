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
	fig, ax = plt.subplots()
	pdfSig,pdfBkg = sig[:,i], bkg[:,i]
	ax.hist(x=pdfSig,density=1,bins=60,alpha=0.6,histtype='step',linewidth=2.5,label='Sig')
	ax.hist(x=pdfBkg,density=1,bins=60,alpha=0.6,color ='r',histtype='step',linewidth=2.5,label='Bkg')
	plt.xlabel('feature '+varname(i))
	plt.ylabel('Entries/Bin')
	plt.legend()

	#box with stats:
	#Not positioned properly, let it be for now:
	#box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
	#textstr = '\n'.join((f'Sig. mean={np.mean(pdfSig):.2f}',f'Sig. std={np.std(pdfSig):.2f}',f'Bkg. mean={np.mean(pdfBkg):.2f}',f'Bkg. std={np.std(pdfBkg):.2f}' ))	
	#ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,verticalalignment='top', bbox=box)
	plt.savefig(outdir+'/plot'+varname(i)+'.png')
	plt.close(fig)
