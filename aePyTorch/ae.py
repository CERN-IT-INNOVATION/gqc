# The autoencoder code. It reduces the number of features using in training the
# actuall ML algorithm from 67 (for the ttH dataset) to a smaller number.
import matplotlib.pyplot as plt
import numpy as np
import os,sys,torch,torchvision,time,argparse,warnings
import torch.nn as nn
import torch.optim as optim
from model import tensorData,AE
from train import train
from splitDatasets import splitDatasets

start_time = time.time()
seed = 100; torch.manual_seed(seed)
torch.autograd.set_detect_anomaly(True)

# Use GPUs if they are available.
with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

defaultlayers = [64, 52, 44, 32, 24, 16]

parser = argparse.ArgumentParser(formatter_class=argparse.
    ArgumentDefaultsHelpFormatter)
parser.add_argument("--input", type=str, required=True, nargs=2,
    help="Path to the datasets that we use.")
parser.add_argument('--lr', type=float, default=2e-03,
    help="The learning rate of the autoencoder.")
parser.add_argument('--layers', type=int, default=defaultlayers, nargs='+',
    help="The layers of the autoencoder.")
parser.add_argument('--batch', type=int, default=64,
    help="The batch size of the data to be processed by the autoencoder.")
parser.add_argument('--epochs',type=int,default=85,
    help="The number of training epochs.")
parser.add_argument('--fileFlag',type=str,default='',
    help='The fileFlag to concatenate to filetag.')
args = parser.parse_args()

def main():
    infiles = args.input

#Sig+Bkg training with shuffle
dataset, validDataset,_ = splitDatasets(infiles)

feature_size = dataset.shape[1]
layers.insert(0,feature_size)#insert at the beginning of the list the input dim.
validation_size = validDataset.shape[0]

#Convert to torch dataset:
dataset = tensorData(dataset)
validDataset = tensorData(validDataset)

train_loader = torch.utils.data.DataLoader(dataset,batch_size = args.batch,shuffle = True)
valid_loader = torch.utils.data.DataLoader(validDataset,batch_size = validation_size,shuffle = True)

model = AE(node_number = layers).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)#create an optimizer object
#betas=(0.9, 0.999) #play with the decay of the learning rate for better results

criterion = nn.MSELoss(reduction = 'mean')#mean-squared error loss

print('Batch size ='+str(batch_size)+', learning_rate='+str(learning_rate)+', layers='+str(model.node_number))

#Prepare to save training outputs:
layersTag = '.'.join(str(inode) for inode in model.node_number[1:])#Don't print input size

#FIXME: Fix lr printing. Issue example:
#> lr1,lr2 = 0.002, 0.0025
#>print(f'lr1={lr1:.0e} and lr2={lr2:.0e}')
#>'lr1=2e-03 and lr2=3e-03'
filetag = 'L'+layersTag+'B'+str(batch_size)+'Lr{:.0e}'.format(learning_rate)+fileFlag#only have 1 decimal lr

outdir = './trained_models/'+filetag+'/'
if not(os.path.exists(outdir)):
	os.mkdir(outdir)

#Print model architecture in output file:
with open(outdir+'modelArchitecture.txt', 'w') as f:
	original_stdout = sys.stdout
	sys.stdout = f # Change the standard output to the file we created.
	print(model)
	sys.stdout = original_stdout # Reset the standard output to its original value

#Call training function:
lossTrainValues,lossValidValues,minValid = train(train_loader,valid_loader,model,criterion,optimizer,epochs,device,outdir)

#Loss -vs- epochs plot
plt.plot(list(range(epochs)),lossTrainValues,label='Training Loss (last batch)')
plt.plot(list(range(epochs)),lossValidValues,label='Validation Loss (1 per epoch)')
plt.ylabel("MSE")
plt.xlabel("epochs")
plt.title("B ="+str(batch_size)+", lr="+str(learning_rate)+", "+str(model.node_number)+', L ={:.6f}'.format(minValid))
plt.legend()

#Save loss plot:
plt.savefig(outdir+'lossVSepochs.png')

end_time = time.time()
train_time = (end_time-start_time)/60
#Save MSE log to check best model:
with open('trained_models/mseLog.txt','a+') as mseLog:
	logEntry = filetag+f': Training time = {train_time:.2f} min, Min. Validation loss = {minValid:.6f}\n'
	mseLog.write(logEntry)

if __name__ == "__main__":
    main()
