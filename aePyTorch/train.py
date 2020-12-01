import torch
#Train function:
def train(train_loader,valid_loader,model,criterion,optimizer,epochs,device,outdir):
	lossTrainValues = []
	lossValidValues = []
	minValid = 99999
	for epoch in range(epochs):
		model.train()
		for i, batch_features in enumerate(train_loader):
			# reshape mini-batch data to [batch_size, feature_size] matrix
			# load it to the active device
			feature_size = batch_features.shape[1]
			batch_features = batch_features.view(-1, feature_size).to(device)
			#compute model output:
			output,_ = model(batch_features.float())#Using both Sig+Bkg arrays gave problems 
								#expecting float getting double
			# compute training reconstruction loss
			train_loss = criterion(output, batch_features.float())
			
			#PyTorch accumulates gradients on subsequent backward passes.
			#Rest gradients, perform a backward pass, and update the weights:
			optimizer.zero_grad()
			train_loss.backward()
			optimizer.step()
	
		#Save best model:
		validDataIter = iter(valid_loader)
		validData= validDataIter.next()
		model.eval()
		output,_ = model(validData.float())
		valid_loss = criterion(output,validData).item()
		if valid_loss < minValid:
			minValid = valid_loss
			print('New minimum of validation loss:')
			torch.save(model.state_dict(), outdir+'bestModel.pt')
			
		lossValidValues.append(valid_loss)
		lossTrainValues.append(train_loss.item())#final batch loss value
		print("epoch : {}/{}, Training loss (last batch) = {:.8f}".format(epoch + 1, epochs, train_loss.item()))
		print("epoch : {}/{}, Validation loss = {:.8f}".format(epoch + 1, epochs, valid_loss))
	
	return lossTrainValues,lossValidValues,minValid
