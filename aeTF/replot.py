from compare import *
from data import *
from tensorflow.keras.models import load_model


def replot(model):
	ae = load_model("out/" + model + "/model")
	final_bkg = ae.decoder(ae.encoder(test_bkg));
	final_sig = ae.decoder(ae.encoder(test_sig));
	encoded_bkg = ae.encoder(test_bkg)
	encoded_sig = ae.encoder(test_sig)
	compare(test_bkg, final_bkg, encoded_bkg, test_sig, final_sig, encoded_sig, model, False)


#for i in np.arange(8,60):
#	replot("B-4-50-" + str(i))
#
##for i in np.arange(3,11):
#	replot("B-" + str(i) + "-50-15")

#replot("B-3-50-20")
#replot("B-5-50-20")
