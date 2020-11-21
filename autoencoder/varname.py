def varname(index):
	jet_feats = ["pt","eta","phi","en","px","py","pz","btag"]
	jet_nvars = len(jet_feats);
	num_jets = 10;
	
	met_feats = ["phi","pt","px","py"]
	met_nvars = len(met_feats);
	
	lep_feats = ["pt","eta","phi","en","px","py","pz"]
	lep_nvars = len(lep_feats);
	

	if (index < jet_nvars * num_jets):
		jet = index // jet_nvars + 1;
		var = index % jet_nvars;
		varstring = "Jet " + str(jet) + " " + jet_feats[var]
		return varstring;

	index -= jet_nvars * num_jets;
	
	if (index < met_nvars):
		var = index % met_nvars;
		varstring = "Met " + met_feats[var];
		return varstring;
	
	index -= met_nvars;

	if (index < lep_nvars):
		var = index % lep_nvars;
		varstring = "Lep " + lep_feats[var];
		return varstring;


	return None;
		


