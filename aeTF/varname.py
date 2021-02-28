def varname(index):
	jet_feats = ["$p_t$","eta","phi","en","$p_x$","$p_y$","$p_z$","btag"]
	jet_nvars = len(jet_feats);
	num_jets = 7;
	
	met_feats = ["phi","$p_t$","$p_x$","$p_y$"]
	met_nvars = len(met_feats);
	
	lep_feats = ["$p_t$","eta","phi","en","$p_x$","$p_y$","$p_z$"]
	lep_nvars = len(lep_feats);
	

	if (index < jet_nvars * num_jets):
		jet = index // jet_nvars + 1;
		var = index % jet_nvars;
		varstring = "Jet " + str(jet) + " " + jet_feats[var]
		return varstring;

	index -= jet_nvars * num_jets;
	
	if (index < met_nvars):
		var = index % met_nvars;
		varstring = "MET " + met_feats[var];
		return varstring;
	
	index -= met_nvars;

	if (index < lep_nvars):
		var = index % lep_nvars;
		varstring = "Lepton " + lep_feats[var];
		return varstring;


	return None;
		


