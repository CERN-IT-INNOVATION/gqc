# Formatting the data depending on what the data is (simulation or real).
from pyjlr.utils import make_p4, make_m2
import argparse
import os
import pandas as pd
import numpy as np
import glob
import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--outdir", type=str,
    default="/scratch/{0}/jlr/".format(os.environ["USER"]), action="store",
    help="The output directory.")
parser.add_argument("--infile", type=str,
    required=True, action="store",
    help="The input folder/data file.")
parser.add_argument("--datatype", type=str,
    choices=datatype_choices.keys(), required=True,
    help="datatype choice")
parser.add_argument("--dataset", type=str,
    help="The name of dataset, determines also if ttCls is used or not.")
parser.add_argument("--target", type=str,
    required=True,
    help="The regression target: binary_classifier, multi_classifier, \
          Higgs_classifier, mbb, jlr")

args = parser.parse_args()

def main():
    # This code imports the data, formats it and stores it in h5 files.
    # An output folder with the given name in outdir will be created.
    opts = choose_data_type(args.datatype); globals().update(opts)
    data = load_files()
    data = set_data_column_headers(data)

    if selection is not None: data = data.query(selection)
    os.makedirs(args.outdir)

    trutha = None

    flats = []
    if jet_feats is not None: flats = jet_formatting(data, flats)
    if lep_feats is not None and nleps > 0: flats = lep_formatting(data, flats)
    if met_feats is not None: flats = met_formatting(data, flats)
    if jet_feats is not None and "jet_cmb" in data.columns:
        flats = jet_combinations_formatting(data, flats)
    if truth_feats is not None: flats = truth_level_formatting(data, flats)

    print('\n-----------------------------------\nMaking the target...')

    if args.target == "jlr": make_jlr_target(data)
    if args.target == "mbb": make_mbb_target(data)
    if args.target == "Higgs_classifier": make_higgsclass_target(data)
    if args.target == "multi_classifier": make_multiclass_target(data)
    if args.target == "binary_classifier": make_binarclass_target(data)

    if 'mem_tth_SL_2w2h2t_p' in data and 'mem_ttbb_SL_2w2h2t_p' in data:
        make_mem(data)


def choose_data_type(user_choice):
    """
    Determines which data type is used in the formatting.

    @user_choice :: The data type given by the user.

    @returns     :: Dictionary containing formatting specifications for the
                    particular data type that is being used.
    """

    switcher = {
        "cms_0l":        lambda : cms_0l(),
        "cms_1l":        lambda : cms_1l(),
        "cms_2l":        lambda : cms_2l(),
        "delphes_1l":    lambda : delphes_1l(),
        "delphes_2l":    lambda : delphes_2l(),
        "delphes_had":   lambda : delphes_had(),
        "mass_1l":       lambda : mass_1l(),
        "class_2016_1l": lambda : class_1l(),
        "class_2016_2l": lambda : class_2l(),
        "cms_2017_0l":   lambda : cms_2017_0l(),
        "cms_2017_1l":   lambda : cms_2017_1l(),
        "cms_2017_2l":   lambda : cms_2017_2l(),
    }
    func   = switcher.get(user_choice, lambda : "Invalid data type given!")
    choice = func()

    return choice

def opts():
    # The default set of formatting options.
    opts = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btag"],
        njets = 10,
        lep_feats = ["pt","eta","phi","en","px","py","pz"],
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        truth_feats = ["pt","eta","phi","en","px","py","pz"],
        evdesc_feats = ["nleps", "njets"],
    )

    return opts

def cms_2017_1l():
    # Options for classifier 2017 using 1 lepton cut.
    cms_2017_1l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        selection = 'nleps == 1',
        nleps = 1,
        truth_feats = None,
        evdesc_feats = ['evt','run','lumi','njets','nleps','nBDeepCSVM'],
    )
    return cms_2017_1l

def cms_2017_2l():
    # Options for classifier 2017 using 2 leptons cut.
    cms_2017_2l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        selection = 'nleps == 2',
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','nbtags'],
    )
    return cms_2017_2l

def cms_2017_0l():
    # Options for classifier 2017 using 0 leptons cut.
    cms_2017_0l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        selection = 'nleps == 0',
        nleps = 0,
        met_feats = ["phi","pt","sumEt","px","py"],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','nbtags'],
    )
    return cms_2017_0l

def class_2016_1l():
    # Options for classifier, cmssw 2016, using one lepton.
    class_2016_1l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","csv"],
        selection = 'nleps == 1 & systematic == 0',
        nleps = 1,
        met_feats = ['phi', 'pt', 'px', 'py'],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','event','run','lumi','systematic'],
    )
    return class_2016_1l

def class_2016_2l():
    # Options for classifier, cmssw 2016, using 2 leptons.
    class_2016_2l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","csv"],
        selection = 'nleps > 1 & systematic == 0',
        met_feats = ['phi', 'pt', 'px', 'py'],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','event','run','lumi','systematic'],
    )
    return class_2016_2l

def mass_1l():
    # Options for mass regression using 1 lepton.
    mass_1l = dict(
        selection = 'nleps == 1 & bb_nMatch == 2',
        nleps = 1,
        met_feats = ["phi","pt","px","py"],
        evdesc_feats = ["nleps", "njets", "nbtags", "nMatch_hb", "bb_nMatch", "m_bb"],
        truth_feats = None,
    )
    return mass_1l


def delphes_had():
    # Options for delphes files, full hadronic selection.
    delphes_had = dict(
        selection = 'nleps == 0',
        met_feats = None,
    )
    return delphes_had

def delphes_1l():
    # Options for delphes files, 1 lepton selection.
    delphes_1l = dict(
        selection = 'nleps == 1',
        nleps = 1,
        met_feats = ["phi","pt","px","py"],
        evdesc_feats = ["nleps", "njets", "nbtags", "nMatch_wq", "nMatch_tb",
                        "nMatch_hb"],
        truth_feats = None # Added because we will not work with partons.
    )
    return delphes_1l

def delphes_2l():
    # Options for delphes files, 2 lepton selection.
    delphes_2l = dict(
        selection = 'nleps > 1',
        met_feats = ["phi","pt","px","py"],
        truth_feats = None # Added because we will not work with partons.
    )
    return delphes_2l

def cms_1l():
    # Options for cms files, 1 lepton selection.
    cms_1l = dict(
        selection = 'nleps == 1',
        nleps = 1,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq",
                        "nMatch_tb", "nMatch_hb"],
    )
    return cms_1l

def cms_2l():
    # Options for cms files, 2 lepton selection.
    cms_2l = dict(
        selection = 'nleps > 1',
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq",
                        "nMatch_tb", "nMatch_hb"],
    )
    return cms_2l

def cms_0l():
    # Options for cms files, 0 lepton selection.
    cms_0l = dict(
        selection = 'nleps == 0',
        nleps = 0,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq",
                        "nMatch_tb", "nMatch_hb"],
    )
    return cms_0l

def load_files():
    """
    Load the files specified by the user in argparse. Write then in hdf format.
    """
    if args.infile.endswith(".h5"):
        print("Loading hdf file {0} ...".format(args.infile))
        return  pd.read_hdf(args.infile)

    file_paths = sorted(glob.glob(args.infile + '/data*.h5'))
    for path in files_paths:
        print("Loading hdf file {0} ...".format(path))
        if file_paths.index(path) == 0: data = pd.read_hdf(path)
        else: data = data.append(pd.read_hdf(path),ignore_index=True)

    data.to_hdf(args.outdir + "data.h5", key='data', format='t', mode='w')
    print("Data shape: {0}".format(data.shape))
    print("Data list: {0}".format(list(df)))

    return data


def set_data_column_headers(data):
    """
    Sets the column headers of the pandas hdf data structure.

    @data :: The pandas hdf data file, already loaded.

    @returns :: The data with updated column headers.
    """

    column_headers = []
    for column in data.columns:
        if column.startswith("jet"):
            attr = column.split('_')
            name = "jets_" + attr[1] + "_" + attr[2]
        elif column.startswith("lep"):
            attr = column.split('_')
            name = "leps_" + attr[1] + "_" + attr[2]
        else: name = column
        column_headers.append(name)
    data.columns = column_headers
    print("Data list: {0}".format(list(data)))

    return data

def hcand(X):
    # Make dijet higgs candidate combination.
    cmb = X[0]
    if type(cmb[0]) == list: return np.zeros((2,njf), np.float32)
    jets = X[1:].values.reshape(1, -1, njf)
    return jets[:, cmb[0,0]].astype(np.float32)

def pad(X, npad=6):
    # Pad top kinematic fit solutions.
    if len(X.shape) < 4:    X = np.zeros((npad,8,4,2))
    elif X.shape[0] < npad: X = np.vstack([X,np.zeros((6-X.shape[0],8,4,2))])
    elif X.shape[0] > npad: X = X[:npad]
    return X.reshape(-1,*X.shape)

def jet_formatting(data, flats):
    """
    Formatting the jets features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting jets...')

    jetsa = None
    onejet = list(range(njets)); number_jet_feats = len(jet_feats)
    jet_col = ["jets_%s_%d"%(feat,jet) for jet in onejet for feat in jet_feats]
    jetsa = data[jet_col].values
    flats.append(jetsa); jetsa = jetsa.reshape(-1, njets, number_jet_feats)
    np.save(args.outdir + "/jets", jetsa)
    print('Jet formatting done.', jetsa.shape)

    return flats

def lep_formatting(data, flats):
    """
    Formatting the lepton features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting leptons...')

    lepsa = None
    number_lep_feats = len(lep_feats)
    lepsa = data[["leps_%s_%d" % (feat,lep) for lep in range(nleps)
        for feat in lep_feats]].values
    flats.append(lepsa)
    lepsa = lepsa.reshape(-1, nleps, number_lep_feats)
    np.save(args.outdir + "/leps", lepsa)
    print('Lepton formatting done.', lepsa.shape)

    return flats

def met_formatting(data, flats):
    """
    Formatting the meta features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting metadata features...')

    meta = None
    data["met_px"] = data["met_" + met_feats[1]] *
        np.cos(data["met_"+met_feats[0]])
    data["met_py"] = data["met_" + met_feats[1]] *
        np.sin(data["met_"+met_feats[0]])
    meta = data[["met_%s" % feat for feat in met_feats]].values
    flats.append(meta)
    np.save(args.outdir + "/met", meta)
    print('Metadata formatting done.', meta.shape)

    return flats

def jet_combinations_formatting(data, flats):
    """
    Formatting the jet combinations features: higgs candidates and kinematic
    fit solutions.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """

    print('Formatting jet combinations...')

    hcanda = None; kina = None
    twojets = list(itertools.combinations(onejet,2))
    twojets2ind ={cmb:icomb for icomb,cmb in enumerate(twojets)}
    jet_cols = ["jets_cmb"]+jet_feat_cols+["jets_jets_m2_%d%d" % x
        for x in twojets]

    data["kin_sols"] = data["kin_sols"].apply(pad)
    hcanda = np.vstack(data[["jets_cmb"] + \
        jet_feat_cols].apply(hcand,axis=1,raw=True).tolist())
    kina = np.vstack(data["kin_sols"].tolist())

    flats.append(hcanda.reshape(hcanda.shape[0],-1))
    flats.append(kina.reshape(kina.shape[0],-1))
    np.save(args.outdir+"/hcand", hcanda)
    np.save(args.outdir+"/kinsols", kina)
    print('Jet combinations formatting done.', hcanda.shape, kina.shape)

    return flats

def truth_level_formatting(data, feats):
    """
    Formatting the truth information features.

    @data  :: The pandas hdf data file, already loaded.
    @flats :: Array containing the values of different features, flatly.

    @returns :: The updated flats array.
    """
    print('Formatting truth level features...')
    ntf = len(truth_feats)
    trutha = data[["%s_%s" % (part,feat) for feat in truth_feats
        for part in ["jlr_top","jlr_atop","jlr_bottom","jlr_abottom"]]].values
    trutha = trutha.reshape(-1, 4, ntf)
    flats.append(trutha)
    np.save(args.outdir + "/truth", trutha)
    print('Done formatting turth level features.')

    return flats

def make_jlr_target(data):
    # Make the target for the jlr classifier.
    jlra = data["JointLikelihoodRatioLog"].values
    np.save(args.outdir + "/target", jlra)
    print('Done making the jlr target.', jlra.shape)

def make_mbb_target(data):
    # Make the target for the mbb classifier.
    m_bb = data["m_bb"].values
    np.save(args.outdir + "/target", m_bb)
    print('Done making the mbb target.', m_bb.shape)

def make_higgsclass_target(data):
    # Make the target for the higgs classifier.
    list_of_flags = []; list_of_masses = []
    for ijet, jjet in itertools.combinations(reversed(range(10)),2):
        matchFlag_ijet = data["jets_matchFlag_{0}".format(ijet)].values
        matchFlag_jjet = data["jets_matchFlag_{0}".format(jjet)].values
        dijet_label = ((matchFlag_ijet + matchFlag_jjet)>1).astype(int)
        list_of_flags.append(dijet_label)
        mass_squared = make_m2(data,"jets",ijet,"jets",jjet)
        mass  = np.sqrt(mass_squared)
        list_of_masses.append(mass)
    target = np.array(list_of_flags).T
    dijet_masses = np.array(list_of_masses).T
    np.save(args.outdir + "/target", target)
    print("Done making the higgsclass target.", target.shape)
    print("Making dijet_masses...")
    np.save(args.outdir + "/dijet_masses", dijet_masses, allow_pickle=False,
        fix_imports=True)
    print("Done making dijet masses.", dijet_masses.shape)

def make_multiclass_target(data):
    # Make the target for the multi classifier.
    target = np.zeros((data.shape[0],6))
    if "ttH" in args.dataset: target[:,0] = 1
    elif "TT" in args.dataset:
        ttCls = data["ttCls"].values
        ttbb = ((ttCls >= 53) & (ttCls <=56))
        target[np.ix_(ttbb, [1])] = 1
        tt2b = (ttCls == 52)
        target[np.ix_(tt2b, [2])] = 1
        ttb = (ttCls == 51)
        target[np.ix_(ttb, [3])] = 1
        ttcc = ((ttCls >= 41) & (ttCls <=45))
        target[np.ix_(ttcc, [4])] = 1
        target[np.ix_(~(ttbb | tt2b | ttb | ttcc), [5])] = 1
    else: raise Exception("Unknown dataset, not sure if signal or background.")
    np.save(args.outdir + "/target", target)
    print('Done making multiclass target.', target.shape)

def make_binarclass_target(data):
    # Make the target for the binary classifier.
    if args.dataset is not None:
        if "ttH" in args.dataset: label = np.full(data.shape[0], 1)
        elif "TT" in args.dataset: label = np.full(data.shape[0], 0)
        else: raise Exception("Dataset is not correctly specified in args.")
        np.save(args.outdir + "/target", label)
        print('Done making binaryclass target.', label.shape)

def make_mem(data):
    # Make MEM -> no idea what this is... needs clarification.
    print('Making MEM...')
    arr = data[["mem_tth_SL_2w2h2t_p", "mem_ttbb_SL_2w2h2t_p"]].values
    np.save(args.outdir + "/mem", arr)
    print('Done making the mem.', arr.shape)

def make_event_desc(data):
    # Making the event description features.
    print('Making evdesc features...')
    arr = data[evdesc_feats].values
    np.save(args.outdir + "/evdesc", arr)
    print('Done making evdesc features.', arr.shape)

if __name__ == "__main__":
    main()
