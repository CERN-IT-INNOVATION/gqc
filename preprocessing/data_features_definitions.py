# Select certain features depending on data_type.

def choose_data_type(user_choice):
    """
    Determines which data type is used in the formatting.

    @user_choice :: String of the data type given by the user.

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
