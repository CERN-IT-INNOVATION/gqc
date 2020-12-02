from pyjlr.utils import make_p4, make_m2
import argparse
import os
import pandas as pd
import numpy as np
import glob

if __name__ == "__main__":  
# default options
    opts = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btag"],
        njets = 10,
        lep_feats = ["pt","eta","phi","en","px","py","pz"],
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        truth_feats = ["pt","eta","phi","en","px","py","pz"],
        evdesc_feats = ["nleps", "njets"],
    )

    # options for classifier, cmssw 2016
    cms_2017_1l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        selection = 'nleps == 1',
        nleps = 1,
        truth_feats = None,
        evdesc_feats = ['evt','run','lumi','njets','nleps','nBDeepCSVM'],
    )

    cms_2017_2l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        selection = 'nleps == 2',
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','nbtags'],
    )

    cms_2017_0l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        selection = 'nleps == 0',
        nleps = 0,
        met_feats = ["phi","pt","sumEt","px","py"],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','nbtags'],
    )

    # options for classifier, cmssw 2016
    class_2016_1l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","csv"],
        selection = 'nleps == 1 & systematic == 0',
        nleps = 1,
        met_feats = ['phi', 'pt', 'px', 'py'],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','event','run','lumi','systematic'],
    )

    class_2016_2l = dict(
        jet_feats = ["pt","eta","phi","en","px","py","pz","csv"],
        selection = 'nleps > 1 & systematic == 0',
        met_feats = ['phi', 'pt', 'px', 'py'],
        truth_feats = None,
        evdesc_feats = ['njets','nleps','event','run','lumi','systematic'],
    )

    # options for mass regression
    mass_1l = dict(
        selection = 'nleps == 1 & bb_nMatch == 2',
        nleps = 1,
        met_feats = ["phi","pt","px","py"],
        evdesc_feats = ["nleps", "njets", "nbtags", "nMatch_hb", "bb_nMatch", "m_bb"],
        truth_feats = None,
    )

    # options for delphes files, full hadronic selection
    delphes_had = dict(
        selection = 'nleps == 0',
        met_feats = None,
    )
    
    # options for delphes files, 1l selection
    delphes_1l = dict(
        selection = 'nleps == 1',
        nleps = 1,
        met_feats = ["phi","pt","px","py"],
        evdesc_feats = ["nleps", "njets", "nbtags", "nMatch_wq", "nMatch_tb", "nMatch_hb"],
        truth_feats = None #Added because we will not work with partons.
    )
    delphes_2l = dict(
        selection = 'nleps > 1',
        met_feats = ["phi","pt","px","py"],
        truth_feats = None #Added because we will not work with partons.   
    )

    # options for cms files, 1l selection
    cms_1l = dict(
        selection = 'nleps == 1',
        nleps = 1,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq", "nMatch_tb", "nMatch_hb"],
    )
    
    cms_2l = dict(
        selection = 'nleps > 1',
        nleps = 2,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq", "nMatch_tb", "nMatch_hb"],
    )
    
    cms_0l = dict(
        selection = 'nleps == 0',
        nleps = 0,
        met_feats = ["phi","pt","sumEt","px","py"],
        jet_feats = ["pt","eta","phi","en","px","py","pz","btagDeepCSV"],
        evdesc_feats = ["nleps", "njets", "nBDeepCSVM", "nMatch_wq", "nMatch_tb", "nMatch_hb"],
    )

    datatype_choices = {
        "cms_0l": cms_0l,
        "cms_1l": cms_1l,
        "cms_2l": cms_2l,
        "delphes_1l": delphes_1l,
        "delphes_2l": delphes_2l,
        "delphes_had": delphes_had,
        "mass_1l": mass_1l,
        "class_2016_1l": class_2016_1l,
        "class_2016_2l": class_2016_2l,
        "cms_2017_0l": cms_2017_0l,
        "cms_2017_1l": cms_2017_1l,
        "cms_2017_2l": cms_2017_2l,
    }


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", type=str,
        default="/scratch/{0}/jlr/".format(os.environ["USER"]), action="store",
        help="output directory"
    )
    parser.add_argument(
        "--infile", type=str,
        required=True, action="store",
        help="input folder"
    )
    parser.add_argument(
        "--datatype", type=str,
        choices=datatype_choices.keys(),
        required=True,
        help="datatype choice"
    )
    parser.add_argument(
        "--dataset", type=str,
        help="name of dataset, determines also if ttCls is used or not"
    )
    parser.add_argument(
        "--target", type=str,
        required=True,
        help="regression target (binary_classifier, multi_classifier, Higgs_classifier, mbb, jlr)"
    )

    
    args = parser.parse_args()

    choose = datatype_choices[args.datatype]
    
    # copy specific options
    opts.update(choose)
    
    # copy default values to globals
    globals().update(opts)
   
    if args.infile.endswith(".h5"):  
        print("loading hdf file {0}".format(args.infile))
        df = pd.read_hdf(args.infile)
    else:
        files = sorted(glob.glob(args.infile + '/data*.h5'))
        for f in files:
            print("loading hdf file {0}".format(f))
            if files.index(f) == 0:
                df = pd.read_hdf(f)
            else:
                d = pd.read_hdf(f)
                df = df.append(d, ignore_index=True)
        df.to_hdf(args.outdir + "data.h5", key='df', format='t', mode='w')
    print("df.shape={0}".format(df.shape))
    
    print(list(df))
    column_headers = [] 
    for c in df.columns:
        if c.startswith("jet"):
            attr = c.split('_')
            name = "jets_" + attr[1] + "_" + attr[2]
        elif c.startswith("lep"):
            attr = c.split('_')
            name = "leps_" + attr[1] + "_" + attr[2]
        else:
            name = c
        column_headers.append(name)
    df.columns = column_headers
    print(list(df))

    if selection is not None:
        df = df.query(selection)
    
    jetsa = None
    hcanda = None
    lepsa = None
    meta = None
    trutha = None
    kina = None

    os.makedirs(args.outdir)
    
    # make dijet higgs candidate combination
    def hcand(X):    
        cmb = X[0]
        # print(cmb)
        if type(cmb[0]) == list:
            return np.zeros( (2,njf),np.float32 )
        jets = X[1:].values.reshape(1,-1,njf)
        return jets[:,cmb[0,0]].astype(np.float32)
    
    
    # pad top kinematic fit solutions
    def pad(X,npad=6):
        if len(X.shape) < 4:
            X = np.zeros((npad,8,4,2))
        elif X.shape[0] < npad:
            X = np.vstack([X,np.zeros((6-X.shape[0],8,4,2))])
        elif X.shape[0] > npad:
            X = X[:npad]
        return X.reshape(-1,*X.shape)
    
    
    flats = []
    
    # --------------------------------------------------------------------------------------------------------------
    # jets
    if jet_feats is not None:
        print('formatting jets...')
        onejet = list(range(njets))
        #for ijet in onejet:
        #    make_p4(df,'jets',ijet)
        njf = len(jet_feats)
        jet_feat_cols = ["jets_%s_%d" % (feat,jet) for jet in onejet for feat in jet_feats  ]
        jetsa = df[jet_feat_cols].values
        flats.append(jetsa)
        jetsa = jetsa.reshape(-1,njets,njf)
        np.save(args.outdir+"/jets",jetsa)
        print('done',jetsa.shape)
        
    # --------------------------------------------------------------------------------------------------------------
    # leptons
    if lep_feats is not None and nleps>0:
        print('formatting leptons...')
        nlf = len(lep_feats)
        #for ilep in range(nleps):
        #    make_p4(df,'leps',ilep)
        lepsa = df[ ["leps_%s_%d" % (feat,lep) for lep in range(nleps) for feat in lep_feats ]  ].values
        flats.append(lepsa)
        lepsa = lepsa.reshape(-1,nleps,nlf) 
        np.save(args.outdir+"/leps",lepsa)
        print('done',lepsa.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # met
    if met_feats is not None:
        print('formatting met...')
        df["met_px"] = df["met_"+met_feats[1]]*np.cos(df["met_"+met_feats[0]])
        df["met_py"] = df["met_"+met_feats[1]]*np.sin(df["met_"+met_feats[0]])
        meta = df[ ["met_%s" % feat for feat in met_feats  ]  ].values 
        flats.append(meta)
        np.save(args.outdir+"/met",meta)
        print('done',meta.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # flat array with all above
    #print('making flat (nokin) features...')
    #flata = np.hstack(flats)
    #np.save(args.outdir+"/flat_nokin",flata)
    #print('done',flata.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # jet combinations: higgs candidates and top kin fit solutions
    if jet_feats is not None and "jet_cmb" in df.columns:
        print('formatting jet combinations...')
        twojets = list(itertools.combinations(onejet,2))
    
        twojets2ind ={  cmb:icomb for icomb,cmb in enumerate(twojets)  }
        jet_cols = ["jets_cmb"]+jet_feat_cols+["jets_jets_m2_%d%d" % x for x in twojets]
        
        df["kin_sols"] = df["kin_sols"].apply(pad)#.apply(lambda x: pad_sequences(x,6,value=np.zeros() ).shape)
        
        hcanda = np.vstack( df[["jets_cmb"]+jet_feat_cols].apply(hcand,axis=1,raw=True).tolist() )
        kina = np.vstack(df["kin_sols"].tolist())
    
        flats.append(hcanda.reshape(hcanda.shape[0],-1))
        flats.append(kina.reshape(kina.shape[0],-1))
        np.save(args.outdir+"/hcand",hcanda)
        np.save(args.outdir+"/kinsols",kina)
        print('done',hcanda.shape,kina.shape)    
        
    # --------------------------------------------------------------------------------------------------------------
    # flat arrat with all above
    #print('making flat features...')
    #flata = np.hstack(flats)
    #np.save(args.outdir+"/flat",flata)
    #print('done',flata.shape)
    # --------------------------------------------------------------------------------------------------------------
    # target
    print('making target...')

    if args.target == "jlr": 
        jlra = df["JointLikelihoodRatioLog"].values
        np.save(args.outdir+"/target",jlra)
        print('done',jlra.shape)

    if args.target == "mbb":
        m_bb = df["m_bb"].values
        np.save(args.outdir+"/target",m_bb)
        print('done',m_bb.shape)

    if args.target == "Higgs_classifier":
        list_of_flags = []
        list_of_masses = []
        import itertools
        for ijet, jjet in itertools.combinations(reversed(range(10)),2):
            matchFlag_ijet = df["jets_matchFlag_{0}".format(ijet)].values
            matchFlag_jjet = df["jets_matchFlag_{0}".format(jjet)].values
            dijet_label = ((matchFlag_ijet + matchFlag_jjet)>1).astype(int)
            list_of_flags.append(dijet_label)

            m2 = make_m2(df,"jets",ijet,"jets",jjet)
            m = np.sqrt(m2)
            list_of_masses.append(m)

        target = np.array(list_of_flags).T
        dijet_masses = np.array(list_of_masses).T
        np.save(args.outdir+"/target",target)
        print("done",target.shape)
        print("making dijet_masses...")
        np.save(args.outdir+"/dijet_masses",dijet_masses, allow_pickle=False, fix_imports=True)
        print("done",dijet_masses.shape)

    if args.target == "multi_classifier":
        target = np.zeros((df.shape[0],6))
        if "ttH" in args.dataset:
            target[:,0] = 1
        elif "TT" in args.dataset:
            ttCls = df["ttCls"].values

            ttbb = ((ttCls >= 53) & (ttCls <=56))
            target[np.ix_(ttbb, [1])] = 1
            tt2b = (ttCls == 52)
            target[np.ix_(tt2b, [2])] = 1
            ttb = (ttCls == 51) 
            target[np.ix_(ttb, [3])] = 1
            ttcc = ((ttCls >= 41) & (ttCls <=45))
            target[np.ix_(ttcc, [4])] = 1
            target[np.ix_(~(ttbb | tt2b | ttb | ttcc), [5])] = 1
        else:
            raise Exception("unknown dataset, not sure if signal or background.")
        np.save(args.outdir+"/target",target)
        print('done',target.shape)


    if args.target == "binary_classifier":
        if args.dataset is not None:
            if "ttH" in args.dataset:
                label = np.full(df.shape[0], 1)
            elif "TT" in args.dataset:
                label = np.full(df.shape[0], 0)
            else:
                raise Exception("Dataset is not correctly specified in arguments.")
            np.save(args.outdir+"/target",label)
            print('done',label.shape)

    # -------------------------------------------------------------------------------------------------------------
    # MEM
    if 'mem_tth_SL_2w2h2t_p' in df and 'mem_ttbb_SL_2w2h2t_p' in df:
        print('making MEM...')
        arr = df[["mem_tth_SL_2w2h2t_p", "mem_ttbb_SL_2w2h2t_p"]].values
        np.save(args.outdir+"/mem",arr)
        print('done', arr.shape)
    # --------------------------------------------------------------------------------------------------------------
    # Event description features
    print('making evdesc...')
    arr = df[evdesc_feats].values
    np.save(args.outdir+"/evdesc",arr)
    print('done', arr.shape)
    
    # --------------------------------------------------------------------------------------------------------------
    # truth level info
    if truth_feats is not None:
        print('formatting truth...')
        ntf = len(truth_feats)
        trutha = df[ ["%s_%s" % (part,feat) for feat in truth_feats for part in ["jlr_top","jlr_atop","jlr_bottom","jlr_abottom"]  ]  ].values 
        trutha = trutha.reshape(-1,4,ntf)
        np.save(args.outdir+"/truth",trutha)    
        print('done')

    # ---------------------------------------------------------------------------------------------------------------
    # write event information to json file
    '''#key doesn't exist
    evt_info = {}
    evt_info[args.dataset] = df["evt"].values
    evt_info[args.dataset] = evt_info[args.dataset].astype(np.int64)
    evt_info[args.dataset] = evt_info[args.dataset].tolist()
    import json
    with open(args.outdir+"/evt.json", 'w') as outfile:
        json.dump(evt_info, outfile)
    print('event info json written')
    '''