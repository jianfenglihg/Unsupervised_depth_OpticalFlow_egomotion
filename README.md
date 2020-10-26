# Unsupervised_geometry
    # mask 
        ## occlusion √
        ## dynamic √
        ## hole no texture √

    # photometric
        ## rigid
            ### DP √

        ## dynamic
            ### DPDy
            ### F √
    
    # geometry
        ## rigid
            ### DPF √
            ### PFD 
            ### FP √
            ### DFP->(PnP)

        ## dynamic
            ### DPDyF
    
    # consis
        ## F bwd fwd consis √
        ## D prev next consis √
        ## P t t+1 t+2

    # network architecher
        ## smaller
        ## attention
        ## CRF

    # auto_set hyper_parameters->loss_weight