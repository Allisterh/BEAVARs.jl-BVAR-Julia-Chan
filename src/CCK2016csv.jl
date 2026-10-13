# ------------------------------------------------------------------
# Minnesota prior with common stochastic volatility as in Carriero, A., Clark, T. E., & Marcellino, M. (2016). Common Drifting Volatility in Large Bayesian VARs. Journal of Business & Economic Statistics, 34(3), 375–390. https://doi.org/10.1080/07350015.2015.1040116
# Differences to the CCK (2016) implementation lie in the specification of the Minnesota prior, where we introduce cross-variable shrinkage as in Chan2020iniw


@doc raw"""
    makeHypSetup(::CCK2016csv_type)

    Generate a hyperparameter strcture for use with the single-frequency models
    
"""
function makeHypSetup(::CCK2016csv_type)
    return hypChan2020()
end


@doc raw"""
    makeDataSetup(::CCK2016csv_type,data_tab::TimeArray; var_list =  colnames(data_tab))

Generate a dataset strcture for use with the single-frequency models
    
# Arguments
    model_type: The custom model type (not a string)
    data_tab:   TimeArray with the data_de
    var_list:   A symbol list with the variable names. Will be used for oredering the variables. Uses by default the names from data_tab if not supplied. note that Symbol lists have a particular synthax.
# Returns
    data_struct: A dataBVAR_TA structure with the data and metadata
"""
function makeDataSetup(::CCK2016csv_type,data_tab::TimeArray; var_list =  colnames(data_tab))
    return data_BVAR(data_tab,values(data_tab), var_list)
end





#-------------------------------------
# The Den of the beavar
#-------------------------------------

function beavar(::CCK2016csv_type, set_struct, hyp_str, data_struct)
    println("Hello Independent Normal Inverse Wishart")
    @unpack data_tab, data_mat, var_list = data_struct;
    freqL_date = BEAVARs.get_data_freq(data_tab);
    datesLF = timestamp(data_tab);
    datesLF_fcast = collect(datesLF[end]+freqL_date:freqL_date:datesLF[end]+freqL_date*(set_struct.n_fcst));
    fdatesLF = [datesLF;datesLF_fcast];
    YY = values(data_struct.data_tab);
    store_β, store_Σ = Chan2020iniw(YY,set_struct,hyp_str);
    out_struct = VAROutput_Chan2020iniw(store_β,store_Σ,YY,fdatesLF)
    return out_struct
end




@with_kw struct VAROutput_CCK2016csv <: BVARmodelOutput
    store_β::Array{}        # 
    store_Σ::Array{}        # 
    store_h::Array{}        # 
    s2_h_store::Array{}     # 
    store_ρ::Array{}        # 
    store_σ_h2::Array{}     # 
    store_eh::Array{}       #
    YY::Array{}             #
end



@doc raw"""
    BEAVARs.init_Minn_csv(YY, p)

"""
function init_Minn_csv(YY,set_struct,hyp_struct)
    @unpack p = set_struct;
    if set_struct.const_loc == 1
        # will err if false, on purpose to catch if we define the constant wrong.
        Y, X, T, n, intercept       = mlagL(YY,p);
    end
    sigmaP_ar4, betaOLS_ar4     = ar4(YY);                                  # do OLS to initialize priors as in CCK2016
    k                           = n*p+intercept
    h                           = zeros(T,);                                # stochastic volatilities
    S_0                         = diagm(sigmaP);                            # initialize S_0
    Σt_inv                      = S_0\I;                                    # initialize Σ^-1   
    Σ_invsp, Σt_LI              = BEAVARs.makeBlkDiag(T*n,n,0,Σt_inv);      # I(T) ⊗ Σ-1 and its indices for update
    XtΣ_inv_den                 = zeros(k*n,T*n);                           # will be X' ( I(T) ⊗ Σ-1 )   from page 6 in Chan 2020 LBA
    XtΣ_inv_X                   = zeros(n*k,n*k);                           # will be X' ( I(T) ⊗ Σ-1 ) X from page 6 in Chan 2020 LBA    
    Xsur                        = BEAVARs.XSurFormMatrix(X,n);              # prepares the SUR form and the indices of the parameters for updating
    K_β                         = zeros(n*k,n*k);                           # Variance covariance matrix of the parameters
    
    priorMinn_struct = BEAVARs.init_priorMinn(n,p,sigmaP_ar4,set_struct.prior_RW,hyp_struct)

    β_hat_struct = BEAVARs.make_βdraw_struct(similar(priorMinn_struct.Vinvβ_prior),similar(priorMinn_struct.Vinvβ_prior),similar(priorMinn_struct.Vinvβ_prior), Xsur, XtΣ_inv_den, XtΣ_inv_X, Σ_invsp, K_β)

    Uvec = similar(vec(Y'))
    Σt   = similar(S_0)
    # iniw_Σt_struct = BEAVARs.make_iniw_Σt_struct(Uvec,Σt,Σt_inv,S_0,hyp_struct.nu0)
    return Y, X, T, n, sigmaP_ar4, betaOLS_ar4, priorMinn_struct, β_hat_struct, iniw_Σt_struct, Σt_LI
end