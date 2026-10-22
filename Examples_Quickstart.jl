using BEAVARs
using TimeSeries, Dates     # these are required only for the example, your data may already have a time-series format

# General setup
T = 80;     # data span
n = 3;      # number of varialbes
L = 2;      # number of lags
nburn = 50;# burn-indraws
nsave = 20;# draws to save
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2020,1,1)+Quarter(T-1),rand(T,n))

# Chan2020minn
model_type1, set_struct, hyp_struct= makeSetup("Chan2020minn",n_burn=nburn;n_save=nsave,p=L)
data_struct = makeDataSetup(model_type1,data)
out_struct = beavar(model_type1, set_struct, hyp_struct, data_struct)

# Chan2020iniw
model_type2, set_struct, hyp_struct= makeSetup("Chan2020iniw",n_burn=nburn;n_save=nsave,p=L)
data_struct = BEAVARs.makeDataSetup(model_type2,data);
out_struct = beavar(model_type2, set_struct, hyp_struct, data_struct);

# CCK2016csv
model_type3, set_struct, hyp_struct= makeSetup("CCK2016csv",n_burn=nburn;n_save=nsave,p=L)
data_struct = BEAVARs.makeDataSetup(model_type3,data);
out_struct = beavar(model_type3, set_struct, hyp_struct, data_struct);


# Chan2020csv
model_type4, set_struct, hyp_struct= makeSetup("Chan2020csv",n_burn=nburn;n_save=nsave,p=L)
data_struct = BEAVARs.makeDataSetup(model_type4,data);
out_struct = beavar(model_type4, set_struct, hyp_struct, data_struct);


# Blagov2027
model_type5, set_struct, hyp_struct= makeSetup("Blagov2027",n_burn=nburn;n_save=nsave,p=L)
dataHF_tab, dataLF_tab, varList = BEAVARs.readSpecMF("bg_L250911","data/Specifications_mfvar.xlsx");
data_struct = makeDataSetup(model_type5,dataHF_tab, dataLF_tab)
out_struct = beavar(model_type5,set_struct,hyp_struct,data_struct)
# out_struct = beavar(model_type,set_struct,hyp_struct,data_struct);


# BGR2010
model_type, set_struct, hyp_struct= makeSetup("BGR2010",n_burn=20;n_save=50,p=2)
data = TimeArray(DateTime(2020,1,1):Quarter(1):DateTime(2027,4,1),rand(30,3))
data_struct = makeDataSetup(model_type,data)
out_struct = beavar(model_type, set_struct, hyp_struct, data_struct)