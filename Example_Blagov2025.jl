using BEAVARs
# Blagov2025
model_type, set_struct, hyp_struct= makeSetup("CPZ2023",n_burn=20;n_save=50,p=2)
dataHF_tab, dataLF_tab, varList = BEAVARs.readSpec("bg_L250911","data/Specifications_mfvar.xlsx");
data_struct = makeDataSetup(model_type,dataHF_tab, dataLF_tab,0)
out_struct = beavar(model_type,set_struct,hyp_struct,data_struct)

Yfit, Yact = BEAVARs.modelFit(out_struct,set_struct)
plot(Yfit[:,2])
plot!(Yact[:,2])

fcast_struct = BEAVARs.forecast(out_struct,set_struct,data_struct)
BEAVARs.forecast_plot(fcast_struct,plot_fcastOnly=1)

