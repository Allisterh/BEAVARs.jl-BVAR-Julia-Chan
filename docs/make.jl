using Pkg
Pkg.activate(@__DIR__)
using Documenter, BEAVARs, TimeSeries, LiveServer, DocumenterTools

makedocs(
    sitename="BEAVARs.jl",
    pages = [
        "Introduction" => "introduction.md",
        "Priors" => "priors.md",
        "Methods" => "methods.md",
        "Forecasting" => "forecasting.md",
        "Structural analysis" => "irfs.md",
        "File library" => [
            "Constructors" => "Constructors.md",
            "Initialization" => "init_functions.md",
            "Data transformation" => "dataPrep.md"
        ],
        "References" => "references.md"
    ]
)
deploydocs(
    target = "build",
    repo = "github.com/borisblagov/BEAVARs.jl.git",
)