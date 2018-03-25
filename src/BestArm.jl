module BestArm
    using Distributions
    using PyPlot
    using HDF5

    include("arms.jl")
    include("utils.jl")
    include("kl_functions.jl")
    include("view_results.jl")
end
