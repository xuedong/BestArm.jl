using PyPlot
using HDF5
using Seaborn
using Distributions

reservoir = "Beta"
alphas = [1.0, 2.0, 3.0, 1.0]
betas = [1.0, 2.0, 1.0, 3.0]
budget = 64

k = 10

for iparam in 1:1
    fig = figure()
    Seaborn.set()
    x = rand(Beta(2, 2), 100000)
    Seaborn.distplot(x, hist=false)

    xlabel("Allocation budget")
    ylabel("Expectation of the simple regret")
    grid("on")
    legend(loc=1)
    if Sys.KERNEL == :Darwin
        savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_fictif.pdf"))
    elseif Sys.KERNEL == :Linux
        savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_fictif.pdf"))
    end
end
