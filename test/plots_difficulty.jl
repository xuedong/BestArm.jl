using HDF5
using StatsPlots

reservoir = "Beta"
dist = "Bernoulli"
alphas = [1.0, 1.0, 1.0, 1.0, 1.0]
betas = [1.0, 2.0, 3.0, 4.0, 5.0]
# alphas = [1.0, 3.0, 1.0, 0.5, 2.0, 5.0, 2.0, 0.3]
# betas = [1.0, 1.0, 3.0, 0.5, 5.0, 2.0, 2.0, 0.7]
# num = 16
budget = 160
mcmc = 1
len = length(alphas)
limit = budget

policy = BestArm.ttts_dynamic
policy_name = "Dynamic TTTS"
abrev = "dttts"


mn = [20, 35, 30, 35, 27, 25, 32, 34, 20, 25, 20, 35, 30, 35, 27, 25, 32, 34, 20, 25, 20, 35, 30, 35, 27, 25, 32, 34, 20, 25, 20, 35, 30, 35, 27, 25, 32, 34, 20, 25, 20, 35, 30, 35, 27, 25, 32, 34, 20, 25]
sx = repeat(["k=1", "k=2", "k=3", "k=4", "k=5"], inner = 10)
# std = [2, 3, 4, 1, 2, 3, 5, 2, 3, 3]
nam = repeat(["i=1", "i=2", "i=3", "i=4", "i=5", "i=6", "i=7", "i=8", "i=9", "i>=10     "], outer = 5)


groupedbar(nam, mn, group = sx, ylabel = "Number of arms", title = "Number of arms pulled by i times", size=(600, 400))

# if Sys.KERNEL == :Darwin
#         savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/difficulty/", "test.pdf"))
# elseif Sys.KERNEL == :Linux
#         savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/difficulty/", "test.pdf"))
# end

num_arms1 = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/infinite/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts", "_N.h5"), "r") do file
read(file, "dttts")
end

png("test")
