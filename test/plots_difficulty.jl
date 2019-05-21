using HDF5
using StatsPlots

reservoir = "Beta"
dist = "Bernoulli"
alphas = [1.0, 1.0, 1.0, 1.0, 1.0]
betas = [1.0, 2.0, 3.0, 4.0, 5.0]
# alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
# betas = [1.0, 1.0, 1.0, 1.0, 1.0]

budget = 160
mcmc = 100
len = length(alphas)

for _ in 1:1
        pulls = zeros(1, 0)
        for iparam in 1:len
                if Sys.KERNEL == :Linux
                        arms = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/difficulty/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts.h5"), "r") do file
                                read(file, "dttts")
                        end
                elseif Sys.KERNEL == :Darwin
                        arms = h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/difficulty/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts.h5"), "r") do file
                                read(file, "dttts")
                        end
                end

                arms /= mcmc
                pulls = hcat(pulls, arms)
        end

        # group = repeat(["alpha=1", "alpha=2", "alpha=3", "alpha=4", "alpha=5"], inner = 10)
        group = repeat(["beta=1", "beta=2", "beta=3", "beta=4", "beta=5"], inner = 10)
        # std = [2, 3, 4, 1, 2, 3, 5, 2, 3, 3]
        xtick = repeat(["i=1", "i=2", "i=3", "i=4", "i=5", "i=6", "i=7", "i=8", "i=9", "i>=10"], outer = 5)


        groupedbar(xtick, pulls, group=group, ylabel="Number of arms", title="Number of arms pulled by i times", size=(600, 400))

        # if Sys.KERNEL == :Darwin
        #         savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/difficulty/", "test.pdf"))
        # elseif Sys.KERNEL == :Linux
        #         savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/difficulty/", "test.pdf"))
        # end

        png("test")
end
