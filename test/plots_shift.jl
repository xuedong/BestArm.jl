using HDF5
using StatsPlots

reservoir = "ShiftedBeta"
dist = "Bernoulli"
alphas = [0.5]
betas = [0.5]
shifts = [0.2, 0.4, 0.6, 0.8, 1.0]

budget = 160
mcmc = 100

for iparam in 1:1
        pulls = zeros(1, 0)
        for i in 1:length(shifts)
                if Sys.KERNEL == :Linux
                        arms = h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/shift_fix/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts_", shifts[i], ".h5"), "r") do file
                                read(file, "dttts")
                        end
                elseif Sys.KERNEL == :Darwin
                        arms = h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/shift_fix/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", "dttts_", shifts[i], ".h5"), "r") do file
                                read(file, "dttts")
                        end
                end

                arms /= mcmc
                pulls = hcat(pulls, arms)
        end

        group = repeat(["shift=0.2", "shift=0.4", "shift=0.6", "shift=0.8", "shift=1.0"], inner = 10)
        # group = repeat(["shift=0.2, effectively sampled=80.95", "shift=0.4, effectively sampled=80.36", "shift=0.6, effectively sampled=78.51", "shift=0.8, effectively sampled=73.43", "shift=1.0, effectively sampled=54.09"], inner = 10)
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
