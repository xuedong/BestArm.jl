using PyPlot
using ProgressMeter
using Distributed
using HDF5

addprocs(3)
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
# @everywhere using BestArm
@everywhere using DistributedArrays

# Problem setting
# reservoir = "ShiftedBeta"
reservoir = "ShiftedBeta"
dist = "Bernoulli"
alphas = [0.5, 1.0, 2.0, 3.0, 1.0]
betas = [0.5, 1.0, 2.0, 1.0, 3.0]
# alphas = [1.0, 3.0, 1.0, 0.5, 2.0, 5.0, 2.0, 0.3]
# betas = [1.0, 1.0, 3.0, 0.5, 5.0, 2.0, 2.0, 0.7]
# num = 16
budget = 160
mcmc = 1000
shifts = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
limit = budget

policy = BestArm.ttts_dynamic
policy_name = "Dynamic TTTS"
abrev = "dttts"

# Options
VERBOSE = true
SAVE = false


# Tests
for iparam in 1:5
	fig = figure()
	arms = zeros(1, length(shifts))
	for i in 1:length(shifts)
		@showprogress 1 string("Computing ", policy_name, "...") for k in 1:mcmc
			rec, N, recs, mu = policy(reservoir, 1, limit, budget, dist, 0.5, false, alphas[iparam], betas[iparam], true, shifts[i])
			# regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
			# regrets += regrets_current
			arms[i] += length(filter(x -> x>1, N))
		end
		# plot(X, reshape(regrets/mcmc, budget, 1), linestyle="-.", label=policy_name)
	end

	if Sys.KERNEL == :Darwin
		h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/shift/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrev, "_", "_shift.h5"), "w") do file
			write(file, abrev, arms)
		end
	elseif Sys.KERNEL == :Linux
		h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/shift/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrev, "_", "_shift.h5"), "w") do file
			write(file, abrev, arms)
		end
	end

	arms /= mcmc
	plot(shifts, reshape(arms, length(shifts), 1), marker="x")

	xlabel("Shift")
	ylabel("Expectation of sampled arms")
	grid("on")
	legend(loc=1)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/shift/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, ".pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/shift/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, ".pdf"))
	end
	close(fig)
end
