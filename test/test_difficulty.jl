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
reservoir = "Beta"
dist = "Bernoulli"
alphas = [1.0, 1.0, 1.0, 1.0, 1.0]
betas = [1.0, 2.0, 3.0, 4.0, 5.0]
# alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
# betas = [1.0, 1.0, 1.0, 1.0, 1.0]
# num = 16
budget = 160
mcmc = 100
len = length(alphas)
limit = budget

policy = BestArm.ttts_dynamic
policy_name = "Dynamic TTTS"
abrev = "dttts"


# Tests
for iparam in 1:len
	# fig = figure()
	arms = zeros(1, 10)
	@showprogress 1 string("Computing ", policy_name, "...") for k in 1:mcmc
		rec, N, recs, mu = policy(reservoir, 1, limit, budget, dist, 0.5, false, alphas[iparam], betas[iparam], true)
		# regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
		# regrets += regrets_current
		for i in 1:9
			arms[i] += length(filter(x -> x==i, N))
		end
		arms[10] += length(filter(x -> x>=10, N))
	end

	if Sys.KERNEL == :Darwin
		h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/difficulty/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrev, ".h5"), "w") do file
			write(file, abrev, arms)
		end
	elseif Sys.KERNEL == :Linux
		h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/difficulty/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrev, ".h5"), "w") do file
			write(file, abrev, arms)
		end
	end
end
