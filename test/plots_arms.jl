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
alphas = [0.5, 1.0, 2.0, 3.0, 1.0]
betas = [0.5, 1.0, 2.0, 1.0, 3.0]
num = 16
budget = 64
mcmc = 10
maxmu = 1.0
default = true
limit = budget

policies = [BestArm.ttts_dynamic]
policy_names = ["Dynamic TTTS"]
abrevs = ["dttts"]
lp = length(policies)


# Options
VERBOSE = true
SAVE = false


# Tests
for iparam in 1:5
	X = 1:budget
	for imeth in 1:lp
		policy = policies[imeth]
		if policy_names[imeth] == "Dynamic TTTS"
			if default
				regrets = zeros(1, budget)
				num_arms = zeros(1, budget+1)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					fig = figure()
					_, N, recs, mu = policy(reservoir, 1, limit, budget, dist, 0.5, false, alphas[iparam], betas[iparam], false)
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget, maxmu)
					regrets += regrets_current
					copy = N .- 1
					if SAVE
						h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/arms/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], ".h5"), "w") do file
							write(file, abrevs[imeth], N, mu)
						end
					end

					bar(mu, N, width=0.01, color="#0f87bf", align="center", alpha=0.5)

					Seaborn.set()
				    x = rand(Beta(alphas[iparam], betas[iparam]), 10000)
				    Seaborn.distplot(x, his	t=false)

					xlabel("Arm means")
					ylabel("Number of pulls distribution")
					grid("on")
					legend(loc=1)
					if Sys.KERNEL == :Darwin
						savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/arms/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, "_", k, ".pdf"))
					elseif Sys.KERNEL == :Linux
						savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/arms/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, "_", k, ".pdf"))
					end
					close(fig)
				end
				# if Sys.KERNEL == :Darwin
				# 	h5open(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/log/arms/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_N.h5"), "w") do file
				# 		write(file, abrevs[imeth], num_arms)
				# 	end
				# elseif Sys.KERNEL == :Linux
				# 	h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/arms/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], "_N.h5"), "w") do file
				# 		write(file, abrevs[imeth], num_arms)
				# 	end
				# end
			end
		end
	end
end
