using PyPlot
using ProgressMeter
using Distributed
using Seaborn
# using HDF5

# addprocs(3)
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
# @everywhere using BestArm
# @everywhere using DistributedArrays

# Problem setting
reservoir = "Beta"
dist = "Bernoulli"
alphas = [1.0, 3.0, 0.5, 1.0, 2.0]
betas = [1.0, 1.0, 0.5, 3.0, 2.0]
# alphas = [1.0, 3.0, 1.0, 0.5, 2.0, 5.0, 2.0, 0.3]
# betas = [1.0, 1.0, 3.0, 0.5, 5.0, 2.0, 2.0, 0.7]
mcmc = 10
default = true
step = 4
lb = 64
ub = 128
budgets = [Int(round(step*i*log2(step*i))) for i in (lb/step):(ub/step)]
narms = [step*i for i in (lb/step):(ub/step)]
lbudget = length(budgets)

policies = [BestArm.seq_halving_infinite, BestArm.ttts_infinite, BestArm.ttts_dynamic]
policy_names = ["ISHA", "TTTS", "Dynamic TTTS"]
abrevs = ["isha", "ttts", "dttts", "siri"]
lp = length(policies)
lparam = length(alphas)


# Options
VERBOSE = true
SAVE = true


# Tests
for iparam in 1:lparam
	fig = figure()
	Seaborn.set(style="darkgrid")
	for imeth in 1:lp
		policy = policies[imeth]
		if policy_names[imeth] == "TTTS"
		  	#regrets_array = @DArray [BestArm.parallel_ttts(mu, budget, dist) for i = 1:mcmc]
			#for i in 1:mcmc
			#	regrets += regrets_array[i]
			#end
			regrets = zeros(1, lbudget)
			for n in 1:lbudget
				num = narms[n]
				budget = budgets[n]
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					rec, _, _, recs, mu = policy(reservoir, num, budget, dist, 0.5, false, alphas[iparam], betas[iparam])
					regret_current = 1 - mu[rec]
					regrets[n] += regret_current
				end
			end
			plot(budgets, reshape(regrets/mcmc, lbudget, 1), marker="o", label=string(policy_names[imeth]))
			if SAVE
				h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/final/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], ".h5"), "w") do file
					write(file, abrevs[imeth], regrets)
				end
			end
		elseif policy_names[imeth] == "Dynamic TTTS"
			regrets = zeros(1, lbudget)
			for n in 1:lbudget
				num = narms[n]
				budget = budgets[n]
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					rec, _, _, mu = policy(reservoir, 1, num, budget, dist, 0.5, false, alphas[iparam], betas[iparam])
					regret_current = 1 - mu[rec]
					regrets[n] += regret_current
				end
			end
			plot(budgets, reshape(regrets/mcmc, lbudget, 1), marker="*", label=policy_names[imeth])
			if SAVE
				h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/final/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], ".h5"), "w") do file
					write(file, abrevs[imeth], regrets)
				end
			end
		elseif policy_names[imeth] == "SiRI"
			regrets = zeros(1, lbudget)
			for n in 1:lbudget
				num = narms[n]
				budget = budgets[n]
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					rec, _, _, _, mu = policy(reservoir, budget, dist, 0.01, 1.0, 1.0, BestArm.mpa, alphas[iparam], betas[iparam])
					regret_current = 1 - mu[rec]
					regrets[n] += regret_current
				end
			end
			plot(budgets, reshape(regrets/mcmc, lbudget, 1), marker="^", label=string(policy_names[imeth], 1.0))
			if SAVE
				h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/final/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], ".h5"), "w") do file
					write(file, abrevs[imeth], regrets)
				end
			end
		else
			regrets = zeros(1, lbudget)
			for n in 1:lbudget
				num = narms[n]
				budget = budgets[n]
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					rec, _, _, _, mu = policy(reservoir, num, budget, dist, BestArm.eba, alphas[iparam], betas[iparam])
					regret_current = 1 - mu[rec]
					regrets[n] += regret_current
				end
			end
			plot(budgets, reshape(regrets/mcmc, lbudget, 1), marker="s", label=string(policy_names[imeth]))
			if SAVE
				h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/final/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", abrevs[imeth], ".h5"), "w") do file
					write(file, abrevs[imeth], regrets)
				end
			end
		end
	end

	xlabel("Allocation budget")
	ylabel("Expectation of the simple regret")
	grid("on")
	legend(loc=1)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ").pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ").pdf"))
	end
	close(fig)
end
