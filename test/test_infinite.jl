using PyPlot
using ProgressMeter
using Distributed

addprocs(3)
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
# @everywhere using BestArm
@everywhere using DistributedArrays

# Problem setting
reservoir = "Beta"
dist = "Bernoulli"
# alphas = [1.0]
# betas = [1.0]
alphas = [1.0, 3.0, 1.0, 0.5, 2.0, 5.0, 2.0, 1.0, 10.0]
betas = [1.0, 1.0, 3.0, 0.5, 5.0, 2.0, 2.0, 10.0, 1.0]
num = 16
budget = 64
mcmc = 1000

# policies = [BestArm.siri]
# policy_names = ["SiRI"]
policies = [BestArm.seq_halving_infinite, BestArm.ttts_infinite, BestArm.ttts_dynamic]
policy_names = ["ISHA", "TTTS", "Dynamic TTTS"]
lp = length(policies)


# Options
VERBOSE = true


# Tests
for iparam in 1:9
	fig = figure()
	X = 1:budget
	for imeth in 1:lp
		policy = policies[imeth]
		if policy_names[imeth] == "TTTS"
		  	#regrets_array = @DArray [BestArm.parallel_ttts(mu, budget, dist) for i = 1:mcmc]
			#for i in 1:mcmc
			#	regrets += regrets_array[i]
			#end
			for i in 2:5
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, _, _, recs, mu = policy(reservoir, Int(2^i), budget, dist, 0.5, true, alphas[iparam], betas[iparam])
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
					regrets += regrets_current
				end
				plot(X, reshape(regrets/mcmc, budget, 1), linestyle="--", label=string(policy_names[imeth], 2^i))
			end
		elseif policy_names[imeth] == "Dynamic TTTS"
			regrets = zeros(1, budget)
			@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
				_, _, recs, mu = policy(reservoir, 1, num, budget, dist, 0.5, true, alphas[iparam], betas[iparam])
				regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
				regrets += regrets_current
			end
			plot(X, reshape(regrets/mcmc, budget, 1), label=policy_names[imeth])
		elseif policy_names[imeth] == "SiRI"
			for beta in 1:1
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, _, _, recs, mu = policy(reservoir, budget, dist, 0.01, 1.0, beta, BestArm.mpa, alphas[iparam], betas[iparam])
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
					regrets += regrets_current
				end
				plot(X, reshape(regrets/mcmc, budget, 1), linestyle="-.", label=string(policy_names[imeth], beta))
			end
		else
			for i in 2:4
				regrets = zeros(1, budget)
				@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
					_, _, _, recs, mu = policy(reservoir, Int(2^i), budget, dist, BestArm.eba, alphas[iparam], betas[iparam])
					regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
					regrets += regrets_current
				end
				plot(X, reshape(regrets/mcmc, budget, 1), label=string(policy_names[imeth], 2^i))
			end
		end
	end

	xlabel("Allocation budget")
	ylabel("Expectation of the simple regret")
	grid("on")
	legend(loc=1)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, ".pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/", reservoir, "(", alphas[iparam], ",", betas[iparam], ")", "_", budget, "_mpa.pdf"))
	end
	close(fig)
end
