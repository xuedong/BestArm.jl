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
alpha = 1.0
beta = 1.0
num = 16
budget = 64
mcmc = 1000

policies = [BestArm.seq_halving_infinite, BestArm.ttts_infinite, BestArm.ttts_dynamic]
policy_names = ["ISHA", "TTTS", "Dynamic TTTS"]
lp = length(policies)


# Options
VERBOSE = true


# Tests
fig = figure()
X = 1:budget
for imeth in 1:lp
	policy = policies[imeth]
	if policy_names[imeth] == "TTTS"
	  	#regrets_array = @DArray [BestArm.parallel_ttts(mu, budget, dist) for i = 1:mcmc]
		#for i in 1:mcmc
		#	regrets += regrets_array[i]
		#end
		for i in 2:4
			regrets = zeros(1, budget)
			@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
				_, _, _, recs, mu = policy(reservoir, Int(2^i), budget, dist, 0.5, true, alpha, beta)
				regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
				regrets += regrets_current
			end
			plot(X, reshape(regrets/mcmc, budget, 1), linestyle="--", label=string(policy_names[imeth],2^i))
		end
	elseif policy_names[imeth] == "Dynamic TTTS"
		regrets = zeros(1, budget)
		@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
			_, _, recs, mu = policy(reservoir, 1, num, budget, dist, 0.5, alpha, beta)
			regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
			regrets += regrets_current
		end
		plot(X, reshape(regrets/mcmc, budget, 1), label = policy_names[imeth])
	else
		for i in 2:4
			regrets = zeros(1, budget)
			@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
				_, _, _, recs, mu = policy(reservoir, Int(2^i), budget, dist, BestArm.eba, alpha, beta)
				regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
				regrets += regrets_current
			end
			plot(X, reshape(regrets/mcmc, budget, 1), label=string(policy_names[imeth],2^i))
		end
	end
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
grid("on")
legend(loc=1)
if Sys.KERNEL == :Darwin
	savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/test/", reservoir, "(", alpha, ",", beta, ")", "_", budget, ".pdf"))
elseif Sys.KERNEL == :Linux
	savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/", reservoir, "(", alpha, ",", beta, ")", "_", budget, ".pdf"))
end
close(fig)
