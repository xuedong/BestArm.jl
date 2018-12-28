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
alpha = 3.0
beta = 1.0
num = 4
budget = 100
mcmc = 1

policies = [BestArm.seq_halving_infinite]
policy_names = ["Sequential Halving"]
# policies = [BestArm.uniform, BestArm.succ_reject, BestArm.ugape_b, BestArm.seq_halving_ref, BestArm.ttts, BestArm.ts, BestArm.at_lucb]
# policy_names = ["Uniform Sampling", "Successive Reject", "UGapEB", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Thompson Sampling", "AT-LUCB"]
lp = length(policies)


# Options
VERBOSE = true


# Tests
fig = figure()
X = 1:budget
for imeth in 1:lp
	policy = policies[imeth]
	regrets = zeros(1, budget)
	if policy_names[imeth] == "Top-Two Thompson Sampling"
		regrets_array = @DArray [BestArm.parallel_ttts(mu, budget, dist) for i = 1:mcmc]
		for i in 1:mcmc
			regrets += regrets_array[i]
		end
	else
		@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
			_, _, _, recs = policy(reservoir, num, budget, dist, BestArm.eba, alpha, beta)
			regrets_current = BestArm.compute_regrets(mu, recs, budget)
			regrets += regrets_current
		end
	end
	plot(X, reshape(regrets/mcmc, budget, 1), label = policy_names[imeth])
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
grid("on")
legend(loc=1)
