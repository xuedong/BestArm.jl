using PyPlot
using ProgressMeter

addprocs(3)
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
@everywhere using BestArm
@everywhere using DistributedArrays

# Problem setting
dist = "Bernoulli"
mu = [0.5, 0.42, 0.4, 0.4, 0.35, 0.35]
budget = 600
mcmc = 10000

policies = [uniform, ucbe, succ_reject, ugape_b, seq_halving_ref, ttts, ttps, ts, at_lucb]
names = ["Uniform Sampling", "UCB-E", "Successive Reject", "UGapEB", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Top-Two Probability Sampling", "Thompson Sampling", "AT-LUCB"]
lp = length(policies)


# Options
VERBOSE = true


# Tests
fig = figure()
X = 1:budget
for imeth in 1:lp
	policy = policies[imeth]
	regrets = zeros(1, budget)
	if names[imeth] == "Top-Two Thompson Sampling"
		regrets_array = @DArray [parallel_ttts(mu, budget, dist) for i = 1:mcmc]
		for i in 1:mcmc
			regrets += regrets_array[i]
		end
	elseif names[imeth] == "Top-Two Probability Sampling"
		regrets_array = @DArray [parallel_ttps(mu, budget, dist) for i = 1:mcmc]
		for i in 1:mcmc
			regrets += regrets_array[i]
		end
	else
		@showprogress 1 string("Computing ", names[imeth], "...") for k in 1:mcmc
			_, _, _, recs = policy(mu, budget, dist)
			regrets_current = compute_regrets(mu, recs, budget)
			regrets += regrets_current
		end
	end
	plot(X, transpose(regrets/mcmc), label = names[imeth])
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
grid("on")
legend(loc=1)
