using PyPlot

addprocs()
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
@everywhere using BestArm
@everywhere using DistributedArrays


# Problem setting
dist = "Bernoulli"
mu = [0.25, 0.3, 0.2, 0.1]
budget = 1000
mcmc = 1000

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
		for k in 1:mcmc
			_, _, _, recs = policy(mu, budget, dist)
			regrets_current = compute_regrets(mu, recs, budget)
			regrets += regrets_current
			if VERBOSE
				println(k*100/mcmc, "%")
			end
		end
	end
	#println(regrets/mcmc)
	#if imeth == 4 || imeth == 7 || imeth == 8
	#	plot(X, transpose(regrets/mcmc), linestyle="-.", label=names[imeth])
	#else
	plot(X, transpose(regrets/mcmc), label = names[imeth])
	#end
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
grid("on")
legend(loc=1)
savefig("results/exp_0.pdf")
close(fig)
