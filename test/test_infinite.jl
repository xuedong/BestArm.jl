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
beta = 3.0
num = 64
budget = 1024
mcmc = 10

policies = [BestArm.seq_halving_infinite, BestArm.ttts_dynamic]
policy_names = ["Sequential Halving", "Dynamic TTTS"]
lp = length(policies)


# Options
VERBOSE = true


# Tests
fig = figure()
X = 1:budget
for imeth in 1:lp
	policy = policies[imeth]
	regrets = zeros(1, budget)
	if policy_names[imeth] == "TTTS"
	  	#regrets_array = @DArray [BestArm.parallel_ttts(mu, budget, dist) for i = 1:mcmc]
		#for i in 1:mcmc
		#	regrets += regrets_array[i]
		#end
		@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
			_, _, _, recs, mu = policy(reservoir, num, budget, dist, 0.5, true, alpha, beta)
			regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
			regrets += regrets_current
		end
	elseif policy_names[imeth] == "Dynamic TTTS"
		@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
			_, _, _, recs, mu = policy(reservoir, num, budget, dist, 0.5, true, alpha, beta)
			regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
			regrets += regrets_current
		end
	else
		@showprogress 1 string("Computing ", policy_names[imeth], "...") for k in 1:mcmc
			_, _, _, recs, mu = policy(reservoir, num, budget, dist, BestArm.eba, alpha, beta)
			regrets_current = BestArm.compute_regrets_reservoir(mu, recs, budget)
			regrets += regrets_current
		end
	end
	plot(X, reshape(regrets/mcmc, budget, 1), label = policy_names[imeth])
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
grid("on")
legend(loc=1)
if Sys.KERNEL == :Darwin
	savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/test/test_infinite.pdf"))
elseif Sys.KERNEL == :Linux
	savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/test_infinite.pdf"))
end
close(fig)
