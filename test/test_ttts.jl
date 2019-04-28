using PyPlot
using ProgressMeter
using HDF5
using Distributed

addprocs(3)
if Sys.KERNEL == :Darwin
	@everywhere include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
elseif Sys.KERNEL == :Linux
	@everywhere include("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/src/BestArm.jl")
end
@everywhere using DistributedArrays

# settings = ["setting0", "setting1", "setting2", "setting3", "setting4", "setting5", "setting6", "setting7", "setting8"]
# policies = [uniform, ucbe, succ_reject, seq_halving_ref, ttts, ts, at_lucb]
# names = ["Uniform Sampling", "UCB-E", "Successive Reject", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Thompson Sampling", "AT-LUCB"]
# abrevs = ["uniform", "ucbe", "succ_reject", "seq_halving_ref", "ttts", "ts", "at_lucb"]
settings = ["setting0"]
dist = "Bernoulli"
mu = [0.5, 0.4, 0.35, 0.3]
budgets = [1000]
mcmcs = [1000]

policies = [BestArm.ttts]
policy_names = ["TTTS"]
abrevs = ["ttts"]
lp = length(policies)
SAVE = false


# Tests
for i in 1:length(settings)
	setting = settings[i]
	budget = budgets[i]
	mcmc = mcmcs[i]

	fig = figure()
	X = 1:budget

	# running tests
	for imeth in 1:lp
		policy = policies[imeth]
		if policy_names[imeth] == "TTTS"
			hits = zeros(1, budget)
			@showprogress 1 string("Computing ", names[imeth], "...") for k in 1:mcmc
				_, _, _, recs = policy(mu, budget, dist, 0.5, true, false)
				for j in 1:budget
					if recs[j] == 1
						hits[j] += 1
					end
				end
			end
			plot(X, reshape(1 .- hits/mcmc, budget, 1), linestyle="--", label=string(policy_names[imeth]))
			if SAVE
				h5open(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/log/ttts/", setting, ".h5"), "w") do file
			    	write(file, abrevs[imeth], hits)
				end
			end
		end

		# subplot(121)
		# plot(X, transpose(regrets/mcmc), label = names[imeth])
		# subplot(122)
		# plot(log10.(X), -log10.(transpose(regrets/mcmc) ./ X), label = names[imeth])
	end

	xlabel("Allocation budget")
	ylabel("Expectation of the simple regret")
	grid("on")
	legend(loc=2)
	if Sys.KERNEL == :Darwin
		savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/misc/results/ttts/", setting, ".pdf"))
	elseif Sys.KERNEL == :Linux
		savefig(string("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/misc/results/ttts/", setting, ".pdf"))
	end
	close(fig)
end
