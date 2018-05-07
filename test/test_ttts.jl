using PyPlot
using ConfParser
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
if Sys.KERNEL == :Darwin
	conf = ConfParse("/Users/xuedong/Programming/PhD/BestArm.jl/test/configs.ini")
elseif Sys.KERNEL == :Linux
	conf = ConfParse("/home/xuedong/Documents/xuedong/phd/work/code/BestArm.jl/test/configs.ini")
end
parse_conf!(conf)

settings = ["setting0", "setting1", "setting2", "setting3", "setting4", "setting5", "setting6", "setting7", "setting8"]
policies = [uniform, ucbe, succ_reject, ugape_b, seq_halving_ref, ttts, ts, at_lucb]
names = ["Uniform Sampling", "UCB-E", "Successive Reject", "UGapEB", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Thompson Sampling", "AT-LUCB"]
lp = length(policies)


# Tests
for setting in settings
	# load experimental setting from .ini file
	dist = retrieve(conf, setting, "distribution")
	dist = String(dist)
	mu = retrieve(conf, setting, "mu")
	mu = map(x -> (v = tryparse(Float64,x); isnull(v) ? 0.0 : get(v)), mu)
	budget = retrieve(conf, setting, "budget")
	budget = parse(Int, budget)
	mcmc = retrieve(conf, setting, "mcmc")
	mcmc = parse(Int, mcmc)

	fig = figure()
	X = 1:budget

	# running tests
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
		subplot(211)
		plot(X, transpose(regrets/mcmc), label = names[imeth])
		subplot(212)
		plot(log10.(X), -log10.(transpose(regrets/mcmc) ./ X), label = names[imeth])
	end

	xlabel("Allocation budget")
	ylabel("Expectation of the simple regret")
	grid("on")
	legend(loc=1)
	savefig(string("/Users/xuedong/Programming/PhD/BestArm.jl/results/ttts/", setting, ".png"))
	close(fig)
end
