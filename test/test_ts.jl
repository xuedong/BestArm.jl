using PyPlot
include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
using BestArm
using ConfParser


# Problem setting
conf = ConfParse("/Users/xuedong/Programming/PhD/BestArm.jl/test/configs.ini")
parse_conf!(conf)

dist = retrieve(conf, "problem0", "distribution")
dist = String(dist)
mu = retrieve(conf, "problem0", "mu")
mu = map(x -> (v = tryparse(Float64,x); isnull(v) ? 0.0 : get(v)), mu)
budget = retrieve(conf, "problem0", "budget")
budget = parse(Int, budget)
mcmc = retrieve(conf, "problem0", "mcmc")
mcmc = parse(Int, mcmc)

policies = [uniform, ucbe, succ_reject, ugape_b, seq_halving_ref, ts, at_lucb]
names = ["Uniform Sampling", "UCB-E", "Successive Reject", "UGapEB", "Sequential Halving with Refresh", "Thompson Sampling", "AT-LUCB"]
lp = length(policies)


# Options
VERBOSE = true


# Tests
fig = figure()
X = 1:budget
for imeth in 1:lp
	policy = policies[imeth]
	regrets = zeros(1, budget)
	for k in 1:mcmc
		_, _, _, recs = policy(mu, budget, dist)
		regrets_current = compute_regrets(mu, recs, budget)
		regrets += regrets_current
		if VERBOSE
			println(k*100/mcmc, "%")
		end
	end
	plot(X, transpose(regrets/mcmc), label = names[imeth])
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
grid("on")
legend(loc=1)
