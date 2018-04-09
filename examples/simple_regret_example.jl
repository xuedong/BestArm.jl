using PyPlot
using BestArm

# Problem setting
dist = "Bernoulli"

mu = ones(1, 20)
mu /= 10
mu[19] = 0.2
mu[20] = 0.25
#println(mu)

<<<<<<< HEAD
budget = 200
mcmc = 100

#policies = [ttps]
#names = ["Top-Two Probability Sampling"]
policies = [uniform, ucbe, succ_reject, ugape_b, seq_halving_ref, seq_halving_no_ref, ttts, ttps]
names = ["Uniform Sampling", "UCB-E", "Successive Reject", "UGapEB", "Sequential Halving without Refresh", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Top-Two Probability Sampling"]
=======
budget = 2000
mcmc = 10

#policies = [ttps]
#names = ["Top-Two Probability Sampling"]
policies = [ucbe, succ_reject, ugape_b, seq_halving_ref, seq_halving_no_ref, ttts, ttps]
names = ["UCB-E", "Successive Reject", "UGapEB", "Sequential Halving without Refresh", "Sequential Halving with Refresh", "Top-Two Thompson Sampling", "Top-Two Probability Sampling"]
>>>>>>> 13c8562ada0b7d55a90a95258f02e6abd10fe259
lp = length(policies)

# Options
VERBOSE = true

# Tests
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
	#println(regrets/mcmc)
	#if imeth == 4 || imeth == 7 || imeth == 8
	#	plot(X, transpose(regrets/mcmc), linestyle="-.", label=names[imeth])
	#else
	plot(X, transpose(regrets/mcmc), label=names[imeth])
	#end
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
ax = axes()
grid("on")
legend(bbox_to_anchor=[1.05, 1], loc=2, borderaxespad=0)
ax[:set_position]([0.06, 0.06, 0.71, 0.91])
show()
