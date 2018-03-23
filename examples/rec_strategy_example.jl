using PyPlot

include("../BAIBudget.jl")
include("../Utils.jl")

# Problem setting
type_dist = "Bernoulli"

mu = ones(1, 20)
mu /= 10
mu[19] = 0.5
mu[20] = 0.9
#println(mu)

budget = 200
mcmc = 1000

policy = UCBE
strats = [eba, edp, mpa]
names = ["Empirical best arm", "Empirical distribution of plays", "Most played arm"]
ls = length(strats)

# Options
VERBOSE = true

# Tests
X = 1:budget
for istrat in 1:ls
	regrets = zeros(1, budget)
	for k in 1:mcmc
		_, _, _, recs = policy(mu, budget, strats[istrat])
		regrets_current = compute_regrets(mu, recs, budget)
		regrets += regrets_current
		if VERBOSE
			if k % 10 == 0
				println(k*100/mcmc, "%")
			end
		end
	end

	plot(X, transpose(regrets/mcmc), label=names[istrat])
end

xlabel("Allocation budget")
ylabel("Expectation of the simple regret")
title("UCB-E with different recommendation strategy")
ax = axes()
grid("on")
legend(bbox_to_anchor=[1.05,1], loc=2, borderaxespad=0)
ax[:set_position]([0.06,0.06,0.71,0.91])
show()
