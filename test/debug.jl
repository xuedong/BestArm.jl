# Test scripts for different algos

# Random seed
#srand(4)

using BestArm

dist = "Bernoulli"

mu = [0.25, 0.9, 0.2, 0.1]

budget = 100
mc = 20
policies = [ttts]
names = ["Thompson Sampling"]
lp = length(policies)

for imeth in 1:lp
	policy = policies[imeth]

	#recs = [nothing for imeth in 1:lp]
	#Ns = [nothing for imeth in 1:lp]
	#mean_list = [nothing for imeth in 1:lp]
	rec, N, means, recs = policy(mu, budget, dist)
	println(names[imeth])
	println(rec)
	println(N)
	println(means)
	println(recs)
end
