include("/Users/xuedong/Programming/PhD/BestArm.jl/src/BestArm.jl")
using BestArm

dist = "Bernoulli"

mu = [0.5, 0.45, 0.425, 0.4, 0.375, 0.35, 0.325, 0.3, 0.275, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125]

budget = 500
mc = 20
policies = [succ_reject]
names = ["Successive Reject"]
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
	# println(recs)
end
