# Test scripts for different algos

# Random seed
#srand(4)

type_dist = "Bernoulli"
include("BAIBudget.jl")

mu = [0.3, 0.25, 0.2, 0.1]

budget = 10
mc = 20
policies = [UniformSampling, UCBE, UCBEAdaptive, SuccReject, UGapEB, UGapEBAdaptive, SeqHalvingNoRef, SeqHalvingRef]
names = ["Uniform Sampling", "UCB-E", "Adaptive UCB-E", "Successive Reject", "UGapEB", "Adaptive UGapEB", "Sequential Halving without Refresh", "Sequential Halving with Refresh"]
#policies = [SeqHalvingNoRef, SeqHalvingRef]
#names = ["No Ref", "Ref"]
lp = length(policies)

for imeth in 1:lp
	policy = policies[imeth]

	#recs = [nothing for imeth in 1:lp]
	#Ns = [nothing for imeth in 1:lp]
	#mean_list = [nothing for imeth in 1:lp]
	rec, N, means, recs = policy(mu, budget)
	println(names[imeth])
	println(rec)
	println(N)
	println(means)
	println(recs)
end

