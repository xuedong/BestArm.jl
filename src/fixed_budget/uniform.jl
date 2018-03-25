function UniformSampling(mu, budget, rec=eba)
	K = length(mu)

	if (budget % K != 0 || budget < 0)
		ArgumentError("Budget must be multiples of number of arms!")
	end

	# Initialization
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	means = zeros(1, K)
	recommendations = zeros(1, budget)
	for a in 1:K
		N[a] = 1
		S[a] = sample_arm(mu[a], type_dist)
		means[a] = S[a]
		recommendations[a] = rec(N, S)
	end

	# Exploration
	for t in (K+1):budget
		# Pick an arm uniformly
		a = t % K
		if (a == 0)
			a = K
		end
		# Update
		S[a] += sample_arm(mu[a], type_dist)
		N[a] += 1
		means[a] = S[a]/N[a]
		recommendations[t] = rec(N, S)
	end

	recommendations = Int.(recommendations)

	return (recommendations[budget], N, means, recommendations)
end
