function siri(reservoir::String, budget::Integer, dist::String,
	delta::Real = 0.01, c::Real = 1.0, beta::Real = 2.0,
	rec::Function = mpa, theta1::Float64 = 1.0, theta2::Float64 = 1.0)
	K = compute_tbeta(budget, beta)
	mu = [sample_reservoir(reservoir, theta1, theta2) for _ in 1:K]

	# Initialization
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	means = zeros(1, K)
	recommendations = zeros(1, budget)
	for a in 1:K
		N[a] = 1
		S[a] = sample_arm(mu[a], dist)
		means[a] = S[a]
		recommendations[a] = rand(1:K)
	end

	# Exploration
	t = K
	while t <= budget
		# Pick an arm based on the B-value
		ucbs = [compute_b_value_siri(means[a], delta, c, beta, N[a], K) for a in 1:K]
		_, maxindx = findmax(ucbs)
		# Update
		pulls = N[maxindx]
		for i in 1:pulls
			S[maxindx] += sample_arm(mu[maxindx], dist)
			N[maxindx] += 1
			if t+i > budget
				break
			else
				recommendations[t+i] = rec(N, S)
			end
		end
		means[maxindx] = S[maxindx]/N[maxindx]
		t += pulls
	end

	recommendations = Int.(recommendations)

	return (recommendations[budget], N, means, recommendations, mu)
end
