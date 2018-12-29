function siri(reservoir::String, budget::Integer,
	delta::Real = 0.01, c::Real = 1.0, beta::Real = 2.0
	dist::String, rec::Function = mpa,
	theta1::Float64 = 1.0, theta2::Float64 = 1.0)
	K = compute_tbeta(budget)
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
		ucbs = [compute_ucb(means[a], alpha, N[a]) for a in 1:K]
		_, maxindx = findmax(ucbes)
		# Update
		for i in 1:N[maxindx]
			S[maxindx] += sample_arm(mu[maxindx], dist)
			N[maxindx] += 1
			recommendations[t+i] = rec(N, S)
		end
		means[maxindx] = S[maxindx]/N[maxindx]
		t += N[maxindx]
	end

	recommendations = Int.(recommendations)

	return (recommendations[budget], N, means, recommendations)
end
