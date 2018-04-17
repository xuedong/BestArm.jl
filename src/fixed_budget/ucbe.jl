function ucbe(mu::Array, budget::Integer, dist::String, rec::Function = eba, alpha::Real = 1)
	K = length(mu)

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
	for t in (K+1):budget
		# Pick an arm based on the B-value
		ucbes = [compute_ucbe(means[a], alpha, N[a]) for a in 1:K]
		_, maxindx = findmax(ucbes)
		# Update
		S[maxindx] += sample_arm(mu[maxindx], dist)
		N[maxindx] += 1
		means[maxindx] = S[maxindx]/N[maxindx]
		recommendations[t] = rec(N, S)
	end

	recommendations = Int.(recommendations)

	return (recommendations[budget], N, means, recommendations)
end


# Adaptive UCB-E
function ucbe_adaptive(mu::Array, budget::Integer, dist::String, rec::Function = eba, c::Real = 1)
	K = length(mu)
	log_bar = compute_log_bar(K)

	# Initialization
	rewards = [[] for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	recommendations = zeros(1, budget)

	# Exploration
	for k in 0:(K-1)
		if k == 0
			H_k = K
		else
			means = [(length(rewards[i])>0)?(mean(rewards[i])):0 for i in 1:K]
			max_mean, _ = findmax(means)
			gaps = [max_mean - means[i] for i in 1:K]
			sorted_gaps = sort(gaps)
			H_k, _ = findmax([i/(sorted_gaps[i]^2) for i in (K-k+1):K])
		end

		start_point = compute_tk(budget, K, k, log_bar) + 1
		end_point = compute_tk(budget, K, k+1, log_bar)
		for t in start_point:end_point
			means = [(length(rewards[i])>0)?(mean(rewards[i])):0 for i in 1:K]
			ucbes = [compute_ucbe(means[i], c*budget/H_k, N[i]) for i in 1:K]
			# Pick an arm based on the B-value
			_, maxindx = findmax(ucbes)
			# Update
			new_sample = sample_arm(mu[maxindx], dist)
			#println(new_sample)
			append!(rewards[maxindx], new_sample)
			S[maxindx] += new_sample
			N[maxindx] += 1
			recommendations[t] = rec(N, S)
		end
	end

	means = [mean(rewards[i]) for i in 1:K]
	recommendations = Int.(recommendations)

	return (recommendations[budget], N, means, recommendations)
end
