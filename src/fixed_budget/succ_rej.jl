function succ_reject(mu::Array{Float64,2}, budget::Integer, dist::String, rec::Function = eba)
	K = length(mu)
	log_bar = compute_log_bar(K)

	# Initialization
	arms = [i for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	rewards = [[] for i in 1:K]
	recommendations = zeros(1, budget)
	s_k = 0

	# Elimination loop
	j = 0
	for k in 1:(K-1)
		n_k = compute_nk(budget, K, k, log_bar) - compute_nk(budget, K, k-1, log_bar)
		#println(n_k)
		#j = 0
		for a in arms
			for i in 1:n_k
				new_sample = sample_arm(mu[a], dist)
				append!(rewards[a], new_sample)
				S[a] += new_sample
				N[a] += 1
				j += 1
				recommendations[j] = arms[rec(N[arms], S[arms])]
			end
			#j += 1
		end
		s_k += (K-k+1) * n_k
		_, minindx = findmin([mean(rewards[a]) for a in arms])
		deleteat!(arms, minindx)
	end

	for t in (j+1):budget
		recommendations[t] = arms[1]
	end

	means = [mean(rewards[i]) for i in 1:K]
	recommendations = Int.(recommendations)
	#println(arms)

	return (arms[1], N, means, recommendations)
end
