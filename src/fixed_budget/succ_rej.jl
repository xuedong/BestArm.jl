function succ_reject(mu::Array, budget::Integer, dist::String, rec::Function = eba)
	K = length(mu)
	log_bar = compute_log_bar(K)

	# Initialization
	arms = [i for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	rewards = [[] for i in 1:K]
	recommendations = zeros(1, budget)
	s_k = 0

	for a in 1:K
		N[a] = 1
		S[a] = sample_arm(mu[a], dist)
	end

	# Elimination loop
	j = 0
	for k in 1:(K-1)
		n_k = compute_nk(budget, K, k, log_bar) - compute_nk(budget, K, k-1, log_bar)
		#println(n_k)
		#j = 0
		for i in 1:n_k
			for a in arms
				new_sample = sample_arm(mu[a], dist)
				append!(rewards[a], new_sample)
				S[a] += new_sample
				N[a] += 1
				j += 1
				if j > budget
					continue
				else
					recommendations[j] = arms[rec(N[arms], S[arms])]
				end
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


# Helper functions for Successive Reject
function compute_nk(n, K, k, log_bar)
	if k == 0
		return 0
	else
		return ceil((n-K)/(log_bar*(K+1-k)))
	end
end


# function sum_nk(n, K, k, log_bar)
# 	s = 0
# 	if k == 1
# 		return s
# 	else
# 		for i in 1:(k-1)
# 			n_k = compute_nk(n, K, i+1, log_bar) - compute_nk(n, K, i, log_bar)
# 			s += (K-i+1) * n_k
# 		end
# 	end
# 	return s
# end
