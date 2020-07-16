function ugape_b(mu::Array, budget::Integer, dist::String, alpha::Real = 1)
	K = length(mu)

	# Initialization
	rewards = [[] for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	recommendations = zeros(1, budget)
	# Pull each arm once
	for a in 1:K
		N[a] = 1
		new_sample = sample_arm(mu[a], dist)
		S[a] = new_sample
		append!(rewards[a], new_sample)
		recommendations[a] = rand(1:K)
	end

	# Exploration
	for t in (K+1):budget
		means = [mean(rewards[i]) for i in 1:K]
		betas = [compute_beta(length(rewards[i]), budget, K, alpha) for i in 1:K]
		ucbs = [means[i] + betas[i] for i in 1:K]
		lcbs = [means[i] - betas[i] for i in 1:K]

		maxucb, maxindx = findmax(ucbs)
		ucbs_copy = copy(ucbs)
		maxucbk, _ = findmax(deleteat!(ucbs_copy, maxindx))

		b_values = [compute_b_value(i, lcbs, maxindx, maxucb, maxucbk) for i in 1:K]

		_, Jt = findmin([b_values[i] for i in 1:K])
		indices_without_Jt = deleteat!([i for i in 1:K], Jt)
		_, maxindx_without_Jt = findmax([ucbs[i] for i in indices_without_Jt])
		ut = indices_without_Jt[maxindx_without_Jt]
		lt = Jt
		It = [lt, ut][findmax([betas[i] for i in [lt, ut]])[2]]

		# Update
		new_sample = sample_arm(mu[It], dist)
		append!(rewards[It], new_sample)
		S[It] += new_sample
		N[It] += 1
		recommendations[t] = findmin(b_values)[2]
	end

	means = [mean(rewards[i]) for i in 1:K]
	betas = [compute_beta(length(rewards[i]), budget, K, alpha) for i in 1:K]
	ucbs = [means[i] + betas[i] for i in 1:K]
	lcbs = [means[i] - betas[i] for i in 1:K]

	maxucb, maxindx = findmax(ucbs)
	ucbs_copy = copy(ucbs)
	maxucbk, _ = findmax(deleteat!(ucbs_copy, maxindx))

	b_values = [compute_b_value(i, lcbs, maxindx, maxucb, maxucbk) for i in 1:K]

	recommendation = findmin(b_values)[2]
	recommendations = Int.(recommendations)

	return (recommendation, N, means, recommendations)
end


# Adaptive UGapE
function ugape_b_adaptive(mu::Array, budget::Integer, dist::String, c::Real = 1)
	K = length(mu)

	# Initialization
	rewards = [[] for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	recommendations = zeros(1, budget)
	# Pull each arm once
	for a in 1:K
		N[a] = 1
		new_sample = sample_arm(mu[a], dist)
		S[a] = new_sample
		append!(rewards[a], new_sample)
		recommendations[a] = rand(1:K)
	end

	# Exploration
	for t in (K+1):budget
		means = [mean(rewards[i]) for i in 1:K]

		# Estimation of H
		gaps = [means[i] + sqrt(1/(2*N[i])) for i in 1:K]
		H_t = sum([1/gaps[i]^2 for i in 1:K])
		betas = [compute_beta(length(rewards[i]), budget, K, 1, c, H_t, true) for i in 1:K]

		ucbs = [means[i] + betas[i] for i in 1:K]
		lcbs = [means[i] - betas[i] for i in 1:K]

		maxucb, maxindx = findmax(ucbs)
		ucbs_copy = copy(ucbs)
		maxucbk, _ = findmax(deleteat!(ucbs_copy, maxindx))

		b_values = [compute_b_value(i, lcbs, maxindx, maxucb, maxucbk) for i in 1:K]

		_, Jt = findmin([b_values[i] for i in 1:K])
		indices_without_Jt = deleteat!([i for i in 1:K], Jt)
		_, maxindx_without_Jt = findmax([ucbs[i] for i in indices_without_Jt])
		ut = indices_without_Jt[maxindx_without_Jt]
		lt = Jt
		It = [lt, ut][findmax([betas[i] for i in [lt, ut]])[2]]

		# Update
		new_sample = sample_arm(mu[It], dist)
		append!(rewards[It], new_sample)
		S[It] += new_sample
		N[It] += 1
		recommendations[t] = findmin(b_values)[2]
	end

	means = [mean(rewards[i]) for i in 1:K]
	gaps = [means[i] + sqrt(1/(2*N[i])) for i in 1:K]
	H_t = sum([1/gaps[i]^2 for i in 1:K])
	betas = [compute_beta(length(rewards[i]), budget, K, 1, c, H_t, true) for i in 1:K]
	ucbs = [means[i] + betas[i] for i in 1:K]
	lcbs = [means[i] - betas[i] for i in 1:K]

	maxucb, maxindx = findmax(ucbs)
	ucbs_copy = copy(ucbs)
	maxucbk, _ = findmax(deleteat!(ucbs_copy, maxindx))

	b_values = [compute_b_value(i, lcbs, maxindx, maxucb, maxucbk) for i in 1:K]

	recommendation = findmin(b_values)[2]
	recommendations = Int.(recommendations)

	return (recommendation, N, means, recommendations)
end


# Helper functions for UGapE
function compute_beta(s, n, K, a, b=1, H=1, automatic=false)
	if automatic
		alpha = b * (n-K)/(4*H)
		return sqrt(alpha/s)
	else
		return sqrt(a*(n-K)/s)
	end
end


# function compute_ucb(i, rewards, n, a)
# 	K = length(rewards)
# 	s = length(rewards[i])
# 	ucb = mean(rewards[i]) + beta(s, n, K, a)
# 	return ucb
# end


# function compute_lcb(i, rewards, n, a)
# 	K = length(rewards)
# 	s = length(rewards[i])
# 	lcb = mean(rewards[i]) - beta(s, n, K, a)
# 	return lcb
# end


# function b_value(k, K, rewards, n, a)
# 	indices_without_k = deleteat!([i for i in 1:K], k)
# 	ucb_list = [compute_ucb(i, rewards, n, a) for i in indices_without_k]
# 	maxval, maxindx = findmax(ucb_list)
# 	return maxval - compute_lcb(k, rewards, n, a)
# end


function compute_b_value(k, lcbs, maxindx, maxucb, maxucbk)
	if k == maxindx
		return maxucbk - lcbs[k]
	else
		return maxucb - lcbs[k]
	end
end
