function UGapEB(mu, budget, alpha=1)
	K = length(mu)

	# Initialization
	rewards = [[] for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	recommendations = zeros(1, budget)
	# Pull each arm once
	for a in 1:K
		N[a] = 1
		new_sample = sample_arm(mu[a], type_dist)
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
		new_sample = sample_arm(mu[It], type_dist)
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
function UGapEBAdaptive(mu, budget, c=1)
	K = length(mu)

	# Initialization
	rewards = [[] for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	recommendations = zeros(1, budget)
	# Pull each arm once
	for a in 1:K
		N[a] = 1
		new_sample = sample_arm(mu[a], type_dist)
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
		new_sample = sample_arm(mu[It], type_dist)
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
