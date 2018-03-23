# Algorithms for Best Arm Identification in the Fixed Budget Setting

# All the algorithms take at least the following arguments:
# mu: vector of means
# budget: as it is, typically an integer
# rec: the recommendation strategy, to be choosen from EBA, EBP and MPA

using Distributions
using PyPlot

include("Arms.jl")
include("Utils.jl")

# Uniform Sampling
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

# UCB-E
# The core problem here is the tuning of alpha,
# which highly depends on the complexity measure H1,
# which has no reason to be known beforehand.
function UCBE(mu, budget, rec=eba, alpha=1)
	K = length(mu)

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
		# Pick an arm based on the B-value
		ucbes = [compute_ucbe(means[a], alpha, N[a]) for a in 1:K]
		_, maxindx = findmax(ucbes)
		# Update
		S[maxindx] += sample_arm(mu[maxindx], type_dist)
		N[maxindx] += 1
		means[maxindx] = S[maxindx]/N[maxindx]
		recommendations[t] = rec(N, S)
	end

	recommendations = Int.(recommendations)

	return (recommendations[budget], N, means, recommendations)
end

# Adaptive UCB-E
function UCBEAdaptive(mu, budget, rec=eba, c=1)
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
			new_sample = sample_arm(mu[maxindx], type_dist)
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

# Successive Reject
function SuccReject(mu, budget, rec=eba)
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
				new_sample = sample_arm(mu[a], type_dist)
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

# Fixed budget UGapE
# The same issue as UCB-E, the tuning of alpha
# depends highly on a complexity measure H_epsilon,
# which is not very accessible in practice.
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

# Sequential Halving without Refresh
# Two versions: Refresh or without refresh
# Question: Is without refresh theoretically promising?
function SeqHalvingNoRef(mu, budget, rec=eba)
	K = length(mu)
	rounds = ceil(log2(K))

	# Initialization
	arms = [i for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	rewards = [[] for i in 1:K]
	s_r = K
	recommendations = zeros(1, budget)

	# Elimination loop
	j = 0
	for r in 1:rounds
		for a in arms
			t_r = tr(budget, K, r)
			for i in 1:t_r
				new_sample = sample_arm(mu[a], type_dist)
				append!(rewards[a], new_sample)
				S[a] += new_sample
				N[a] += 1
				j += 1
				recommendations[j] = arms[rec(N[arms], S[arms])]
			end
		end
		for i in 1:(s_r-ceil(s_r/2))
			_, minindx = findmin([mean(rewards[a]) for a in arms])
			deleteat!(arms, minindx)
		end
		s_r = ceil(s_r/2)
	end
	
	for t in (j+1):budget
		recommendations[t] = rand(arms)
	end

	means = [mean(rewards[i]) for i in 1:K]
	recommendation = rand(arms)
	recommendations = Int.(recommendations)

	return(recommendation, N, means, recommendations)
end

# Sequential Halving with Refresh
function SeqHalvingRef(mu, budget, rec=eba)
	K = length(mu)
	rounds = ceil(log2(K))

	# Initialization
	arms = [i for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	rewards = [[] for i in 1:K]
	s_r = K
	total_picks = 0
	recommendations = zeros(1, budget)

	# Elimination loop
	j = 0
	for r in 1:rounds
		t_r = tr(budget, K, r)
		for a in arms
			for i in 1:t_r
				new_sample = sample_arm(mu[a], type_dist)
				append!(rewards[a], new_sample)
				S[a] += new_sample
				N[a] += 1
				j += 1
				recommendations[j] = arms[rec(N[arms], S[arms])]
			end
		end
		for i in 1:(s_r-ceil(s_r/2))
			_, minindx = findmin([mean(rewards[a][Int(total_picks+1):Int(total_picks+t_r)]) for a in arms])
			deleteat!(arms, minindx)
			#println(arms)
		end
		total_picks += t_r
		s_r = ceil(s_r/2)
	end

	for t in (j+1):budget
		recommendations[t] = rand(arms)
	end

	means = [mean(rewards[i]) for i in 1:K]
	recommendation = rand(arms)
	recommendations = Int.(recommendations)

	return(recommendation, N, means, recommendations)
end
