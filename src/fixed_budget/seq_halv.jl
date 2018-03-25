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
