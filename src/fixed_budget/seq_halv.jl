function seq_halving_no_ref(mu::Array, budget::Integer, dist::String, rec::Function = eba)
	K = length(mu)
	rounds = ceil(log2(K))

	# Initialization
	arms = [i for i in 1:K]
	N = Int.(zeros(1, K))
	S = zeros(1, K)
	rewards = [[] for i in 1:K]
	s_r = K
	recommendations = zeros(1, budget)

	for a in 1:K
		N[a] = 1
		S[a] = sample_arm(mu[a], dist)
	end

	# Elimination loop
	j = 0
	for r in 1:rounds
		t_r = tr(budget, K, r)
		for i in 1:t_r
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
function seq_halving_ref(mu::Array, budget::Integer, dist::String, rec::Function = eba)
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

	for a in 1:K
		N[a] = 1
		S[a] = sample_arm(mu[a], dist)
	end

	# Elimination loop
	j = 0
	for r in 1:rounds
		t_r = tr(budget, K, r)
		for i in 1:t_r
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


# Sequential Halving for Infinitely-Many Armed Bandits
function seq_halving_infinite(reservoir::String, num::Integer,
	budget::Integer, dist::String, rec::Function = eba,
	alpha::Float64 = 1.0, beta::Float64 = 1.0)
	mu = [sample_reservoir(reservoir, alpha, beta) for _ in 1:num]
	rounds = ceil(log2(num))

	# Initialization
	arms = [i for i in 1:num]
	N = Int.(zeros(1, num))
	S = zeros(1, num)
	rewards = [[] for i in 1:num]
	s_r = num
	recommendations = zeros(1, budget)

	for a in 1:num
		N[a] = 1
		S[a] = sample_arm(mu[a], dist)
	end

	# Elimination loop
	j = 0
	for r in 1:rounds
		t_r = tr(budget, num, r)
		for i in 1:t_r
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

	means = [mean(rewards[i]) for i in 1:num]
	recommendation = rand(arms)
	recommendations = Int.(recommendations)

	return(recommendation, N, means, recommendations)
end
