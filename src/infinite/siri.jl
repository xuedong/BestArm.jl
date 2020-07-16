function siri(reservoir::String, budget::Integer, dist::String,
	delta::Real = 0.01, c::Real = 1.0, beta::Real = 2.0,
	rec::Function = mpa, theta1::Float64 = 1.0, theta2::Float64 = 1.0,
	final::Bool = true)
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
		if final == false
			recommendations[a] = rand(1:K)
		end
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
				if final == false
					recommendations[t+i] = rec(N, S)
				end
			end
		end
		means[maxindx] = S[maxindx]/N[maxindx]
		t += pulls
	end

	recommendations = Int.(recommendations)
	if final
		recommendation = rec(N, S)
	else
		recommendation = recommendations[budget]
	end

	return (recommendation, N, means, recommendations, mu)
end


# Helper functions for SiRI
function compute_tbeta(n::Integer, beta::Real, a::Real = 0.3)
	b = min(beta, 2)
	if beta < 2
		a_n = a
	elseif beta == 2
		a_n = a / (log2(n))^2
	else
		a_n = a / log2(n)
	end

	tbeta = Int(ceil(a_n * n^(b/2)))
	return tbeta
end


function compute_b_value_siri(mean::Real, delta::Real, c::Real, beta::Real,
	num_pulls::Integer, num_arms::Integer)
	b = min(beta, 2)
	tbeta = floor(log2(num_arms))
	index = c / num_pulls * log(2^(2*tbeta/b)/(num_pulls*delta))

	return mean + 2*sqrt(index) + 2*index
end
