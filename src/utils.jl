# Compute simple regrets
function compute_regrets(mu::AbstractArray{<:Real}, recommendations::AbstractArray{<:Integer}, budget::Integer)
	maxmu = argmax(mu)[1]
	regrets = zeros(1, budget)
	for i in 1:budget
		regrets[i] = maxmu - mu[recommendations[i]]
	end

	return regrets
end


# Compute simple regrets for a normalized reservoir
function compute_regrets_reservoir(mu::AbstractArray{<:Real}, recommendations::AbstractArray{<:Integer}, budget::Integer, maxmu::Float64 = 0.25)
	regrets = zeros(1, budget)
	for i in 1:budget
		regrets[i] = maxmu - mu[recommendations[i]]
	end

	return regrets
end


# Recommendation strategies
# EDP: Empirical distribution of plays
function edp(N, S)
	total = sum(N)
	K = length(N)
	p = [N[i]/total for i in 1:K]
	d = Categorical(p)
	return rand(d)
end


# EBA: Empirical best arm
function eba(N, S)
	K = length(N)
	means = S ./ N
	idx = (LinearIndices(means .== maximum(means)))[findall(means .== maximum(means))]
	best = idx[floor(Int, length(idx) * rand()) + 1]
	return best
end


# MPA: Most played arm
function mpa(N, S)
	idx = (LinearIndices(N .== maximum(N)))[findall(N .== maximum(N))]
	best = idx[floor(Int, length(idx) * rand()) + 1]
	return best
end


# Helper functions for UCB-E
function compute_ucbe(mean, a, s)
	if s > 0
		return mean + sqrt(a/s)
	else
		return 10^10
	end
end


function compute_log_bar(k)
	return 0.5 + sum([(1/i) for i in 2:k])
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


# Helper functions for Adaptive UCB-E
function compute_tk(n, K, k, log_bar)
	if k == 0
		tk = 0
	elseif k == 1
		tk = K * compute_nk(n, K, k, log_bar)
	elseif k == K
		tk = n
	else
		tk = sum([compute_nk(n, K, i, log_bar) for i in 1:(k-1)]) + (K-k+1) * compute_nk(n, K, k, log_bar)
	end
	return Int(floor(tk))
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


# Helper functions for Sequential Halving
function sr(n, r)
	s_r = n
	if r == 1
		return s_r
	else
		for i in 2:r
			s_r = ceil(s_r/2)
		end
		return s_r
	end
end


function tr(budget, n, r)
	s_r = sr(n, r)
	t_r = round(budget/(s_r*ceil(log2(n))))
	return t_r
end


# Helper function for AT-LUCB
function compute_deviation(n::Integer, u::Array, t::Integer, delta::Real, k1::Real = 1.25)
	return sqrt.(log.(k1*n*(t^4)/delta)./(2*u))
end


# Memory usage check
# function memuse()
#   return string(round(Int, parse(Int, readstring(`ps -p 29563 -o rss=`))/1024), "M")
# end


# Helper function for SiRI
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
