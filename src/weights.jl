function dico_solve(f, x_min, x_max, delta = 1e-11)
	# find m such that f(m)=0 using dichotomix search
	lower = x_min
	upper = x_max
	sgn = f(x_min)
	while (u - l) > delta
		m = (u+l)/2
		if f(m) * sgn > 0
			l = m
		else
			u = m
		end
	end
	m = (u+l)/2
	return m
end


# Computing the optimal weights
function big_i(alpha, mu1, mu2, dist)
	if (alpha == 0) | (alpha == 1)
		return 0
	else
		mid = alpha * mu1 + (1-alpha) * mu2
		return alpha * d(mu1, mid, dist) + (1-alpha) * d(mu2, mid, dist)
	end
end


muddle(mu1, mu2, nu1, nu2) = (nu1 * mu1 + nu2 * mu2)/(nu1 + nu2)


function cost(mu1, mu2, nu1, nu2, dist)
  	if (nu1 == 0) & (nu2 == 0)
     	return 0
  	else
     	alpha = nu1/(nu1+nu2)
     	return ((nu1 + nu2) * big_i(alpha, mu1, mu2, dist))
  	end
end


function xkofy(y, k, mu, dist, delta = 1e-11)
	# return x_k(y), i.e. finds x such that g_k(x)=y
	g(x) = (1+x) * cost(mu[1], mu[k], 1/(1+x), x/(1+x), dist) - y
	xMax = 1
	while g(xMax) < 0
		xMax = 2 * xMax
	end
	return dico_solve(x->g(x), 0, xMax, 1e-11)
end


function aux(y, mu, dist)
	# returns F_mu(y) - 1
	K = length(mu)
	x = [xkofy(y, k, mu, dist) for k in 2:K]
	m = [muddle(mu[1], mu[k], 1, x[k-1]) for k in 2:K]
	return (sum([d(mu[1], m[k-1], dist)/(d(mu[k], m[k-1], dist)) for k in 2:K])-1)
end


function one_step_opt(mu, dist, delta::Real = 1e-11)
	yMax = 0.5
	if d(mu[1], mu[2], dist) == Inf
		# find yMax such that aux(yMax, mu) > 0
		while aux(yMax, mu, dist) < 0
			yMax = yMax * 2
		end
	else
		yMax = d(mu[1] , mu[2], dist)
	end
	y = dico_solve(y -> aux(y, mu, dist), 0, yMax, delta)
	x = [xkofy(y, k, mu, dist, delta) for k in 2:length(mu)]
	pushfirst!(x, 1)
	nuOpt = x/sum(x)
	return nuOpt[1]*y, nuOpt
end


function optimal_weights(mu, dist, delta::Real = 1e-11)
	# returns T*(mu) and w*(mu)
	K = length(mu)
	IndMax = (LinearIndices(mu .== maximum(mu)))[findall(mu .== maximum(mu))]
	L = length(IndMax)
	if (L > 1)
		# multiple optimal arms
		vOpt = zeros(1,K)
		vOpt[IndMax] = 1/L
		return 0, vOpt
	else
		mu = vec(mu)
		index = sortperm(mu, rev = true)
		mu = mu[index]
		unsorted = vec(collect(1:K))
		invindex = zeros(Int, K)
		invindex[index] = unsorted
		# one-step optim
		vOpt, NuOpt = one_step_opt(mu, dist, delta)
		# back to good ordering
		nuOpt = NuOpt[invindex]
		NuOpt = zeros(1,K)
		NuOpt[1,:] = nuOpt
		return vOpt, NuOpt
	end
end


# Computing the parameterized optimal weights
function c_k(x, k, mu, dist, beta::Real = 0.5)
	average = mu[1]*beta/(beta+x) + mu[k]*x/(beta+x)
	return beta * d(mu[1], average, dist) + x * d(mu[k], average, dist)
end


function inverse(y, k, mu, dist, beta::Real = 0.5, delta::Real = 1e-11)
	# return x_k(y), i.e. finds x such that g_k(x)=y
	g(x) = c_k(x, k, mu, dist, beta, delta) - y
	x_max = 1
	while g(x_max) < 0
		x_max = 2 * x_max
	end
	return dico_solve(x->g(x), 0, x_max, 1e-11)
end


function optimal_weights_parameterized(mu, dist, delta::Real == 1e-11)
	K = length(mu)
