using Distributions

include("KLFunctions.jl")

function d(p, q, type_dist)
	if type_dist == "Bernoulli"
		return dBernoulli(p, q)
	elseif type_dist == "Poisson"
		return dPoisson(p, q)
	elseif type_dist == "Exponential"
		return dExpo(p, q)
	elseif type_dist == "Gaussian"
		return dGaussian(p, q)
	end
end

function dup(p, level, type_dist)
	if type_dist == "Bernoulli"
		return dupBernoulli(p, level)
	elseif type_dist == "Poisson"
		return dupPoisson(p, level)
	elseif type_dist == "Exponential"
		return dupExpo(p, level)
	elseif type_dist == "Gaussian"
		return dupGaussian(p, level)
	end
end

function dlow(p, level, type_dist)
	if type_dist == "Bernoulli"
		return dlowBernoulli(p, level)
	elseif type_dist == "Poisson"
		return dlowPoisson(p, level)
	elseif type_dist == "Exponential"
		return dlowExpo(p, level)
	elseif type_dist == "Gaussian"
		return dlowGaussian(p, level)
	end
end

function sample_arm(mu, type_dist)
	if type_dist == "Bernoulli"
		return (rand() < mu)
	elseif type_dist == "Poisson"
		return rand(Poisson(mu))
	elseif type_dist == "Exponential"
		return -mu*log(rand())
	elseif type_dist == "Gaussian"
		return mu+sigma*randn()
	end
end

function bdot(theta, type_dist)
	if type_dist == "Bernoulli"
		return exp(theta)/(1+exp(theta))
	elseif type_dist == "Poisson"
		return exp(theta)
	elseif type_dist == "Exponential"
		return -log(-theta)
	elseif type_dist == "Gaussian"
		return sigma^2*theta
	end
end

function bdotinv(mu, type_dist)
	if type_dist == "Bernoulli"
		return log(mu/(1-mu))
	elseif type_dist == "Poisson"
		return log(mu)
	elseif type_dist == "Exponential"
		return -exp(-mu)
	elseif type_dist == "Gaussian"
		mu/sigma^2
	end
end
