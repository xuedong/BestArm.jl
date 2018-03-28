function d(p, q, dist)
	if dist == "Bernoulli"
		return dBernoulli(p, q)
	elseif dist == "Poisson"
		return dPoisson(p, q)
	elseif dist == "Exponential"
		return dExpo(p, q)
	elseif dist == "Gaussian"
		return dGaussian(p, q)
	end
end

function dup(p, level, dist)
	if dist == "Bernoulli"
		return dupBernoulli(p, level)
	elseif dist == "Poisson"
		return dupPoisson(p, level)
	elseif dist == "Exponential"
		return dupExpo(p, level)
	elseif dist == "Gaussian"
		return dupGaussian(p, level)
	end
end

function dlow(p, level, dist)
	if dist == "Bernoulli"
		return dlowBernoulli(p, level)
	elseif dist == "Poisson"
		return dlowPoisson(p, level)
	elseif dist == "Exponential"
		return dlowExpo(p, level)
	elseif dist == "Gaussian"
		return dlowGaussian(p, level)
	end
end

function sample_arm(mu, dist)
	if dist == "Bernoulli"
		return (rand() < mu)
	elseif dist == "Poisson"
		return rand(Poisson(mu))
	elseif dist == "Exponential"
		return -mu*log(rand())
	elseif dist == "Gaussian"
		return mu+sigma*randn()
	end
end

function bdot(theta, dist)
	if dist == "Bernoulli"
		return exp(theta)/(1+exp(theta))
	elseif dist == "Poisson"
		return exp(theta)
	elseif dist == "Exponential"
		return -log(-theta)
	elseif dist == "Gaussian"
		return sigma^2*theta
	end
end

function bdotinv(mu, dist)
	if dist == "Bernoulli"
		return log(mu/(1-mu))
	elseif dist == "Poisson"
		return log(mu)
	elseif dist == "Exponential"
		return -exp(-mu)
	elseif dist == "Gaussian"
		mu/sigma^2
	end
end
