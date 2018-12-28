function sample_resevoir(reservoir::String, mu::Float64, sigma::Float64 = 1.0)
	if dist == "Bernoulli"
		return (rand() < mu)
	elseif dist == "Beta"
		return rand(Beta(mu, sigma))
	elseif dist == "Poisson"
		return rand(Poisson(mu))
	elseif dist == "Exponential"
		return - mu * log(rand())
	elseif dist == "Gaussian"
		return mu + sigma * randn()
	end
end
