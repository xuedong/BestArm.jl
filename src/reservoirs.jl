function sample_reservoir(reservoir::String, mu::Float64, sigma::Float64 = 1.0)
	if reservoir == "Bernoulli"
		return (rand() < mu)
	elseif reservoir == "Beta"
		return rand(Beta(mu, sigma))
	elseif reservoir == "Poisson"
		return rand(Poisson(mu))
	elseif reservoir == "Exponential"
		return - mu * log(rand())
	elseif reservoir == "Gaussian"
		return mu + sigma * randn()
	end
end
