function sample_reservoir(reservoir::String, theta1::Float64, theta2::Float64 = 1.0, shift::Float64 = 2.0)
	if reservoir == "Bernoulli"
		return (rand() < theta1)
	elseif reservoir == "Beta"
		return rand(Beta(theta1, theta2))
	elseif reservoir == "ShiftedBeta"
		return rand(Beta(theta1, theta2))/shift
	elseif reservoir == "Poisson"
		return rand(Poisson(theta1))
	elseif reservoir == "Exponential"
		return - theta1 * log(rand())
	elseif reservoir == "Gaussian"
		return theta1 + theta2 * randn()
	end
end
