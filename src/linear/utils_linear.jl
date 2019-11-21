function compute_observation(context::Array, theta::Array, sigma::Real=1)
	obs = dot(context, theta) + rand(Normal(0, sigma^2))[1]
end


function update_design_inverse(matrix::Array, context::Array)
	matrix = matrix - matrix * context * transpose(context) * matrix / (1 + transpose(context) * matrix * context)
	return matrix
end
