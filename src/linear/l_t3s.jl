function l_t3s(contexts::Array, theta::Array, delta::Real, rate::Function,
	dist::String, sigma::Real=1, kappa::Real=1, frac::Real=0.5,
	stopping::Symbol=:Chernoff)
    condition = true
   	num_contexts = length(contexts)
	dim = length(contexts[1])
   	num_pulls = zeros(1, num_contexts)
   	rewards = zeros(1, num_contexts)
	t = 0

	# Initialize the prior
	lambda = sigma^2 / kappa^2
	design_inverse = Matrix{Float64}(1/lambda * I, dim, dim)
	z_t = vec(zeros(1, dim))
   	rls = vec(zeros(1, dim))
	var = Matrix{Float64}(kappa^2 * I, dim, dim)

	# Play each arm once
	for c in 1:num_contexts
		t += 1
		new_reward = compute_observation(contexts[c], theta, sigma)
		rewards[c] += new_reward
		num_pulls[c] = 1
	end

	best = 1
   	while (condition)
      	empirical_means = rewards ./ num_pulls
      	# Empirical best arm
      	best = randmax(empirical_means)
      	# Compute the stopping statistic
      	num_pulls_best = num_pulls[best]
      	reward_best = rewards[best]
      	empirical_mean_best = reward_best / num_pulls_best
      	weighted_means = (reward_best .+ rewards) ./ (num_pulls_best .+ num_pulls)

      	index = collect(1:num_contexts)
      	deleteat!(index, best)
		# Compute the minimum GLR
		score = minimum([num_pulls_best * d(empirical_mean_best, weighted_means[i], dist) + num_pulls[i] * d(empirical_means[i], weighted_means[i], dist) for i in 1:num_contexts if i != best])
      	if (score > rate(t, delta))
         	# Stop
         	condition = false
      	elseif (t > 1e7)
         	condition = false
         	best = 0
         	print(num_pulls)
         	print(rewards)
         	num_pulls = zeros(1, num_contexts)
      	else
         	ts = zeros(num_contexts)
         	for a = 1:num_contexts
            	if dist == "Gaussian"
               		ts[a] = sum(rand(MvNormal(rls, var)) .* contexts[a])
				end
         	end

         	new_sample = argmax(ts)
         	if (rand() > frac)
            	challenger = new_sample
            	condition = true
            	while (new_sample == challenger)
					ts = zeros(num_contexts)
		         	for a = 1:num_contexts
		            	if dist == "Gaussian"
		               		ts[a] = sum(rand(MvNormal(rls, var)) .* contexts[a])
		            	end
		         	end
               		challenger = argmax(ts)
            	end
            	new_sample = challenger
         	end
         	# Play the selected arm
	      	t += 1
			new_reward = compute_observation(contexts[new_sample], theta, sigma)
	      	rewards[new_sample] += new_reward
	      	num_pulls[new_sample] += 1

			# Update the posterior
			design_inverse = update_design_inverse(design_inverse, contexts[new_sample])
			z_t += new_reward * contexts[new_sample]
			rls = design_inverse * z_t
			var = sigma^2 * design_inverse
	   	end
   	end
   	recommendation = best
   	return recommendation, num_pulls
end


# Helper functions of L-T3S
function compute_observation(context::Array, theta::Array, sigma::Real=1)
	obs = dot(context, theta) + rand(Normal(0, sigma^2))[1]
end


function update_design_inverse(matrix::Array, context::Array)
	matrix = matrix - matrix * context * transpose(context) * matrix / (1 + transpose(context) * matrix * context)
	return matrix
end
