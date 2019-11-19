function l_t3s(contexts::Array, delta::Real, rate::Function, dist::String,
	alpha::Real=1, beta::Real=1, frac::Real=0.5, stopping::Symbol=:Chernoff)
    condition = true
   	num_contexts = length(contexts)
	dim = length(contexts[1])
   	num_pulls = zeros(1, contexts)
   	rewards = zeros(1, contexts)

   	rls = zeros(1, dim)
	var = zeros(dim, dim)

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
		score = minimum([num_pulls_best * d(empirical_mean_best, weighted_means[i], dist) + num_pulls[i] * d(empirical_means[i], weighted_means[i], dist) for i in 1:num_arms if i != empirical_best])
      	if (score > rate(t, delta))
         	# stop
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
         	# draw arm I
	      	t += 1
	      	rewards[new_sample] += compute_observation(contexts[new_sample], theta)
	      	num_pulls[new_sample] += 1
	   	end
   	end
   	recommendation = best
   	return recommendation, num_pulls
end


# Helper functions of L-T3S
function compute_observation(context::Array, theta::Array, sigma::Real=1)
	
end
