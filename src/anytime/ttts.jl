function ttts(mu::Array, budget::Integer,
	dist::String, frac::Real = 0.5, default::Bool = true,
	final::Bool = true)
	num = length(mu)
    N = zeros(1, num)
    S = zeros(1, num)
    means = ones(1, num) * -Inf
    probs = ones(1, num) / num
	max_probs = zeros(1, budget)
    recommendations = zeros(1, budget)

    best = 1
    for t in 1:budget
        means = [N[i] == 0 ? -Inf : S[i]/N[i] for i in 1:num]
		if final == false
	        if default
	            idx = (LinearIndices(means .== maximum(means)))[findall(means .== maximum(means))]
	            best = idx[floor(Int, length(idx) * rand()) + 1]
	            recommendations[t] = best
	        else
	            idx = (LinearIndices(probs .== maximum(probs)))[findall(probs .== maximum(probs))]
	            best = idx[floor(Int, length(idx) * rand()) + 1]
	            recommendations[t] = best
				max_probs[t] = probs[best]

	            for a in 1:num
	                if dist == "Bernoulli"
	                    alpha = 1
	                    beta = 1
	                    function f(x)
	                        prod = pdf.(Beta(alpha + S[a], beta + N[a] - S[a]), x)[1]
	                        # println(prod)
	                        for i in 1:num
	                            if i != a
	                                prod *= cdf.(Beta(alpha + S[i], beta + N[i] - S[i]), x)[1]
	                                # println(prod)
	                            end
	                        end
	                        return prod
	                    end
	                    val, _ = hquadrature(f, 0.0, 1.0)
	                    probs[a] = val
	                end
	            end
	        end
		end

        TS = zeros(num)
        for a in 1:num
            if dist == "Bernoulli"
                alpha = 1
                beta = 1
                TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
			elseif dist == "Gaussian"
				TS[a] = rand(Normal(S[a] / N[a], 1.0 / N[a]), 1)[1]
			end
        end
        I = argmax(TS)
        if (rand() > frac)
            J = I
			count = 1
            while (I == J) && (count < 10000)
                TS = zeros(num)
                if dist == "Bernoulli"
                    alpha = 1
                    beta = 1
                    for a = 1:num
                        TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
					end
				elseif dist == "Gaussian"
					for a = 1:num
						TS[a] = rand(Normal(S[a] / N[a], 1.0 / N[a]), 1)[1]
					end
                end
                J = argmax(TS)
				count += 1
            end
            I = J
        end
        # draw arm I
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
    end

	if final == false
    	recommendation = best
	else
		if default
			idx = (LinearIndices(means .== maximum(means)))[findall(means .== maximum(means))]
			best = idx[floor(Int, length(idx) * rand()) + 1]
			recommendation = best
		else
			for a in 1:num
				if dist == "Bernoulli"
					alpha = 1
					beta = 1
					function f(x)
						prod = pdf.(Beta(alpha + S[a], beta + N[a] - S[a]), x)[1]
						# println(prod)
						for i in 1:num
							if i != a
								prod *= cdf.(Beta(alpha + S[i], beta + N[i] - S[i]), x)[1]
								# println(prod)
							end
						end
						return prod
					end
					val, _ = hquadrature(f, 0.0, 1.0)
					probs[a] = val
				end
			end

			idx = (LinearIndices(probs .== maximum(probs)))[findall(probs .== maximum(probs))]
			best = idx[floor(Int, length(idx) * rand()) + 1]
			recommendation = best
		end
	end

    recommendations = Int.(recommendations)

    return (recommendation, N, means, recommendations, max_probs)
end


function parallel_ttts(mu::Array, budget::Integer, dist::String)
	_, _, _, recs = ttts(mu, budget, dist)
	regrets = compute_regrets(mu, recs, budget)
    # gc()
	return regrets
end


function ttts_infinite(reservoir::String, num::Integer, budget::Integer,
	dist::String, frac::Real = 0.5, default::Bool = true,
	theta1::Float64 = 1.0, theta2::Float64 = 1.0,
	final::Bool = true, shift::Real = 1.0)
	mu = [sample_reservoir(reservoir, theta1, theta2, shift) for _ in 1:num]
    N = zeros(1, num)
    S = zeros(1, num)
    means = ones(1, num) * -Inf
    probs = ones(1, num) / num
    recommendations = zeros(1, budget)

    best = 1
    for t in 1:budget
        means = [N[i] == 0 ? -Inf : S[i]/N[i] for i in 1:num]
		if final == false
	        if default
	            idx = (LinearIndices(means .== maximum(means)))[findall(means .== maximum(means))]
	            best = idx[floor(Int, length(idx) * rand()) + 1]
	            recommendations[t] = best
	        else
	            idx = (LinearIndices(probs .== maximum(probs)))[findall(probs .== maximum(probs))]
	            best = idx[floor(Int, length(idx) * rand()) + 1]
	            recommendations[t] = best

	            for a in 1:num
	                if dist == "Bernoulli"
	                    alpha = 1
	                    beta = 1
	                    function f(x)
	                        prod = pdf.(Beta(alpha + S[a], beta + N[a] - S[a]), x)[1]
	                        # println(prod)
	                        for i in 1:num
	                            if i != a
	                                prod *= cdf.(Beta(alpha + S[i], beta + N[i] - S[i]), x)[1]
	                                # println(prod)
	                            end
	                        end
	                        return prod
	                    end
	                    val, _ = hquadrature(f, 0.0, 1.0)
	                    probs[a] = val
	                end
	            end
	        end
		end

        TS = zeros(num)
        for a in 1:num
            if dist == "Bernoulli"
                alpha = 1
                beta = 1
                TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
			elseif dist == "Gaussian"
				TS[a] = rand(Normal(S[a] / N[a], sqrt(1.0 / N[a])), 1)[1]
			end
        end
        I = argmax(TS)
        if (rand() > frac)
            J = I
			count = 1
            while (I == J) && (count < 10000)
                TS = zeros(num)
                if dist == "Bernoulli"
                    alpha = 1
                    beta = 1
                    for a = 1:num
                        TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
					end
				elseif dist == "Gaussian"
					for a = 1:num
						TS[a] = rand(Normal(S[a] / N[a], sqrt(1.0 / N[a])), 1)[1]
					end
                end
                J = argmax(TS)
				count += 1
            end
            I = J
        end
        # draw arm I
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
    end

	if final == false
    	recommendation = best
	else
		if default
			idx = (LinearIndices(means .== maximum(means)))[findall(means .== maximum(means))]
			best = idx[floor(Int, length(idx) * rand()) + 1]
			recommendation = best
		else
			for a in 1:num
				if dist == "Bernoulli"
					alpha = 1
					beta = 1
					function f(x)
						prod = pdf.(Beta(alpha + S[a], beta + N[a] - S[a]), x)[1]
						# println(prod)
						for i in 1:num
							if i != a
								prod *= cdf.(Beta(alpha + S[i], beta + N[i] - S[i]), x)[1]
								# println(prod)
							end
						end
						return prod
					end
					val, _ = hquadrature(f, 0.0, 1.0)
					probs[a] = val
				end
			end

			idx = (LinearIndices(probs .== maximum(probs)))[findall(probs .== maximum(probs))]
			best = idx[floor(Int, length(idx) * rand()) + 1]
			recommendation = best
		end
	end

    recommendations = Int.(recommendations)

    return (recommendation, N, means, recommendations, mu)
end


function ttts_dynamic(reservoir::String, num::Integer, limit::Integer,
	budget::Integer, dist::String, frac::Real = 0.5,
	default::Bool = true, theta1::Float64 = 1.0, theta2::Float64 = 1.0,
	final::Bool = true, shift::Real = 1.0)
	mu = [sample_reservoir(reservoir, theta1, theta2, shift) for _ in 1:num]
	S_0 = 1
    N = [0 for _ in 1:num]
    S = [0 for _ in 1:num]
    # means = [-Inf for _ in 1:num]
    recommendations = zeros(1, budget)
	dynamic_num = num

    best = 1
    for t in 1:budget
		if final == false
			if default
				recommendations[t] = mpa(N, S)
			else
		        # means = [N[i] == 0 ? -Inf : S[i]/N[i] for i in 1:dynamic_num]
				probs = [1/dynamic_num for _ in 1:dynamic_num]
		        for a in 1:dynamic_num
		            if dist == "Bernoulli"
		                alpha = 1
		                beta = 1
		                function f(x)
		                    prod = pdf.(Beta(alpha + S[a], beta + N[a] - S[a]), x)[1]
		                    # println(prod)
		                    for i in 1:dynamic_num
		                        if i != a
		                            prod *= cdf.(Beta(alpha + S[i], beta + N[i] - S[i]), x)[1]
		                            # println(prod)
		                        end
		                    end
		                    return prod
		                end
		                # val, _ = hquadrature(f, 0.0, 1.0)
						val, _ = hquadrature(f, 0.0, shift)
						probs[a] = val
		            end
				end

				idx = (LinearIndices(probs .== maximum(probs)))[findall(probs .== maximum(probs))]
		        best = idx[floor(Int, length(idx) * rand()) + 1]
		        recommendations[t] = best
			end
		end

        TS = zeros(dynamic_num)
        for a in 1:dynamic_num
            if dist == "Bernoulli"
                alpha = 1
                beta = 1
            	# TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
				TS[a] = betainvcdf(alpha + S[a], beta + N[a] - S[a], rand(1)[1]*betacdf(alpha + S[a], beta + N[a] - S[a], shift))
			elseif dist == "Gaussian"
				TS[a] = rand(Normal(S[a] / N[a], sqrt(1.0 / N[a])), 1)[1]
			end
        end
		# TS_0 = rand(Beta(S_0, 1.0), 1)[1] * shift
		TS_0 = betainvcdf(S_0, 1.0, rand(1)[1]*betacdf(S_0, 1.0, shift))
        I = argmax(vcat(TS, TS_0))
        if (rand() > frac)
            J = I
			count = 1
            while (I == J) && (count < 10000)
                TS = zeros(dynamic_num)
                if dist == "Bernoulli"
                    alpha = 1
                    beta = 1
                    for a = 1:dynamic_num
                        # TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
						TS[a] = betainvcdf(alpha + S[a], beta + N[a] - S[a], rand(1)[1]*betacdf(alpha + S[a], beta + N[a] - S[a], shift))
					end
				elseif dist == "Gaussian"
					for a = 0:dynamic_num
						TS[a] = rand(Normal(S[a] / N[a], sqrt(1.0 / N[a])), 1)[1]
					end
                end
				# TS_0 = rand(Beta(S_0, 1.0), 1)[1] * shift
				TS_0 = betainvcdf(S_0, 1.0, rand(1)[1]*betacdf(S_0, 1.0, shift))
		        J = argmax(vcat(TS, TS_0))
				count += 1
            end
            I = J
        end

        # draw arm I
		if I == dynamic_num + 1
			if sample_arm(1.0, "Bernoulli")
				if dynamic_num < limit
					dynamic_num += 1
				end
				new = sample_reservoir(reservoir, theta1, theta2, shift)
				push!(N, 0)
				push!(S, 0)
				push!(mu, new)
        		S[I] += sample_arm(mu[I], dist)
        		N[I] += 1
				S_0 += 1
			end
		else
			S[I] += sample_arm(mu[I], dist)
        	N[I] += 1
		end
    end

	if final == false
    	recommendation = best
	else
		if default
			recommendation = mpa(N, S)
		else
			# means = [N[i] == 0 ? -Inf : S[i]/N[i] for i in 1:dynamic_num]
			probs = [1/dynamic_num for _ in 1:dynamic_num]
			for a in 1:dynamic_num
				if dist == "Bernoulli"
					alpha = 1
					beta = 1
					function f(x)
						prod = pdf.(Beta(alpha + S[a], beta + N[a] - S[a]), x)[1]
						# println(prod)
						for i in 1:dynamic_num
							if i != a
								prod *= cdf.(Beta(alpha + S[i], beta + N[i] - S[i]), x)[1]
								# println(prod)
							end
						end
						return prod
					end
					# val, _ = hquadrature(f, 0.0, 1.0)
					val, _ = hquadrature(f, 0.0, shift)
					probs[a] = val
				end
			end

			idx = (LinearIndices(probs .== maximum(probs)))[findall(probs .== maximum(probs))]
			best = idx[floor(Int, length(idx) * rand()) + 1]
			recommendation = best
		end
	end

    recommendations = Int.(recommendations)

    return (recommendation, N, recommendations, mu)
end


function parallel_ttts_dynamic(reservoir::String, num::Integer, budget::Integer,
	dist::String, frac::Real = 0.5,
	theta1::Float64 = 1.0, theta2::Float64 = 1.0)
	_, _, _, recs = ttts_dynamic(mu, budget, dist)
	regrets = compute_regrets(mu, recs, budget)
    # gc()
	return regrets
end
