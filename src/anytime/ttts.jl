function ttts(mu::Array, budget::Integer, dist::String, frac::Real = 0.5, default::Bool = true)
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
    means = zeros(1, K)
    probs = ones(1, K) / K
    recommendations = zeros(1, budget)

    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
        recommendations[a] = rand(1:K)
    end

    best = 1
    for t in (K+1):budget
        means = S ./ N
        if default
            idx = (LinearIndices(means .== maximum(means)))[findall(means .== maximum(means))]
            best = idx[floor(Int, length(idx) * rand()) + 1]
            recommendations[t] = best
        else
            idx = find(probs .== maximum(probs))
            best = idx[floor(Int, length(idx) * rand()) + 1]
            recommendations[t] = best

            for a in 1:K
                if dist == "Bernoulli"
                    alpha = 1
                    beta = 1
                    function f(x)
                        prod = pdf.(Beta(alpha + S[a], beta + N[a] - S[a]), x)[1]
                        # println(prod)
                        for i in 1:K
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

        TS = zeros(K)
        for a in 1:K
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
            while (I == J)
                TS = zeros(K)
                if dist == "Bernoulli"
                    alpha = 1
                    beta = 1
                    for a = 1:K
                        TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
					end
				elseif dist == "Gaussian"
					for a = 1:K
						TS[a] = rand(Normal(S[a] / N[a], 1.0 / N[a]), 1)[1]
					end
                end
                J = argmax(TS)
            end
            I = J
        end
        # draw arm I
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
    end

    recommendation = best
    recommendations = Int.(recommendations)

    return (recommendation, N, means, recommendations)
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
	final::Bool = true)
	mu = [sample_reservoir(reservoir, theta1, theta2) for _ in 1:num]
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
	            idx = find(probs .== maximum(probs))
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

    return (recommendation, N, means, recommendations, mu)
end


function ttts_dynamic(reservoir::String, num::Integer, limit::Integer,
	budget::Integer, dist::String, frac::Real = 0.5,
	default::Bool = true, theta1::Float64 = 1.0, theta2::Float64 = 1.0,
	final::Bool = true)
	mu = [sample_reservoir(reservoir, theta1, theta2) for _ in 1:num]
    N = [0 for _ in 1:num]
    S = [0 for _ in 1:num]
    # means = [-Inf for _ in 1:num]
    recommendations = zeros(1, budget)
	dynamic_num = num

    best = 1
    for t in 1:budget
		new = sample_reservoir(reservoir, theta1, theta2)
		push!(N, 0)
		push!(S, 0)
		push!(mu, new)
		if dynamic_num < limit
			dynamic_num += 1
		end

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
		                val, _ = hquadrature(f, 0.0, 1.0)
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
                TS = zeros(dynamic_num)
                if dist == "Bernoulli"
                    alpha = 1
                    beta = 1
                    for a = 1:dynamic_num
                        TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
					end
				elseif dist == "Gaussian"
					for a = 1:dynamic_num
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
