function ttts(mu::Array, budget::Integer, dist::String, frac::Real = 0.5)
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
    @showprogress 1 "Computing..." for t in (K+1):budget
        means = S ./ N
        # idx = find(means .== maximum(means))
        # best = idx[floor(Int, length(idx) * rand()) + 1]
        # recommendations[t] = best
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
                val, _ = hcubature(f, 0.0, 1.0)
                probs[a] = val
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
        I = indmax(TS)
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
                J = indmax(TS)
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
	return regrets
end
