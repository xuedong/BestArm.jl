function ttts(mu::Array, budget::Integer, dist::String, frac::Real = 0.5)
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
    means = zeros(1, K)
    recommendations = zeros(1, budget)

    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
        recommendations[a] = eba(N, S)
    end

    best = 1
    for t in (K+1):budget
        means = S ./ N
        idx = find(means .== maximum(means))
        # Empirical best arm
        best = idx[floor(Int, length(idx) * rand()) + 1]
        recommendations[t] = best

        TS = zeros(K)
        for a in 1:K
            if dist == "Bernouilli"
                alpha = 0.5
                beta = 0.5
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
                    alpha = 0.5
                    beta = 0.5
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
