function ts(mu::Array, budget::Integer, dist::String)
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
    means = zeros(1, K)
    recommendations = zeros(1, budget)

    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
        recommendations[a] = mpa(N, S)
    end

    for t in (K+1):budget
        means = S ./ N
        # Most played arm
        recommendations[t] = mpa(N, S)

        TS = zeros(K)
        for a in 1:K
            if dist == "Bernouilli"
                alpha = 1
                beta = 1
                TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
			elseif dist == "Gaussian"
				TS[a] = rand(Normal(S[a] / N[a], 1.0 / N[a]), 1)[1]
			end
        end
        I = indmax(TS)
        # draw arm I
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
    end

    recommendation = mpa(N, S)
    recommendations = Int.(recommendations)

    return (recommendation, N, means, recommendations)
end
