function ttps(mu::Array, budget::Integer, dist::String, frac::Real = 0.5)
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
    probs = zeros(1, K)
    recommendations = zeros(1, budget)

    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
        recommendations[a] = eba(N, S)
        probs[a] = 1.0 / K
    end

    best = 1
    for t in (K+1):budget
        idx = find(probs .== maximum(probs))
        # Empirical best arm
        best = idx[floor(Int, length(idx) * rand()) + 1]
        recommendations[t] = best

        for a in 1:K
            if dist == "Bernoulli"
                alpha = 0.5
                beta = 0.5
                function f(x)
                    prod::Real = pdf(Beta(alpha + S[a], beta + N[a] - S[a]), x)
                    for i in 1:K
                        if i != a
                            prod *= cdf(Beta(alpha + S[i], beta + N[i] - S[i]), x)
                        end
                    end
                end
                probs[a] = hcubature(f, 0.0, 1.0)
            end
        end
        I = indmax(probs)
        if (rand() > frac)
            I = indmax(vcat(probs[1:(I-1)], probs[(I+1):end]))
        end
        # draw arm I
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
    end

    recommendation = best
    recommendations = Int.(recommendations)

    return (recommendation, N, means, recommendations)
end
