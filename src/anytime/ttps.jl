function ttps(mu::Array, budget::Integer, dist::String, frac::Real = 0.5)
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
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

    return (recommendation, N, probs, recommendations)
end


function parallel_ttps(mu::Array, budget::Integer, dist::String)
	_, _, _, recs = ttps(mu, budget, dist)
	regrets = compute_regrets(mu, recs, budget)
	return regrets
end
