function ttts(mu::Array, budget::Integer, dist::String)
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)

    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
    end
end
