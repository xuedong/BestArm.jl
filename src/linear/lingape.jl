function lingape(contexts::Array, theta::Array, delta::Real, rate::Function, dist::String)
    # Initialization
    condition = true
    num_contexts = length(contexts)
    dim = length(contexts[1])
    num_pulls = zeros(1, num_contexts)
    rewards = zeros(1, num_contexts)
    t = 0

    # Initialize the prior
    lambda = sigma^2 / kappa^2
    design_inverse = Matrix{Float64}(1 / lambda * I, dim, dim)
    square_root_inverse = Matrix{Float64}(kappa / sigma * I, dim, dim)
    z_t = vec(zeros(1, dim))
    rls = vec(zeros(1, dim))
    var = Matrix{Float64}(kappa^2 * I, dim, dim)

    # Play each arm once
    for c = 1:num_contexts
        new_reward = compute_observation(contexts[c], theta, sigma)
        rewards[c] += new_reward
        num_pulls[c] = 1

        design_inverse = update_design_inverse(design_inverse, contexts[c])
        square_root_inverse = update_square_root(square_root_inverse, contexts[c])
        z_t += new_reward * contexts[c]
        rls = design_inverse * z_t
        var = sigma^2 * design_inverse
    end

    t = K
    Best = 1
    while (condition)
        Mu = S ./ N
        # Empirical best arm
        Best = randmax(Mu)
        # Find the challenger
        UCB = zeros(1, K)
        LCB = zeros(1, K)
        for a = 1:K
            UCB[a] = dup(Mu[a], rate(t, delta) / N[a], dist)
            LCB[a] = dlow(Mu[a], rate(t, delta) / N[a], dist)
        end
        B = zeros(1, K)
        for a = 1:K
            Index = collect(1:K)
            deleteat!(Index, a)
            B[a] = maximum(UCB[Index]) - LCB[a]
        end
        Value = minimum(B)
        Best = argmin(B)[2]
        UCB[Best] = 0
        Challenger = argmax(UCB)
        # choose which arm to draw
        t = t + 1
        I = (N[Best] < N[Challenger]) ? Best : Challenger
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
        # check stopping condition
        condition = (Value > 0)
        if (t > 1000000)
            condition = false
            Best = 0
            N = zeros(1, K)
        end
    end
    recommendation = Best
    return (recommendation, N)
end
