function t3c(
    mu::Array,
    delta::Real,
    rate::Function,
    dist::String,
    frac::Real = 0.5,
    alpha::Real = 1,
    beta::Real = 1,
    stopping::String = "chernoff",
)
   # T3C with Chernoff stopping rule
    continuing = true
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
   # initialization
    for a = 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
    end
    t = K
    TrueBest = 1
    while continuing
        Mu = S ./ N
      # Empirical best arm
        Best = randmax(Mu)
        TrueBest = randmax(Mu)
      # Compute the stopping statistic
        NB = N[Best]
        SB = S[Best]
        MuB = SB / NB
        MuMid = (SB .+ S) ./ (NB .+ N)
        Score = minimum([NB * d(MuB, MuMid[i], dist) + N[i] * d(Mu[i], MuMid[i], dist) for i in 1:K if i != Best])
      # compute the best arm and the challenger
        for a = 1:K
            if dist == "Gaussian"
                Mu[a] = rand(Normal(S[a] / N[a], alpha / sqrt(N[a])), 1)[1]
            elseif dist == "Bernoulli"
                Mu[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
            end
        end
        Best = randmax(Mu)
        NB = N[Best]
        MuB = Mu[Best]
        MuMid = (NB * MuB .+ N .* Mu) ./ (NB .+ N)
        Challenger = 1
        NewScore = Inf
        for i = 1:K
            if i != Best
                score = NB * d(MuB, MuMid[i], dist) + N[i] * d(Mu[i], MuMid[i], dist)
                if (score < NewScore)
                    Challenger = i
                    NewScore = score
                end
            end
        end
        I = 1
        if (Score > rate(t, delta))
         # stop
            continuing = false
        elseif (t > 1e7)
         # stop and return (0,0)
            continuing = false
            TrueBest = 0
            print(N)
            print(S)
            N = zeros(1, K)
        else
         # continue and sample an arm
            if (rand() > frac)
            # TS sample
                I = Best
            else
            # choose between the arm and its Challenger
                I = Challenger
            end
        end
      # draw the arm
        t += 1
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
    end
    return TrueBest, N
end


"""
    t3c_optimal(mu::Array, delta::Real, rate::Function, dist::String,
        alpha::Real = 1, beta::Real = 1, stopping::String = "chernoff")

Optimal version of T3C?
"""
function t3c_optimal(
    mu::Array,
    delta::Real,
    rate::Function,
    dist::String,
    alpha::Real = 1,
    beta::Real = 1,
    stopping::String = "chernoff",
)
    condition = true
    num_arms = length(mu)
    num_pulls = zeros(1, num_arms)
    rewards = zeros(1, num_arms)
    # Initialization
    for a = 1:num_arms
        num_pulls[a] = 1
        rewards[a] = sample_arm(mu[a], dist)
    end

    # Exploration phase
    t = num_arms
    true_best = 1
    while condition
        empirical_means = rewards ./ num_pulls

        # Store the empirical best arm
        empirical_best = randmax(empirical_means)
        true_best = randmax(empirical_means)

        # Compute the stopping statistic
        num_pulls_best = num_pulls[empirical_best]
        reward_best = rewards[empirical_best]
        empirical_mean_best = reward_best / num_pulls_best
        weighted_means = (reward_best .+ rewards) ./ (num_pulls_best .+ num_pulls)
        # Compute the minimum GLR
        score = minimum([num_pulls_best * d(empirical_mean_best, weighted_means[i], dist) + num_pulls[i] * d(empirical_means[i], weighted_means[i], dist) for i in 1:num_arms if i != empirical_best])

        # Compute the best arm and the challenger
        for a = 1:num_arms
            if dist == "Gaussian"
                empirical_means[a] = rand(Normal(rewards[a] / num_pulls[a], alpha / sqrt(num_pulls[a])), 1)[1]
            elseif dist == "Bernoulli"
                empirical_means[a] = rand(Beta(alpha + rewards[a], beta + num_pulls[a] - rewards[a]), 1)[1]
            end
        end
        empirical_best = randmax(empirical_means)
        num_pulls_best = num_pulls[empirical_best]
        empirical_mean_best = empirical_means[empirical_best]
        weighted_means = (num_pulls_best * empirical_mean_best .+ num_pulls .* empirical_means) ./ (num_pulls_best .+ num_pulls)

        # Compute the challenger
        challenger = 1
        new_score = Inf
        for i = 1:num_arms
            if i != empirical_best
                score_i = num_pulls_best * d(empirical_mean_best, weighted_means[i], dist) + num_pulls[i] * d(empirical_means[i], weighted_means[i], dist)
                if (score_i < new_score)
                    challenger = i
                    new_score = score_i
                end
            end
        end

        # Choose the next arm to sample
        new_sample = 1
        if (score > rate(t, delta))
            # Stop
            condition = false
        elseif (t > 1e7)
            # Stop and return (0,0)
            condition = false
            true_best = 0
            println(num_pulls)
            println(rewards)
            num_pulls = zeros(1, num_arms)
        else
            # Continue and sample the arm less pulled
            if (num_pulls[empirical_best] > num_pulls[challenger])
                new_sample = challenger
            else
                new_sample = empirical_best
            end
        end
        # Draw the arm
        t += 1
        rewards[new_sample] += sample_arm(mu[new_sample], dist)
        num_pulls[new_sample] += 1
    end
    return true_best, num_pulls
end
