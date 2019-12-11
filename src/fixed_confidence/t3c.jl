function t3c(
    mu::Array,
    delta::Real,
    rate::Function,
    dist::String,
    alpha::Real = 1,
    beta::Real = 1,
    frac::Real = 0.5,
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
        score = minimum([num_pulls_best * d(empirical_mean_best, weighted_means[i], dist) +
                         num_pulls[i] * d(empirical_means[i], weighted_means[i], dist) for i in 1:num_arms if i != empirical_best])

        # Compute the best arm and the challenger
        ts = zeros(num_arms)
        for a = 1:num_arms
            if dist == "Gaussian"
                ts[a] = rand(
                    Normal(rewards[a] / num_pulls[a], alpha / sqrt(num_pulls[a])),
                    1,
                )[1]
            elseif dist == "Bernoulli"
                ts[a] = rand(
                    Beta(alpha + rewards[a], beta + num_pulls[a] - rewards[a]),
                    1,
                )[1]
            end
        end
        ts_best = randmax(ts)
        ts_pulls_best = num_pulls[ts_best]
        ts_mean_best = empirical_means[ts_best]
        ts_weighted_means = (ts_pulls_best * ts_mean_best .+
                             num_pulls .* empirical_means) ./ (ts_pulls_best .+ num_pulls)

        # Compute the challenger
        challenger = 1
        new_score = Inf
        for i = 1:num_arms
            if i != ts_best
                score_i = compute_transportation_general(
                    empirical_means,
                    ts_weighted_means,
                    num_pulls,
                    ts_best,
                    i,
                    dist,
                )

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
            if (rand() > frac)
                new_sample = challenger
            else
                new_sample = ts_best
            end
        end
        # Draw the arm
        t += 1
        rewards[new_sample] += sample_arm(mu[new_sample], dist)
        num_pulls[new_sample] += 1
    end
    return true_best, num_pulls
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
        score = minimum([num_pulls_best * d(empirical_mean_best, weighted_means[i], dist) +
                         num_pulls[i] * d(empirical_means[i], weighted_means[i], dist) for i in 1:num_arms if i != empirical_best])

        # Compute the best arm and the challenger
        ts = zeros(num_arms)
        for a = 1:num_arms
            if dist == "Gaussian"
                ts[a] = rand(
                    Normal(rewards[a] / num_pulls[a], alpha / sqrt(num_pulls[a])),
                    1,
                )[1]
            elseif dist == "Bernoulli"
                ts[a] = rand(
                    Beta(alpha + rewards[a], beta + num_pulls[a] - rewards[a]),
                    1,
                )[1]
            end
        end
        ts_best = randmax(ts)
        ts_pulls_best = num_pulls[ts_best]
        ts_mean_best = empirical_means[ts_best]
        ts_weighted_means = (ts_pulls_best * ts_mean_best .+
                             num_pulls .* empirical_means) ./ (ts_pulls_best .+ num_pulls)

        # Compute the challenger
        challenger = 1
        new_score = Inf
        for i = 1:num_arms
            if i != ts_best
                score_i = compute_transportation_general(
                    empirical_means,
                    ts_weighted_means,
                    num_pulls,
                    ts_best,
                    i,
                    dist,
                )
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
            if (num_pulls[ts_best] > num_pulls[challenger])
                new_sample = challenger
            else
                new_sample = ts_best
            end
        end
        # Draw the arm
        t += 1
        rewards[new_sample] += sample_arm(mu[new_sample], dist)
        num_pulls[new_sample] += 1
    end
    return true_best, num_pulls
end


function compute_transportation_general(
    means::Array,
    weighted_means::Array,
    num_pulls::Array,
    idx_i::Int,
    idx_j::Int,
    dist::String,
)
    if means[idx_j] < means[idx_i]
        cost = num_pulls[idx_i] * d(means[idx_i], weighted_means[idx_i], dist) +
               num_pulls[idx_j] * d(means[idx_j], weighted_means[idx_j], dist)
    else
        cost = 0
    end
    return cost
end
