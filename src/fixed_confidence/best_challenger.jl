"""
    best_challenger(mu::Array, delta::Real, rate::Function, dist::String,
        alpha::Real=1, beta::Real=1, fe::Bool=false, ts::Bool=false,
        challenger::Symbol=:Transportation, stopping::Symbol=:Chernoff)

Given a bandit model `mu`, a confidence level `delta`, the function returns a
guess of the best arm when reaching the confidence level. The function compares
the transportation cost (if `challenger=:Transportation`), the number of pulls
(if `challenger=:Pull`) or the proportion of pulls (if `challenger=:Proportion`)
to pick between the best arm and the challenger.
"""
function best_challenger(mu::Array, delta::Real, rate::Function, dist::String,
    alpha::Real=1, beta::Real=1, fe::Bool=false, ts::Bool=false,
    challenger::Symbol=:Transportation, stopping::Symbol=:Chernoff)
    # Flag of whether we should stop the algorithm or not
    condition = true

    num_arms = length(mu)
    num_pulls = zeros(1, num_arms)
    rewards = zeros(1, num_arms)
    # Initialization
    for a in 1:num_arms
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
        empirical_mean_best = num_pulls_best / reward_best
        weighted_means = (empirical_mean_best .+ rewards) ./
            (num_pulls_best .+ num_pulls)
        # Compute the minimum GLR
        score = minimum([num_pulls_best *
            d(empirical_mean_best, weighted_means[i], dist) +
            num_pulls[i] * d(empirical_means[i], weighted_means[i], dist)
            for i in 1:num_arms if i != empirical_best])

        # Compute the best arm and the challenger
        if ts
            for a = 1:num_arms
                # When the underlying distribution is Gaussian,
                # alpha refers to sigma
                if dist == "Gaussian"
                    empirical_means[a] =
                        rand(Normal(rewards[a] / num_pulls[a],
                        alpha / sqrt(num_pulls[a])), 1)[1]
                elseif dist == "Bernoulli"
                    empirical_means[a] =
                        rand(Beta(alpha + rewards[a],
                        beta + num_pulls[a] - rewards[a]), 1)[1]
                end
            end
            empirical_best = randmax(empirical_means)
            num_pulls_best = num_pulls[empirical_best]
            empirical_mean_best = empirical_means[empirical_best]
            weighted_means =
                (num_pulls_best * empirical_mean_best .+
                num_pulls .* empirical_means) ./ (num_pulls_best .+ num_pulls)
        end

        # Compute the challenger
        challenger = 1
        new_score = Inf
        for i = 1:num_arms
            if i != empirical_best
                score = num_pulls_best *
                    d(empirical_mean_best, weighted_means[i], dist) +
                    num_pulls[i] *
                    d(empirical_means[i], weighted_means[i], dist)
                if (score < new_score)
                    challenger = i
                    new_score = score
                end
            end
        end

        # Choose the next arm to sample
        new_sample = 1
        if (score > rate(t, delta))
            # Stop
            condition = false
        elseif (t > 1e7)
            # Stop and consider the trial as a fail
            condition = false
            true_best = 0
            print(num_pulls)
            print(rewards)
            num_pulls = zeros(1, num_arms)
        else
            # Continue and sample an arm
    	    if fe && (minimum(num_pulls) <= max(sqrt(t) - num_arms/2, 0))
                # Forced exploration
                sample = randmax(-num_pulls)
            else
    			if challenger == :Proportion
                    _, weights = optimal_weights(empirical_means, 1e-11)
     				new_sample =
                        (num_pulls_best /
                        (num_pulls_best + num_pulls[challenger]) <
                        weights[empirical_best] /
                        (weights[empirical_best] + weights[challenger])) ?
                        empirical_best : challenger
     			elseif challenger == :Transportation
     				new_sample =
                        (d(empirical_mean_best,
                        weighted_means[challenger], dist) >
                        d(empirical_means[challenger],
                        weighted_means[challenger], dist)) ?
                        empirical_best : challenger
     			elseif challenger == :Pull
     				new_sample =
                        (num_pulls[empirical_best] < num_pulls[challenger]) ?
                        empirical_best : challenger
     			end
            end
        end
        # draw the arm
        t += 1
        rewards[new_sample] += sample_arm(mu[new_sample], dist)
        num_pulls[new_sample] += 1
    end
    return true_best, num_pulls
end
