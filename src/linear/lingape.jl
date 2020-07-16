function lingape(
    contexts::Array,
    true_theta::Array,
    delta::Real,
    rate::Function,
    dist::String,
    sigma::Real = 0,
    kappa::Real = 1,
    epsilon::Real = 0.0,
)
    # Initialization
    condition = true
    num_contexts = length(contexts)
    dim = length(contexts[1])
    num_pulls = zeros(1, num_contexts)
    rewards = zeros(1, num_contexts)
    t = 0

    # Initialize the prior
    lambda = sigma^2 / kappa^2
    design = Matrix{Float64}(lambda * I, dim, dim)
    design_inverse = Matrix{Float64}(I, dim, dim)
    #square_root_inverse = Matrix{Float64}(kappa / sigma * I, dim, dim)
    z_t = vec(zeros(1, dim))
    rls = vec(zeros(1, dim))
    var = Matrix{Float64}(kappa^2 * I, dim, dim)

    # Play each arm once
    for c = 1:num_contexts
        t += 1
        new_reward = compute_observation(contexts[c], true_theta, sigma)
        rewards[c] += new_reward
        num_pulls[c] = 1

        design = update_design(design, contexts[c])
        design_inverse = update_design_inverse(design_inverse, contexts[c])
        #square_root_inverse = update_square_root(square_root_inverse, contexts[c])
        z_t += new_reward * contexts[c]
        rls = design_inverse * z_t
        var = sigma^2 * design_inverse
    end

    best = 1
    while condition
        empirical_means = [dot(contexts[c], rls) for c = 1:num_contexts]
        # Empirical best arm
        best = randmax(empirical_means)

        if (t > 1e7)
            condition = false
            best = 0
            println(num_pulls)
            println(rewards)
            num_pulls = zeros(1, num_contexts)
        else
            #c_t = compute_error_width(design, true_theta, sigma, kappa, delta)
            c_t = sqrt(2*rate(t, delta))
            ambiguous = randmax([compute_gap(contexts[i], contexts[best], rls) +
                                 compute_confidence(
                contexts[i],
                contexts[best],
                design_inverse,
            ) * c_t for i = 1:num_contexts])
            ucb = maximum([compute_gap(contexts[i], contexts[best], rls) +
                           compute_confidence(contexts[i], contexts[best], design_inverse) *
                           c_t for i = 1:num_contexts])
            if ucb <= epsilon
                recommendation = best
                return recommendation, num_pulls
            end
            new_sample = randmin([compute_confidence(
                contexts[best],
                contexts[ambiguous],
                update_design_inverse(design_inverse, contexts[i]),
            ) for i = 1:num_contexts])

            # Play the selected arm
            t += 1
            new_reward = compute_observation(contexts[new_sample], true_theta, sigma)
            rewards[new_sample] += new_reward
            num_pulls[new_sample] += 1

            # Update the posterior
            design = update_design(design, contexts[new_sample])
            design_inverse = update_design_inverse(design_inverse, contexts[new_sample])
            z_t += new_reward * contexts[new_sample]
            rls = design_inverse * z_t
            var = sigma^2 * design_inverse
        end
    end
    recommendation = best
    return recommendation, num_pulls
end


# Helper functions
function compute_gap(context1::Array, context2::Array, theta::Array)
    gap = dot(context1 - context2, theta)
    return gap
end
