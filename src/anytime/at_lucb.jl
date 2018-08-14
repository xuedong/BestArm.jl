function at_lucb(mu::Array, budget::Integer, dist::String, delta_1::Real = 0.01, alpha::Real = 0.5, epsilon::Real = 0)
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
    means = zeros(1, K)
    recommendations = zeros(1, budget)
    delta_s = delta_1

    term = false
    t = 1
    s = 1
    best = 1
    while t <= budget
        if term
            term_s = true
            J = best
            ucbs = copy(means)
            ucbs[J] = -Inf
            lcbs = copy(means[J])

            while term_s
                s += 1
                delta_s *= alpha
                bonus = compute_deviation(K, N, t, delta_s)

                gap = maximum(ucbs+bonus) - minimum(lcbs-bonus)
                term_s = (gap < epsilon)
            end
        else
            if s == 1
                if @isdefined best
                    J = best
                else
                    J = argmax(means)[2]
                end
            end
        end

        if t == 1
            best = J
        end

        bonus = compute_deviation(K, N, t, delta_s)
        ucbs = means + bonus
        ucbs[best] = -Inf
        lcbs = means - bonus

        l = argmax(ucbs)[2]
        first_sample = sample_arm(mu[l], dist)
        N[l] += 1
        S[l] += first_sample
        means[l] = S[l] / N[l]

        h = best
        second_sample = sample_arm(mu[h], dist)
        N[h] += 1
        S[h] += second_sample
        means[h] = S[h] / N[h]

        recommendations[t] = J
        recommendations[t+1] = J
        t += 2

        best = argmax(means)[2]

        bonus = compute_deviation(K, N, t, delta_s)
        ucbs = means + bonus
        ucbs[best] = -Inf
        lcbs = means - bonus
        gap = maximum(ucbs) - minimum(lcbs)
        term = (gap < epsilon)
    end

    recommendations = Int.(recommendations)

    return(best, N, means, recommendations)
end
