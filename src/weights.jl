function dico_solve(f, x_min, x_max, delta = 1e-11)
# find m such that f(m)=0 using dichotomix search
    lower = x_min
    upper = x_max
    sgn = f(x_min)
    while (upper - lower) > delta
        middle = (upper + lower) / 2
        if f(middle) * sgn > 0
            lower = middle
        else
            upper = middle
        end
    end
    middle = (upper + lower) / 2
    return middle
end


# Computing the optimal weights
function big_i(alpha, mu1, mu2, dist)
    if (alpha == 0) | (alpha == 1)
        return 0
    else
        mid = alpha * mu1 + (1 - alpha) * mu2
        return alpha * d(mu1, mid, dist) + (1 - alpha) * d(mu2, mid, dist)
    end
end


muddle(mu1, mu2, nu1, nu2) = (nu1 * mu1 + nu2 * mu2) / (nu1 + nu2)


function cost(mu1, mu2, nu1, nu2, dist)
    if (nu1 == 0) & (nu2 == 0)
        return 0
    else
        alpha = nu1 / (nu1 + nu2)
        return ((nu1 + nu2) * big_i(alpha, mu1, mu2, dist))
    end
end


function xkofy(y, k, mu, dist, delta = 1e-11)
# return x_k(y), i.e. finds x such that g_k(x)=y
    g(x) = (1 + x) * cost(mu[1], mu[k], 1 / (1 + x), x / (1 + x), dist) - y
    x_max = 1
    while g(x_max) < 0
        x_max = 2 * x_max
    end
    return dico_solve(x -> g(x), 0, x_max, 1e-11)
end


function aux(y, mu, dist)
    # returns F_mu(y) - 1
    K = length(mu)
    x = [xkofy(y, k, mu, dist) for k = 2:K]
    m = [muddle(mu[1], mu[k], 1, x[k-1]) for k = 2:K]
    return (sum([d(mu[1], m[k-1], dist) / (d(mu[k], m[k-1], dist)) for k = 2:K]) - 1)
end


function one_step_opt(mu, dist, delta::Real = 1e-11)
    y_max = 0.5
    if d(mu[1], mu[2], dist) == Inf
        # find y_max such that aux(y_max, mu) > 0
        while aux(y_max, mu, dist) < 0
            y_max = y_max * 2
        end
    else
        y_max = d(mu[1], mu[2], dist)
    end
    y = dico_solve(y -> aux(y, mu, dist), 0, y_max, delta)
    x = [xkofy(y, k, mu, dist, delta) for k = 2:length(mu)]
    pushfirst!(x, 1)
    nu_optimal = x / sum(x)
    return nu_optimal[1] * y, nu_optimal
end


function optimal_weights(mu, dist, delta::Real = 1e-11)
# returns T*(mu) and w*(mu)
    num_arms = length(mu)
    maxs = (LinearIndices(mu .== maximum(mu)))[findall(mu .== maximum(mu))]
    num_maxs = length(maxs)
    if (num_maxs > 1)
        # multiple optimal arms
        v_optimal = zeros(1, num_arms)
        v_optimal[maxs] = 1 / num_maxs
        return 0, v_optimal
    else
        mu = vec(mu)
        index = sortperm(mu, rev = true)
        mu = mu[index]
        unsorted = vec(collect(1:num_arms))
        invindex = zeros(Int, num_arms)
        invindex[index] = unsorted
        # one-step optimization
        v_optimal, nu_optimal = one_step_opt(mu, dist, delta)
        # back to good ordering
        nu = nu_optimal[invindex]
        nu_optimal = zeros(1, num_arms)
        nu_optimal[1, :] = nu
        return v_optimal, nu_optimal
    end
end


# Computing the parameterized optimal weights
function c_k(x, k, mu, dist, beta::Real = 0.5)
    average = mu[1] * beta / (beta + x) + mu[k] * x / (beta + x)
    return beta * d(mu[1], average, dist) + x * d(mu[k], average, dist)
end


function inverse(y, k, mu, dist, beta::Real = 0.5, delta::Real = 1e-11)
# return x_k(y), i.e. finds x such that g_k(x)=y
    g(x) = c_k(x, k, mu, dist, beta) - y
    x_max = 1
    while g(x_max) < 0
        x_max = 2 * x_max
    end
    return dico_solve(x -> g(x), 0, x_max, delta)
end


function target(y, mu, dist, beta::Real = 0.5, delta::Real = 1e-11)
    K = length(mu)
    x = [inverse(y, k, mu, dist, beta, delta) for k = 2:K]
    return sum(x) - 1 + beta
end


function gamma_beta(mu, dist, beta::Real = 0.5, delta::Real = 1e-11)
    y_max = 0.5
    if beta * d(mu[1], mu[2], dist) == Inf
# find yMax such that aux(yMax, mu) > 0
        while target(y_max, mu, dist, beta, delta) < 0
            y_max = y_max * 2
        end
    else
        y_max = beta * d(mu[1], mu[2], dist)
    end
    gamma = dico_solve(y -> target(y, mu, dist, beta, delta), 0, y_max, delta)
    return gamma
end
