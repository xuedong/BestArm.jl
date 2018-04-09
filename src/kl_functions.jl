# Bernoulli distributions
function dBernoulli(p::Float64, q::Float64)
    res = 0
    if (p != q)
        if (p <= 0) p = eps() end
        if (p >= 1) p = 1-eps() end
        res = (p * log(p/q) + (1-p) * log((1-p)/(1-q)))
    end
    return(res)
end


function dupBernoulli(p::Float64, level::Float64)
    # KL upper confidence bound:
    # return qM>p such that d(p,qM)=level
    lM = p
    uM = min(min(1, p + sqrt(level/2)), 1)
    for j = 1:16
        qM = (uM + lM)/2
        if dBernoulli(p, qM) > level
            uM = qM
        else
            lM = qM
        end
    end
    return(uM)
end


function dlowBernoulli(p::Float64, level::Float64)
    # KL lower confidence bound:
    # return lM<p such that d(p,lM)=level
    lM = max(min(1, p - sqrt(level/2)), 0)
    uM = p
    for j = 1:16
        qM = (uM + lM)/2
        if dBernoulli(p, qM) > level
            lM = qM
        else
            uM = qM
        end
    end
    return(lM)
end


# Poisson distributions
function dPoisson(p::Float64, q::Float64)
    if (p == 0)
        res = q
    else
        res = q - p + p * log(p/q)
    end
    return(res)
end


function dupPoisson(p::Float64, level::Float64)
    # KL upper confidence bound: generic way
    # return qM>p such that d(p,qM)=level
    lM = p
    # finding an upper bound
    uM = max(2 * p, 1)
    while (dPoisson(p, uM) < level)
        uM = 2 * uM
    end
    for j = 1:16
        qM = (uM + lM)/2
        if dPoisson(p, qM) > level
            uM = qM
        else
            lM = qM
        end
    end
    return(uM)
end


function dlowPoisson(p::Float64, level::Float64)
    # KL lower confidence bound: generic way
    # return lM<p such that d(p,lM)=level
    # finding a lower bound
    lM = p / 2
    if p != 0
        while (dPoisson(p, lM) < level)
            lM = lM/2
        end
    end
    uM = p
    for j = 1:16
        qM = (uM + lM)/2
        if dPoisson(p, qM) > level
            lM = qM
        else
            uM = qM
        end
    end
    return(lM)
end


# Exponential distributions
function dExpo(p::Float64, q::Float64)
    res = 0
    if (p != q)
        if (p <= 0) | (q <= 0)
            res = Inf
        else
            res = p/q - 1 - log(p/q)
        end
    end
    return(res)
end


function dupExpo(p::Float64, level::Float64)
    # KL upper confidence bound: generic way
    # return qM>p such that d(p,qM)=level
    lM = p
    # finding an upper bound
    uM = max(2 * p, 1)
    while (dExpo(p, uM) < level)
        uM = 2 * uM
    end
    for j = 1:16
        qM = (uM + lM)/2
        if dExpo(p, qM) > level
            uM = qM
        else
            lM = qM
        end
    end
    return(uM)
end


function dlowExpo(p::Float64, level::Float64)
    # KL lower confidence bound: generic way
    # return lM<p such that d(p,lM)=level
    # finding a lower bound
    lM = p/2
    if p != 0
        while (dExpo(p, lM) < level)
            lM = lM/2
        end
    end
    uM = p
    for j = 1:16
        qM = (uM + lM)/2;
        if dExpo(p, qM) > level
            lM = qM;
        else
            uM = qM;
        end
    end
    return(lM)
end


# Gaussian distributions
function dGaussian(p::Float64, q::Float64, sigma::Float64 = 1.0)
    (p - q)^2 / (2 * sigma^2)
end


function dupGaussian(p::Float64, level::Float64, sigma::Float64 = 1.0)
    p + sigma * sqrt(2 * level)
end


function dlowGaussian(p::Float64, level::Float64, sigma::Float64 = 1.0)
    p - sigma * sqrt(2 * level)
end
