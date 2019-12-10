function ttts_c(
    mu::Array,
    delta::Real,
    rate::Function,
    dist::String,
    alpha::Real = 1,
    beta::Real = 1,
    frac::Real = 0.5,
    stopping::Symbol = :Chernoff,
)
    # Chernoff stopping rule combined with the PTS sampling rule
    condition = true
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
    # Initialization
    for a = 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
    end
    t = K
    Best = 1
    while (condition)
        Mu = S ./ N
        # Empirical best arm
        Best = randmax(Mu)
        # Compute the stopping statistic
        NB = N[Best]
        SB = S[Best]
        muB = SB / NB
        MuMid = (SB .+ S) ./ (NB .+ N)
        Index = collect(1:K)
        deleteat!(Index, Best)
        Score = minimum([NB * d(muB, MuMid[i], dist) + N[i] * d(Mu[i], MuMid[i], dist) for i in Index])
        if (Score > rate(t, delta))
            # stop
            condition = false
        elseif (t > 1e7)
            condition = false
            Best = 0
            println(N)
            println(S)
            N = zeros(1, K)
        else
            TS = zeros(K)
            for a = 1:K
                if dist == "Gaussian"
                    TS[a] = rand(Normal(S[a] / N[a], alpha / sqrt(N[a])), 1)[1]
                elseif dist == "Bernoulli"
                    TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
                end
            end
            I = argmax(TS)
            if (rand() > frac)
                J = I
                condition = true
                while (I == J)
                    TS = zeros(K)
                    for a = 1:K
                        if dist == "Gaussian"
                            TS[a] = rand(Normal(S[a] / N[a], alpha / sqrt(N[a])), 1)[1]
                        elseif dist == "Bernoulli"
                            TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
                        end
                    end
                    J = argmax(TS)
                end
                I = J
            end

            # draw arm I
            t += 1
            S[I] += sample_arm(mu[I], dist)
            N[I] += 1
        end
    end
    recommendation = Best
    return (recommendation, N)
end


function ttts_optimal(
    mu::Array,
    delta::Real,
    rate::Function,
    dist::String,
    alpha::Real = 1,
    beta::Real = 1,
    stopping::Symbol = :Chernoff,
)
    # Chernoff stopping rule combined with the PTS sampling rule
    condition = true
    K = length(mu)
    N = zeros(1, K)
    S = zeros(1, K)
    # Initialization
    for a = 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
    end
    t = K
    Best = 1
    while (condition)
        Mu = S ./ N
        # Empirical best arm
        Best = randmax(Mu)
        # Compute the stopping statistic
        NB = N[Best]
        SB = S[Best]
        muB = SB / NB
        MuMid = (SB .+ S) ./ (NB .+ N)
        Index = collect(1:K)
        deleteat!(Index, Best)
        Score = minimum([NB * d(muB, MuMid[i], dist) + N[i] * d(Mu[i], MuMid[i], dist) for i in Index])
        if (Score > rate(t, delta))
            # stop
            condition = false
        elseif (t > 1e7)
            condition = false
            Best = 0
            println(N)
            println(S)
            N = zeros(1, K)
        else
            TS = zeros(K)
            for a = 1:K
                if dist == "Gaussian"
                    TS[a] = rand(Normal(S[a] / N[a], alpha / sqrt(N[a])), 1)[1]
                elseif dist == "Bernoulli"
                    TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
                end
            end
            ts_best = argmax(TS)

            challenger = ts_best
            condition = true
            while (challenger == ts_best)
                TS = zeros(K)
                for a = 1:K
                    if dist == "Gaussian"
                        TS[a] = rand(Normal(S[a] / N[a], alpha / sqrt(N[a])), 1)[1]
                    elseif dist == "Bernoulli"
                        TS[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
                    end
                end
                challenger = argmax(TS)
            end

            if (N[ts_best] > N[challenger])
                I = challenger
            else
                I = ts_best
            end

            # Draw the arm
            t += 1
            S[I] += sample_arm(mu[I], dist)
            N[I] += 1
        end
    end
    recommendation = Best
    return (recommendation, N)
end
