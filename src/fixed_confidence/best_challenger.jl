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
    # Chernoff stopping rule, sampling based on the "best challenger"
    # described in experimental section of Garivier and Kaufmann [2016]
    continuing = true
    K=length(mu)
    N = zeros(1,K)
    S = zeros(1,K)
    # initialization
    for a in 1:K
        N[a] = 1
        S[a] = sample_arm(mu[a], dist)
    end
    t=K
    TrueBest=1
    while continuing
        Mu=S./N
        # Empirical best arm
        Best=randmax(Mu)
        TrueBest=randmax(Mu)
        # Compute the stopping statistic
        NB=N[Best]
        SB=S[Best]
        MuB=SB/NB
        MuMid=(SB.+S)./(NB.+N)
        Score=minimum([NB*d(MuB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist) for i in 1:K if i!=Best])
        # compute the best arm and the challenger
        if ts
            for a=1:K
                if typeDistribution == "Gaussian"
                    Mu[a] = rand(Normal(S[a] / N[a], alpha / sqrt(N[a])), 1)[1]
                elseif typeDistribution == "Bernoulli"
                    Mu[a] = rand(Beta(alpha + S[a], beta + N[a] - S[a]), 1)[1]
                end
            end
            Best=randmax(Mu)
            NB=N[Best]
            MuB=Mu[Best]
            MuMid=(NB*MuB.+N.*Mu)./(NB.+N)
        end
        Challenger=1
        NewScore=Inf
        for i=1:K
            if i!=Best
                score=NB*d(MuB, MuMid[i], dist)+N[i]*d(Mu[i], MuMid[i], dist)
                if (score<NewScore)
                    Challenger=i
                    NewScore=score
                end
            end
        end
        I = 1
        if (Score > rate(t,delta))
            # stop
            continuing=false
        elseif (t >1e7)
            # stop and return (0,0)
            continuing=false
            TrueBest=0
            print(N)
            print(S)
            N=zeros(1,K)
        else
            # continue and sample an arm
    	    if (fe)&&(minimum(N) <= max(sqrt(t) - K/2,0))
                # forced exploration
                I=randmax(-N)
            else
    			if challenger == :Proportion
     				I = (NB/(NB+N[Challenger]) < Dist[Best]/(Dist[Best]+Dist[Challenger])) ? Best : Challenger
     			elseif challenger == :Transportation
     				I = (d(MuB, MuMid[Challenger], dist) > d(Mu[Challenger], MuMid[Challenger], dist)) ? Best : Challenger
     			elseif challenger == :Pull
     				I = (N[Best] < N[Challenger]) ? Best : Challenger
     			end
            end
        end
        # draw the arm
        t += 1
        S[I] += sample_arm(mu[I], dist)
        N[I] += 1
    end
    return TrueBest,N
end
